# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bounded pinned storage for fine-grained activation offloading."""

import ctypes
import logging
import math
import time
import weakref
import threading
from dataclasses import dataclass, field

import torch

from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


def _cuda_runtime():
    """Return the cuda-python runtime bindings."""
    try:
        import cuda.bindings.runtime as cuda_runtime  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError("A fixed pinned activation buffer requires cuda-python") from error
    return cuda_runtime


def _free_host(address: int, device: int) -> None:
    """Free a cudaHostAlloc allocation with the allocating device current.

    Garbage collection may run this on any thread; selecting the device keeps the
    runtime from creating a context on another GPU.
    """
    cuda_runtime = _cuda_runtime()
    with torch.cuda.device(device):
        (error,) = cuda_runtime.cudaFreeHost(address)
    if error != cuda_runtime.cudaError_t.cudaSuccess:
        logger.warning("cudaFreeHost of the pinned activation buffer failed: %s", error)


class PinnedActivationBuffer:
    """Pack saved activations into one fixed, reusable pinned buffer.

    Allocations append within the buffer. Once all backups have been reloaded,
    the next allocation reuses the buffer after waiting for its H2D readers.
    Interleaved chunks share one budget; storage is not reclaimed until all
    outstanding backups are consumed. No individual-tensor free list is needed.

    Captured addresses remain reserved for this object's lifetime: replay does
    not execute Python allocation/release bookkeeping. Each capture owns a
    separate range, reusable within that capture after its readers finish.
    Eager epochs use the remaining tail and cannot overwrite graph backups.

    Capacity may be any positive multiple of 256 bytes and is pinned at that size:
    the storage comes from cudaHostAlloc directly rather than from the PyTorch
    pinned host allocator, which would round the request up to the next power
    of two. Storage is allocated at construction and freed once this object and
    every view of it are gone. Captured graphs hold raw addresses, so keep this
    object alive while any graph that uses it can replay. Exhaustion raises an
    error; the buffer never grows.
    """

    ALIGNMENT = 256

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes < self.ALIGNMENT or capacity_bytes % self.ALIGNMENT:
            raise ValueError(
                "Pinned activation buffer capacity must be a positive multiple of 256 bytes"
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Create the pinned activation buffer before CUDA graph capture")
        self.capacity_bytes = capacity_bytes
        self.peak_live_bytes = 0
        self.live_bytes = 0
        self._storage = self._allocate_storage(capacity_bytes)
        self._typed_storage: dict[torch.dtype, torch.Tensor] = {}
        self._offset = 0
        self.graph_reserved_bytes = 0
        self._capture_id: int | None = None
        self._capture_start = 0
        self._live: dict[int, int] = {}
        self._reader_streams: dict[int, torch.cuda.Stream] = {}
        self._pending_events: list[torch.cuda.Event] = []
        self._captured_events: list[torch.cuda.Event] = []
        self._writers_waited: set[int] = set()

    def _allocate_storage(self, capacity_bytes: int) -> torch.Tensor:
        """Pin exactly ``capacity_bytes`` (rounded up to whole pages) with cudaHostAlloc.

        This is the same page-locked, mapped allocation the PyTorch pinned host
        allocator makes, without its power-of-two rounding. ``torch.frombuffer`` keeps
        the backing object alive while any view of the storage exists, so its finalizer
        frees the pages only after the last view is gone. The finalizer does not run at
        interpreter exit: cudaFreeHost synchronizes the device, and process exit
        releases the pages anyway.
        """
        cuda_runtime = _cuda_runtime()
        started = time.perf_counter()
        error, address = cuda_runtime.cudaHostAlloc(
            capacity_bytes, cuda_runtime.cudaHostAllocDefault
        )
        if error != cuda_runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(
                f"cudaHostAlloc of {capacity_bytes} bytes for the pinned activation "
                f"buffer failed: {error}"
            )
        memory = (ctypes.c_uint8 * capacity_bytes).from_address(address)
        self._free = weakref.finalize(memory, _free_host, address, torch.cuda.current_device())
        self._free.atexit = False
        log_single_rank(
            logger,
            logging.INFO,
            f"Pinned {capacity_bytes / 2**30:.2f} GiB for the activation offload buffer in "
            f"{time.perf_counter() - started:.1f} s",
        )
        return torch.frombuffer(memory, dtype=torch.uint8)

    def allocate(
        self, shape: tuple, dtype: torch.dtype, *, stream: torch.cuda.Stream | None = None
    ) -> torch.Tensor:
        """Allocate a view whose first write uses the current stream.

        Pass the already-active stream to avoid rediscovering it for every tensor.
        This does not switch streams; the caller still enqueues the copy on it.
        """
        if stream is None:
            stream = torch.cuda.current_stream(torch.cuda.current_device())
        capturing = torch.cuda.is_current_stream_capturing()
        epoch_start = self.graph_reserved_bytes
        if capturing:
            cuda_runtime = _cuda_runtime()
            error, _, capture_id, *_ = cuda_runtime.cudaStreamGetCaptureInfo(stream.cuda_stream)
            if error != cuda_runtime.cudaError_t.cudaSuccess:
                raise RuntimeError(f"Cannot identify pinned buffer capture: {error}")
            if capture_id != self._capture_id:
                self._capture_id = capture_id
                self._capture_start = max(epoch_start, self._offset if self._live else 0)
            epoch_start = self._capture_start
        if not self._live:
            self._writers_waited.clear()
            self._offset = epoch_start
        else:
            self._offset = max(self._offset, epoch_start)
        if stream.cuda_stream not in self._writers_waited:
            if capturing:
                # External event nodes retain CUDA handles, not Python owners.
                # Keep every captured wait/record handle alive with the buffer.
                self._captured_events.extend(self._pending_events)
            for event in self._pending_events:
                stream.wait_event(event)
            self._writers_waited.add(stream.cuda_stream)
        itemsize = dtype.itemsize
        nbytes = math.prod(shape) * itemsize
        aligned = ((nbytes + self.ALIGNMENT - 1) & -self.ALIGNMENT) or self.ALIGNMENT
        if self.capacity_bytes - self._offset < aligned:
            raise RuntimeError(
                "Pinned activation buffer budget exceeded: "
                f"request={aligned} bytes, capacity={self.capacity_bytes} bytes, "
                f"live={self.live_bytes} bytes, "
                f"graph_reserved={self.graph_reserved_bytes} bytes, "
                f"unused_tail={self.capacity_bytes - self._offset} bytes. "
                "Increase fine_grained_offloading_buffer_size_gib or reduce offloaded activations."
            )
        start = self._offset
        typed_storage = self._typed_storage.get(dtype)
        if typed_storage is None:
            typed_storage = self._storage.view(dtype)
            self._typed_storage[dtype] = typed_storage
        strides = [1] * len(shape)
        for axis in range(len(shape) - 2, -1, -1):
            strides[axis] = strides[axis + 1] * max(shape[axis + 1], 1)
        tensor = typed_storage.as_strided(shape, strides, start // itemsize)
        self._offset += aligned
        if capturing:
            self.graph_reserved_bytes = max(self.graph_reserved_bytes, self._offset)
        self._live[id(tensor)] = nbytes
        self.live_bytes += nbytes
        self.peak_live_bytes = max(self.peak_live_bytes, self.live_bytes)
        return tensor

    def release(self, tensor: torch.Tensor, *, stream: torch.cuda.Stream | None = None) -> None:
        """Retire a backup after its final H2D read is enqueued on the active stream."""
        if torch.cuda.is_current_stream_capturing():
            # Even a backup allocated eagerly can be read by a captured graph.
            # Capture retires its logical use, not the graph's address ownership.
            end = (
                tensor.storage_offset() * tensor.element_size()
                + tensor.numel() * tensor.element_size()
            )
            end = (end + self.ALIGNMENT - 1) & -self.ALIGNMENT
            self.graph_reserved_bytes = max(self.graph_reserved_bytes, end)
        self.live_bytes -= self._live.pop(id(tensor))
        if stream is None:
            stream = torch.cuda.current_stream(torch.cuda.current_device())
        self._reader_streams[stream.cuda_stream] = stream
        if not self._live:
            # Events protect reuse across streams and across iterations. External
            # events also remain observable after a captured graph is replayed.
            self._pending_events = []
            for reader in self._reader_streams.values():
                event = torch.cuda.Event(external=True)
                event.record(reader)
                self._pending_events.append(event)
                if torch.cuda.is_current_stream_capturing():
                    self._captured_events.append(event)
            self._reader_streams.clear()


@dataclass
class _EventSafeBlock:
    """A byte range managed by the event-safe eager allocator."""

    start: int
    size: int
    logical_bytes: int = 0
    tensor_id: int | None = None
    events: list = field(default_factory=list)
    graph_owned: bool = False

    @property
    def end(self) -> int:
        return self.start + self.size


class _EventSafePinnedActivationBuffer:
    """Pinned arena with best-fit reuse and a separate graph capture arena."""

    ALIGNMENT = 256

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes < self.ALIGNMENT or capacity_bytes % self.ALIGNMENT:
            raise ValueError(
                "Pinned activation buffer capacity must be a positive multiple of 256 bytes"
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Create the pinned activation buffer before CUDA graph capture")
        self.capacity_bytes = capacity_bytes
        self.peak_live_bytes = 0
        self.live_bytes = 0
        self._storage = self._allocate_storage(capacity_bytes)
        self._state_lock = threading.RLock()
        self._typed_storage: dict[torch.dtype, torch.Tensor] = {}
        self._free_blocks = [_EventSafeBlock(0, capacity_bytes)]
        self._allocations: dict[int, _EventSafeBlock] = {}
        self._stats = {
            "allocation_requests": 0,
            "release_requests": 0,
            "eager_reuses": 0,
            "graph_allocations": 0,
            "split_operations": 0,
            "coalesce_operations": 0,
            "event_waits": 0,
        }
        self._high_watermark = 0
        self._offset = 0
        self.graph_reserved_bytes = 0
        self._capture_id: int | None = None
        self._capture_start = 0
        self._capture_offset = 0
        self._captured_events: list[torch.cuda.Event] = []

    def _allocate_storage(self, capacity_bytes: int) -> torch.Tensor:
        cuda_runtime = _cuda_runtime()
        started = time.perf_counter()
        error, address = cuda_runtime.cudaHostAlloc(
            capacity_bytes, cuda_runtime.cudaHostAllocDefault
        )
        if error != cuda_runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(
                f"cudaHostAlloc of {capacity_bytes} bytes for the pinned activation buffer failed: "
                f"{error}"
            )
        memory = (ctypes.c_uint8 * capacity_bytes).from_address(address)
        self._free = weakref.finalize(memory, _free_host, address, torch.cuda.current_device())
        self._free.atexit = False
        log_single_rank(
            logger,
            logging.INFO,
            f"Pinned {capacity_bytes / 2**30:.2f} GiB for the activation offload buffer in "
            f"{time.perf_counter() - started:.1f} s",
        )
        return torch.frombuffer(memory, dtype=torch.uint8)

    def _align(self, nbytes: int) -> int:
        return ((nbytes + self.ALIGNMENT - 1) & -self.ALIGNMENT) or self.ALIGNMENT

    def _view(self, start: int, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        itemsize = dtype.itemsize
        typed_storage = self._typed_storage.get(dtype)
        if typed_storage is None:
            typed_storage = self._storage.view(dtype)
            self._typed_storage[dtype] = typed_storage
        strides = [1] * len(shape)
        for axis in range(len(shape) - 2, -1, -1):
            strides[axis] = strides[axis + 1] * max(shape[axis + 1], 1)
        return typed_storage.as_strided(shape, strides, start // itemsize)

    def _reclaim_completed_events(self) -> None:
        for block in self._free_blocks:
            if block.events:
                block.events = [event for event in block.events if not event.query()]

    def _insert_free(self, block: _EventSafeBlock) -> None:
        blocks = self._free_blocks + [block]
        blocks.sort(key=lambda item: item.start)
        merged = []
        for current in blocks:
            if merged and merged[-1].end == current.start:
                previous = merged[-1]
                previous.size += current.size
                previous.events.extend(current.events)
                self._stats["coalesce_operations"] += 1
            else:
                merged.append(current)
        self._free_blocks = merged

    def _reserve_graph_prefix(self, end: int) -> None:
        if end <= self.graph_reserved_bytes:
            return
        self.graph_reserved_bytes = end
        retained = []
        for block in self._free_blocks:
            if block.end <= end:
                continue
            if block.start < end:
                block = _EventSafeBlock(end, block.end - end, events=list(block.events))
            retained.append(block)
        self._free_blocks = retained

    def _find_best_fit(self, required: int):
        candidates = [block for block in self._free_blocks if block.size >= required]
        return min(candidates, key=lambda block: (block.size, block.start), default=None)

    def _capacity_error(self, required: int, capturing: bool) -> RuntimeError:
        self._reclaim_completed_events()
        free_bytes = sum(block.size for block in self._free_blocks)
        largest = max((block.size for block in self._free_blocks), default=0)
        retired = sum(block.size for block in self._free_blocks if block.events)
        pending = sum(len(block.events) for block in self._free_blocks)
        return RuntimeError(
            "Pinned activation buffer budget exceeded: "
            f"request={required} bytes, capacity={self.capacity_bytes} bytes, "
            f"live={self.live_bytes} bytes, graph_reserved={self.graph_reserved_bytes} bytes, "
            f"free={free_bytes} bytes, largest_free_block={largest} bytes, "
            f"retired={retired} bytes, pending_events={pending}, "
            f"high_watermark={self._high_watermark} bytes, "
            f"unused_tail={max(0, self.capacity_bytes - self._high_watermark)} bytes, "
            f"capture={capturing}. Increase fine_grained_offloading_buffer_size_gib, "
            "reduce offloaded activations, or reduce the number of outstanding microbatches."
        )

    def _allocate_eager(self, shape, dtype, nbytes, aligned, stream):
        self._reclaim_completed_events()
        block = self._find_best_fit(aligned)
        if block is None:
            raise self._capacity_error(aligned, capturing=False)
        self._free_blocks.remove(block)
        for event in block.events:
            stream.wait_event(event)
            self._stats["event_waits"] += 1
        if block.size > aligned:
            self._stats["split_operations"] += 1
            self._insert_free(
                _EventSafeBlock(block.start + aligned, block.size - aligned, events=list(block.events))
            )
        block.size = aligned
        block.logical_bytes = nbytes
        block.tensor_id = None
        block.events = []
        block.graph_owned = False
        tensor = self._view(block.start, shape, dtype)
        block.tensor_id = id(tensor)
        self._allocations[id(tensor)] = block
        self._high_watermark = max(self._high_watermark, block.end)
        self._offset = self._high_watermark
        return tensor

    def _capture_info(self, stream) -> None:
        cuda_runtime = _cuda_runtime()
        error, _, capture_id, *_ = cuda_runtime.cudaStreamGetCaptureInfo(stream.cuda_stream)
        if error != cuda_runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cannot identify pinned buffer capture: {error}")
        if capture_id != self._capture_id:
            self._capture_id = capture_id
            self._capture_start = max(
                self.graph_reserved_bytes,
                self._high_watermark if self._allocations else self.graph_reserved_bytes,
            )
            self._capture_offset = self._capture_start
            # A graph capture may start in a retired eager block. Preserve the
            # same dependency that eager reuse uses before the graph writes it.
            for block in self._free_blocks:
                if block.end > self._capture_start:
                    for event in block.events:
                        stream.wait_event(event)
                        self._captured_events.append(event)

    def allocate(self, shape: tuple, dtype: torch.dtype, *, stream=None) -> torch.Tensor:
        with self._state_lock:
            return self._allocate_impl(shape, dtype, stream=stream)

    def _allocate_impl(self, shape: tuple, dtype: torch.dtype, *, stream=None) -> torch.Tensor:
        self._stats["allocation_requests"] += 1
        if stream is None:
            stream = torch.cuda.current_stream(torch.cuda.current_device())
        capturing = torch.cuda.is_current_stream_capturing()
        nbytes = math.prod(shape) * dtype.itemsize
        aligned = self._align(nbytes)
        if capturing:
            self._capture_info(stream)
            if self.capacity_bytes - self._capture_offset < aligned:
                raise self._capacity_error(aligned, capturing=True)
            start = self._capture_offset
            self._capture_offset += aligned
            self._high_watermark = max(self._high_watermark, self._capture_offset)
            self._offset = self._high_watermark
            self._reserve_graph_prefix(self._capture_offset)
            tensor = self._view(start, shape, dtype)
            self._allocations[id(tensor)] = _EventSafeBlock(
                start, aligned, nbytes, id(tensor), graph_owned=True
            )
            self._stats["graph_allocations"] += 1
        else:
            tensor = self._allocate_eager(shape, dtype, nbytes, aligned, stream)
            self._stats["eager_reuses"] += 1
        self.live_bytes += nbytes
        self.peak_live_bytes = max(self.peak_live_bytes, self.live_bytes)
        return tensor

    def release(self, tensor: torch.Tensor, *, stream=None) -> None:
        with self._state_lock:
            self._release_impl(tensor, stream=stream)

    def _release_impl(self, tensor: torch.Tensor, *, stream=None) -> None:
        self._stats["release_requests"] += 1
        block = self._allocations.pop(id(tensor), None)
        if block is None:
            raise ValueError("Attempting to release a tensor not allocated by this buffer")
        self.live_bytes -= block.logical_bytes
        if torch.cuda.is_current_stream_capturing():
            # Captured replay does not execute Python release bookkeeping.
            self._reserve_graph_prefix(max(self.graph_reserved_bytes, block.end))
            return
        if stream is None:
            stream = torch.cuda.current_stream(torch.cuda.current_device())
        try:
            event = torch.cuda.Event(external=True)
        except TypeError:
            event = torch.cuda.Event()
        event.record(stream)
        block.tensor_id = None
        block.logical_bytes = 0
        block.events = [event]
        self._insert_free(block)

    def get_diagnostics(self) -> dict:
        with self._state_lock:
            return self._get_diagnostics_impl()

    def _get_diagnostics_impl(self) -> dict:
        """Return allocator diagnostics without synchronizing CUDA streams."""
        self._reclaim_completed_events()
        free_bytes = sum(block.size for block in self._free_blocks)
        largest = max((block.size for block in self._free_blocks), default=0)
        retired = sum(block.size for block in self._free_blocks if block.events)
        return {
            "stats": dict(self._stats),
            "capacity_bytes": self.capacity_bytes,
            "live_bytes": self.live_bytes,
            "peak_live_bytes": self.peak_live_bytes,
            "graph_reserved_bytes": self.graph_reserved_bytes,
            "free_bytes": free_bytes,
            "largest_free_block_bytes": largest,
            "retired_bytes": retired,
            "pending_event_count": sum(len(block.events) for block in self._free_blocks),
            "free_block_count": len(self._free_blocks),
            "allocation_count": len(self._allocations),
            "high_watermark_bytes": self._high_watermark,
            "unused_tail_bytes": max(0, self.capacity_bytes - self._high_watermark),
        }
        

PinnedActivationBuffer = _EventSafePinnedActivationBuffer