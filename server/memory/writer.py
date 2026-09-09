"""The write path: one daemon thread owns every embedding and DB write.

The hot path (orchestrator, ASR thread, WS reader) only ever enqueues. Nothing
in a live turn may block on an embedder or on SQLite — an LLM-mediated write
path costs seconds per item and would stall captions or TTS.

Frame dedup is a cheap cascade, ordered by cost: a wall-clock throttle on the
hot path, then a 64-bit difference hash, then descriptor cosine. A static scene
still leaves one keyframe every `memory_keyframe_force_s` so "what was on the
table earlier" has something to hit.

Queue overflow drops FRAMES first and never an utterance — losing a spoken turn
from memory is a real regression; losing one of many near-identical frames is
not.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..config import Settings
from ..logging_conf import get_logger
from . import embed as embed_mod
from .store import KIND_FRAME, KIND_UTTERANCE, SPACE_IMAGE, SPACE_TEXT, MemoryStore

log = get_logger(__name__)


@dataclass
class _FrameState:
    last_hash: Optional[int] = None
    last_vec: Optional[np.ndarray] = None
    last_kept_at: float = 0.0


class MemoryWriter:
    """Shared across sessions; every job carries its own conversation_id."""

    def __init__(self, settings: Settings, store: MemoryStore, *, media: Any = None,
                 text_embedder: Any = None, image_embedder: Any = None) -> None:
        self.settings = settings
        self.store = store
        self.media = media
        self.text = text_embedder or embed_mod.build_text_embedder(settings)
        self.image = image_embedder or embed_mod.build_image_embedder(settings)
        self._q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=512)
        self._thread: Optional[threading.Thread] = None
        self._frames: Dict[str, _FrameState] = {}
        self._stopping = False
        self.stats = {"utterances": 0, "frames_kept": 0, "frames_skipped": 0, "dropped": 0}

    # ---- lifecycle ----

    def start(self) -> None:
        if self._thread is not None:
            return
        self.store.open()
        self._thread = threading.Thread(target=self._run, name="memory-writer", daemon=True)
        self._thread.start()
        log.info("memory writer started (text=%s dim=%s, image=%s dim=%s)",
                 getattr(self.text, "name", "?"), getattr(self.text, "dim", "?"),
                 getattr(self.image, "name", "?"), getattr(self.image, "dim", "?"))

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._stopping = True
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)
        self._thread = None

    # ---- producers (hot path: enqueue only) ----

    def _put(self, job: Dict[str, Any], *, droppable: bool) -> None:
        if self._stopping:
            return
        try:
            self._q.put_nowait(job)
        except queue.Full:
            self.stats["dropped"] += 1
            if not droppable:
                # make room by discarding one droppable job, then retry once
                try:
                    self._q.get_nowait()
                    self._q.put_nowait(job)
                except (queue.Empty, queue.Full):
                    pass

    def note_utterance(self, conversation_id: str, role: str, text: str, *, lang: str,
                       session_ts: Optional[float] = None, media_ts: Optional[float] = None,
                       importance: float = 0.5) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._put({"t": "utterance", "conv": conversation_id, "role": role, "text": text,
                   "lang": lang, "session_ts": session_ts, "media_ts": media_ts,
                   "importance": importance}, droppable=False)

    def note_frame(self, conversation_id: str, jpeg: bytes, *, session_ts: Optional[float] = None,
                   media_ts: Optional[float] = None, lang: str = "zh") -> None:
        if not jpeg:
            return
        self._put({"t": "frame", "conv": conversation_id, "jpeg": jpeg, "lang": lang,
                   "session_ts": session_ts, "media_ts": media_ts}, droppable=True)

    # ---- consumer ----

    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                break
            try:
                self._handle(job)
            except Exception as exc:  # noqa: BLE001 — memory must never kill a session
                log.warning("memory writer job %s failed: %s", job.get("t"), exc)
            finally:
                self._q.task_done()

    def _handle(self, job: Dict[str, Any]) -> None:
        kind = job.get("t")
        if kind == "utterance":
            self._handle_utterance(job)
        elif kind == "frame":
            self._handle_frame(job)

    def _handle_utterance(self, job: Dict[str, Any]) -> None:
        conv, text = job["conv"], job["text"]
        item_id = self.store.add_item(
            conv, KIND_UTTERANCE, text=text, role=job.get("role"), lang=job.get("lang"),
            session_ts=job.get("session_ts"), media_ts=job.get("media_ts"),
            importance=float(job.get("importance", 0.5)))
        vec = self.text.encode([text])[0]
        self.store.add_vector(conv, item_id, SPACE_TEXT, vec)
        # late interaction: token matrices alongside the pooled vector. Gated on
        # the capability, not the config flag — the first call lazily loads the
        # colbert head and flips `supports_late`; a missing head degrades to
        # pooled-only retrieval with no other change.
        encode_tokens = getattr(self.text, "encode_tokens", None)
        if self.settings.memory_late_interaction and callable(encode_tokens):
            try:
                self.store.add_vector_late(conv, item_id, encode_tokens([text])[0])
            except Exception as exc:  # noqa: BLE001 — pooled retrieval still works
                log.debug("memory: late-interaction encode failed (%s)", exc)
        self.stats["utterances"] += 1

    def _handle_frame(self, job: Dict[str, Any]) -> None:
        conv, jpeg = job["conv"], job["jpeg"]
        state = self._frames.setdefault(conv, _FrameState())
        now = time.monotonic()
        forced = (now - state.last_kept_at) >= max(1.0, float(self.settings.memory_keyframe_force_s))

        digest = embed_mod.dhash(jpeg)
        if not forced and embed_mod.hamming(digest, state.last_hash) <= 6:
            self.stats["frames_skipped"] += 1
            return
        vec = self.image.encode_images([jpeg])[0]
        if (not forced and state.last_vec is not None
                and float(vec @ state.last_vec) >= float(self.settings.memory_keyframe_sim_threshold)):
            self.stats["frames_skipped"] += 1
            return

        media_hash = None
        if self.media is not None:
            try:
                media_hash = self.media.put_bytes(jpeg, orig_name="keyframe.jpg").get("hash")
            except Exception as exc:  # noqa: BLE001 — CAS rejection must not lose the vector
                log.debug("memory: keyframe not stored in CAS (%s)", exc)
        item_id = self.store.add_item(
            conv, KIND_FRAME, text="", lang=job.get("lang"), session_ts=job.get("session_ts"),
            media_ts=job.get("media_ts"), media_hash=media_hash, importance=0.4)
        self.store.add_vector(conv, item_id, SPACE_IMAGE, vec)
        state.last_hash = digest
        state.last_vec = vec
        state.last_kept_at = now
        self.stats["frames_kept"] += 1

    # ---- test/manual helper ----

    def warmup(self) -> None:
        """Eagerly load the lazy embedders. Called from the app lifespan
        (off-loop, before "ready"): without it the FIRST live turn pays the
        full BGE-M3/Chinese-CLIP weight load on the recall path. The hashing/
        descriptor fallbacks make this a no-op; every failure degrades to the
        normal lazy path."""
        try:
            self.text.encode(["memory warmup"])
        except Exception as exc:  # noqa: BLE001
            log.debug("memory warmup: text embedder not preloaded (%s)", exc)
        encode_tokens = getattr(self.text, "encode_tokens", None)
        if self.settings.memory_late_interaction and callable(encode_tokens):
            try:
                encode_tokens(["memory warmup"])  # loads the colbert head too
            except Exception as exc:  # noqa: BLE001 — late lane is optional
                log.debug("memory warmup: late interaction not preloaded (%s)", exc)
        encode_texts = getattr(self.image, "encode_texts", None)
        if callable(encode_texts):
            try:
                encode_texts(["memory warmup"])
            except Exception as exc:  # noqa: BLE001
                log.debug("memory warmup: image embedder not preloaded (%s)", exc)

    def drain(self, timeout: float = 10.0) -> None:
        """Block until the queue is empty (tests + manual probing only)."""
        deadline = time.monotonic() + timeout
        while not self._q.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        self._q.join()

    def forget(self, conversation_id: str) -> None:
        self._frames.pop(conversation_id, None)
