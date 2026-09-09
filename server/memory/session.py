"""Per-session memory facade — the only object the orchestrator talks to.

Everything session-scoped lives here and dies with the session: the injected-id
table, the prefetch cache, the rolling conversation language, the keyframe
throttle. The shared, process-wide pieces (store, writer, embedders) are passed
in. Retrieval is always scoped to this session's `conversation_id`, so one
session can never read another's memories.

Two-lane read path: `prefetch()` runs off the ASR partial thread and parks
ranked candidates in a cache (gold evidence is usually retrievable well before
the user stops talking); `recall_for_turn()` then only has to re-gate against
the final transcript, so the turn itself pays almost nothing.

Re-injection (design §5): injected items STAY in retrieval and ranking —
excluding them up front made an explicit re-request ("不是那个,我是说之前那台相机")
silently unanswerable. They are dropped only AFTER ranking, by the distance
gate (fewer than `memory_reinject_distance` text-KV tokens since last shown, or
`memory_reinject_max_copies` copies already injected). Positions are the
worker's exact text-token count when the orchestrator plumbs it through, else
an internal running estimate of noted turns + injected blocks.
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..config import Settings
from ..logging_conf import get_logger
from . import inject as inject_mod
from . import lang as lang_mod
from . import rewrite as rewrite_mod
from .retrieval import Candidate, Retriever
from .store import KIND_FRAME, KIND_UTTERANCE, MemoryStore
from .writer import MemoryWriter

log = get_logger(__name__)


@dataclass
class RecallResult:
    block: str
    items: List[Dict[str, Any]]
    ids: List[int]
    # JPEGs to re-push through the frame queue (channel V). Empty unless
    # MEMORY_INJECT_FRAMES — a matched frame with no caption is worth nothing as
    # text, so either the pixels come back or the frame is not recalled at all.
    frames: List[bytes] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.block)


EMPTY_RECALL = RecallResult(block="", items=[], ids=[], frames=[])


@dataclass
class _InjectedEntry:
    """Per-item re-injection bookkeeping, in TEXT-KV TOKEN positions."""
    copies: int = 0          # times injected this session (cap: memory_reinject_max_copies)
    last_tokens: float = 0.0  # session text-token position when last shown


class MemorySession:
    def __init__(self, conversation_id: str, settings: Settings, store: MemoryStore,
                 writer: MemoryWriter, *, default_lang: str = "zh", facts: Any = None) -> None:
        self.conversation_id = conversation_id
        self.settings = settings
        self.store = store
        self.writer = writer
        self.retriever = Retriever(store, writer.text, writer.image, settings)
        # background fact extraction (memory/facts.py); None = facts off
        self._facts = facts
        self.lang_state = lang_mod.LanguageState(default=default_lang)
        self._t0 = time.monotonic()
        self._injected: Dict[int, _InjectedEntry] = {}
        self._suppressed: Set[int] = set()
        self._last_frame_at = 0.0
        self._prefetch: Optional[Tuple[str, List[Candidate]]] = None
        self._lock = threading.Lock()
        # internal text-KV estimate, used only when the orchestrator does not
        # pass the worker's exact count (keeps the distance gate testable and
        # self-contained): noted turns + injected blocks, in estimate_tokens
        self._est_tokens = 0.0
        # estimated INJECTED tokens over the session's life. Deliberately
        # survives note_rollover: exhaustion is a rollover signal (design §5),
        # not a per-KV budget
        self._lifetime_tokens = 0.0
        # estimate of the last formatted block, committed by mark_injected
        self._pending_block_tokens = 0
        self.stats = {"recalls": 0, "injected": 0, "gated_out": 0}

    # ---- clock ----

    def session_ts(self, explicit: Optional[float] = None) -> float:
        if explicit is not None:
            return float(explicit)
        return time.monotonic() - self._t0

    @property
    def language(self) -> str:
        return self.lang_state.current

    # ---- write path (all non-blocking) ----

    def note_user_turn(self, text: str, *, media_ts: Optional[float] = None,
                       session_ts: Optional[float] = None) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._est_tokens += inject_mod.estimate_tokens(text)
        self.lang_state.observe(text)
        self.writer.note_utterance(
            self.conversation_id, "user", text,
            lang=lang_mod.detect_lang(text, default=self.language),
            session_ts=self.session_ts(session_ts), media_ts=media_ts, importance=0.65)

    def note_assistant_turn(self, text: str, *, media_ts: Optional[float] = None) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._est_tokens += inject_mod.estimate_tokens(text)
        self.writer.note_utterance(
            self.conversation_id, "assistant", text,
            lang=lang_mod.detect_lang(text, default=self.language),
            session_ts=self.session_ts(), media_ts=media_ts, importance=0.4)

    def note_frame(self, jpeg: bytes, timestamp: Optional[float] = None,
                   media_ts: Optional[float] = None) -> None:
        """Hot-path throttle only — real dedup happens on the writer thread."""
        now = time.monotonic()
        if (now - self._last_frame_at) < max(0.0, float(self.settings.memory_keyframe_min_interval_s)):
            return
        self._last_frame_at = now
        self.writer.note_frame(self.conversation_id, jpeg, session_ts=self.session_ts(timestamp),
                               media_ts=media_ts, lang=self.language)

    # ---- read path ----

    def prefetch(self, partial_text: str) -> None:
        """Best-effort warm-up from an ASR partial. Never raises, never gates."""
        text = (partial_text or "").strip()
        if len(text) < 2 or not self.settings.memory_prefetch_on_partials:
            return
        try:
            # the same rewrite rules as recall_for_turn, so the cache key and
            # the eventual query agree (the time WINDOW is applied at recall)
            search_q, _ = self._prepare_query(text)
            found = self.retriever.search(self.conversation_id, search_q, exclude=self._skip_ids(),
                                          limit=max(1, int(self.settings.memory_retrieval_topk)))
            if found:
                with self._lock:
                    self._prefetch = (search_q, self.retriever.rank(found))
        except Exception as exc:  # noqa: BLE001
            log.debug("memory prefetch failed: %s", exc)

    def _skip_ids(self) -> Set[int]:
        # ONLY suppression excludes up front. Injected ids stay in retrieval and
        # ranking; the distance gate decides after ranking (design §5).
        return set(self._suppressed)

    def _prepare_query(self, text: str) -> Tuple[str, Optional[rewrite_mod.TimeWindow]]:
        """Rule-based pre-retrieval rewrite (memory/rewrite.py): a confident
        relative-time parse yields a session-time window; a short deictic
        past-reference query is augmented with the most recent user turns so
        raw embedding similarity has something to bite on. Conservative:
        anything unrecognized passes through untouched."""
        window = rewrite_mod.parse_time_window(text, self.session_ts())
        query = text
        if rewrite_mod.is_deictic_query(text):
            recent = self.store.recent(self.conversation_id, [KIND_UTTERANCE], limit=8)
            texts = [i.text for i in recent if i.role == "user" and i.text][:2]
            query = rewrite_mod.augment_query(text, texts)
        return query, window

    def recall_for_turn(self, text: str, *, now_tokens: Optional[float] = None) -> RecallResult:
        """Blocking (call via to_thread): rewrite → search → rank → time window
        → re-injection distance gate → diversify → admission gate → format.

        `now_tokens` is the session's text-KV token count; when None the
        internal running estimate is used (see module docstring)."""
        query = (text or "").strip()
        if not query:
            return EMPTY_RECALL
        if self._lifetime_tokens >= max(1, int(self.settings.memory_inject_session_max_tokens)):
            # lifetime budget exhausted: stop recalling SILENTLY — this is a
            # rollover signal, not a reason to raise the cap (design §5)
            return EMPTY_RECALL
        if now_tokens is not None:
            # the worker's exact count supersedes the estimate, never rewinds it
            self._est_tokens = max(self._est_tokens, float(now_tokens))
        self._pending_block_tokens = 0
        try:
            search_q, window = self._prepare_query(query)
            candidates = self._candidates(search_q)
            if not candidates:
                return EMPTY_RECALL
            candidates = self._time_gate(candidates, window)
            candidates = self._distance_gate(candidates, self._est_tokens)
            if not candidates:
                self.stats["gated_out"] += 1
                return EMPTY_RECALL
            limit = max(1, int(self.settings.memory_retrieval_topk))
            # diversify/gate read the RAW query — the augmentation is search-only
            diverse = self.retriever.diversify(candidates, query, limit=limit)
            keep = self.retriever.gate(query, diverse)
            self.stats["recalls"] += 1
            if not keep:
                self.stats["gated_out"] += 1
                return EMPTY_RECALL
            return self._format(keep)
        except Exception as exc:  # noqa: BLE001 — memory never breaks a turn
            log.warning("memory recall failed: %s", exc)
            return EMPTY_RECALL

    def _candidates(self, query: str) -> List[Candidate]:
        with self._lock:
            cached = self._prefetch
            self._prefetch = None
        if cached is not None:
            cached_q, cands = cached
            # the partial is a prefix of the final transcript often enough that
            # reusing it is the common case; otherwise fall through to a search
            if cached_q and (query.startswith(cached_q) or cached_q.startswith(query)):
                skip = self._skip_ids()
                fresh = [c for c in cands if c.item.id not in skip]
                if fresh:
                    return fresh
        found = self.retriever.search(self.conversation_id, query, exclude=self._skip_ids(),
                                      limit=max(1, int(self.settings.memory_retrieval_topk)))
        return self.retriever.rank(found) if found else []

    def _time_gate(self, candidates: List[Candidate],
                   window: Optional[rewrite_mod.TimeWindow]) -> List[Candidate]:
        """A confident relative-time window restricts candidates to items whose
        ORIGINAL session_ts falls inside it (boosted so they keep their rank
        edge). A window with no hits drops nothing — the parse was confident,
        the corpus may simply not reach back that far."""
        if window is None or not candidates:
            return candidates
        lo, hi = window
        inside = [c for c in candidates
                  if c.item.session_ts is not None and lo <= c.item.session_ts <= hi]
        if not inside:
            return candidates
        for cand in inside:
            cand.score += 0.5
        inside.sort(key=lambda c: c.score, reverse=True)
        return inside

    def _distance_gate(self, candidates: List[Candidate], now_tokens: float) -> List[Candidate]:
        """Re-injection control, applied AFTER ranking (design §5): an
        already-shown item competes with fresh ones and is dropped only if it
        is still inside the effective window — fewer than
        `memory_reinject_distance` text-KV tokens since it was last shown — or
        has hit its per-session copy cap. Fresh candidates are never touched."""
        max_copies = max(1, int(self.settings.memory_reinject_max_copies))
        distance = max(0, int(self.settings.memory_reinject_distance))
        out: List[Candidate] = []
        for cand in candidates:
            entry = self._injected.get(cand.item.id)
            if entry is None:
                out.append(cand)
                continue
            if entry.copies >= max_copies:
                continue
            if (now_tokens - entry.last_tokens) < distance:
                continue
            out.append(cand)
        return out

    def _format(self, keep: Sequence[Candidate]) -> RecallResult:
        """Chronological by ORIGINAL time, under a hard token budget."""
        ordered = sorted(keep, key=lambda c: (c.item.session_ts or 0.0))
        budget = max(0, int(self.settings.memory_inject_max_tokens))
        lines: List[str] = []
        items: List[Dict[str, Any]] = []
        ids: List[int] = []
        frames: List[bytes] = []
        used = 0
        for cand in ordered:
            item = cand.item
            is_frame = item.kind == KIND_FRAME
            jpeg = None
            if is_frame and not item.text:
                # cross-modal retrieval can match a frame that was never
                # captioned; it is only useful if the pixels come back with it
                if not self.settings.memory_inject_frames:
                    continue
                jpeg = self._load_frame(item.media_hash)
                if jpeg is None:
                    continue
            # re-injects ride the SHORT form (design §5): the first clause of
            # the verbatim text, not a byte-repeat of the whole turn
            entry = self._injected.get(item.id)
            if entry is not None and entry.copies >= 1 and item.text:
                body = inject_mod.first_clause(item.text)
            else:
                body = item.text or ("画面" if self.language.startswith("zh") else "camera view")
            line = inject_mod.format_recall_line(body, item.session_ts, frame=is_frame)
            cost = inject_mod.estimate_tokens(line)
            if used + cost > budget:
                break
            lines.append(line)
            used += cost
            ids.append(item.id)
            if jpeg is not None:
                frames.append(jpeg)
            items.append({"id": item.id, "text": item.text, "session_ts": item.session_ts,
                          "kind": item.kind, "media": item.media_hash, "score": round(cand.score, 4)})
        if not lines:
            return EMPTY_RECALL
        block = inject_mod.build_recall_block(lines)
        self._pending_block_tokens = inject_mod.estimate_tokens(block)
        return RecallResult(block=block, items=items, ids=ids, frames=frames)

    def _load_frame(self, media_hash: Optional[str]) -> Optional[bytes]:
        media = getattr(self.writer, "media", None)
        if not media_hash or media is None:
            return None
        try:
            return media.load_bytes(media_hash)
        except Exception:  # noqa: BLE001
            return None

    def mark_injected(self, ids: Sequence[int], *, text_tokens: Optional[float] = None) -> None:
        """Commit an injected block: bump each item's copy count, stamp the
        text-KV position it was shown at, and fold the block's estimated tokens
        into the lifetime budget and the position estimate.

        `text_tokens` is the worker's exact session text-token count when the
        orchestrator plumbs it through; otherwise the internal estimate."""
        pos = float(text_tokens) if text_tokens is not None else self._est_tokens
        for item_id in ids:
            entry = self._injected.setdefault(int(item_id), _InjectedEntry())
            entry.copies += 1
            entry.last_tokens = pos
            try:
                self.store.mark_injected(int(item_id), self.conversation_id)
            except Exception:  # noqa: BLE001
                pass
        if ids:
            block_tokens = float(self._pending_block_tokens)
            self._pending_block_tokens = 0
            self._lifetime_tokens += block_tokens
            # the block itself occupies KV from here on — keep the estimate monotone
            self._est_tokens = max(self._est_tokens, pos) + block_tokens
            self.stats["injected"] += len(ids)

    def note_rollover(self, kept_item_ids: Sequence[int], *, text_tokens: float = 0.0) -> None:
        """Re-seat after a KV rollover (the rollover machinery itself is a
        separate workstream): the injected table is recomputed from EXACTLY the
        ids that went into the new prefix (design §5) — clearing it blindly
        would re-inject items already in the prefix as duplicates. Distances
        reset (each kept item was last shown at the new prefix's token count);
        `_suppressed` PERSISTS ("不是那个" survives a rollover); the lifetime
        injected-token budget does NOT reset.
        """
        self._injected = {int(i): _InjectedEntry(copies=1, last_tokens=float(text_tokens))
                          for i in kept_item_ids}
        self._est_tokens = max(0.0, float(text_tokens))
        self._pending_block_tokens = 0

    def suppress(self, ids: Sequence[int]) -> None:
        """User said 'not that one' — stop surfacing these for this session."""
        self._suppressed.update(int(i) for i in ids)

    # ---- background fact extraction (never on the hot path) ----

    async def maybe_extract_facts(self, item_id: Optional[int], user_text: str,
                                  context_turns: Optional[Sequence[str]] = None) -> None:
        """Fire-and-forget from the orchestrator once an assistant reply
        finalizes a QA pair (a completed reply is the segment boundary,
        design §3). Facts enrich the RETRIEVAL INDEX KEY only — the model
        never sees them. Never raises; every failure is a silent no-op.
        """
        extractor = self._facts
        text = (user_text or "").strip()
        if extractor is None or not text:
            return
        if (self.settings.memory_fact_scope or "user") != "user":
            return
        try:
            if item_id is None:
                item_id = await self._resolve_user_item(text)
            if item_id is None:
                return
            if context_turns is None:
                context_turns = await asyncio.to_thread(self._recent_context, int(item_id))
            await extractor.extract_and_rekey(self.conversation_id, int(item_id), text,
                                              context_turns)
        except Exception as exc:  # noqa: BLE001 — memory must never kill a session
            log.debug("memory fact extraction failed: %s", exc)

    async def _resolve_user_item(self, text: str) -> Optional[int]:
        """Find the stored row for a just-noted user turn. The writer thread
        owns inserts, so the row lands a few ms after note_user_turn — a short
        poll beats wiring row ids back through the enqueue-only hot path."""
        for _ in range(30):
            found = await asyncio.to_thread(self._find_user_item, text)
            if found is not None:
                return found
            await asyncio.sleep(0.1)
        return None

    def _find_user_item(self, text: str) -> Optional[int]:
        for item in self.store.recent(self.conversation_id, [KIND_UTTERANCE], limit=12):
            if item.role == "user" and (item.text or "").strip() == text:
                return item.id
        return None

    def _recent_context(self, exclude_id: int) -> List[str]:
        """Surrounding turns (chronological) to interpret an elliptical user
        turn. Assistant turns are CONTEXT ONLY — never a fact source."""
        out: List[str] = []
        for item in self.store.recent(self.conversation_id, [KIND_UTTERANCE], limit=8):
            if item.id == exclude_id or not item.text:
                continue
            role = "用户" if item.role == "user" else "助手"
            out.append(f"{role}: {item.text}")
        out.reverse()
        return out

    # ---- lifecycle ----

    def close(self) -> None:
        self.writer.forget(self.conversation_id)
        self.store.forget_session_cache(self.conversation_id)

    def status(self) -> Dict[str, Any]:
        return {"items": self.store.count(self.conversation_id),
                "injected_tokens": int(self._lifetime_tokens),
                "lang": self.language, **self.stats}
