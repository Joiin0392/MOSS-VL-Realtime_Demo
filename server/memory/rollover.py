"""Rollover (compaction) — design §6.

MOSS-VL's text KV is append-only and grows forever; ~80% of that growth is
timestamp wrappers for frames whose vision KV the frame window already
evicted. When the worker's exact TEXT-token count (the token-counter patch in
realtime/mossvl_patches.py, surfaced via the 1 Hz status) crosses a threshold,
this module rebuilds the prefix and the orchestrator re-seats the session:

- idle trigger `memory_rollover_idle_tokens`, only at a `<|silence|>` idle moment
- hard trigger `memory_rollover_hard_tokens`, fires regardless (but never
  mid-streaming-assistant-output — the orchestrator defers a live response)
- anti-thrash floor: skip when less than `memory_rollover_min_progress` of the
  text KV would actually be reclaimed

The new prefill is ONE role:system message at position 0 — base prompt /
recall-format declaration / pinned / verbatim-hold / summary / handle line —
followed by the last `memory_rollover_tail_turns` turns verbatim as real
user/assistant tail messages. The summary is RE-DERIVED from the full raw
journal each rollover (never summary-of-summary) on the offline sglang plane;
provider != "offline" or an absent/unloaded/failing plane degrades to
verbatim-tail-only with NO error (1-GPU boxes have no offline plane, §7).
Correction detection and assembly are lexical/string-concat — no model (§7).
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Callable, List, Optional, Sequence, Tuple

from ..config import Settings
from ..logging_conf import get_logger
from ..schemas import ChatMessage, ChatRequest, GenerationParams
from . import inject as inject_mod
from .store import KIND_PINNED, KIND_UTTERANCE, MemoryItem, MemoryStore

log = get_logger(__name__)

# Layer token budgets in estimate_tokens units (over-estimating is the safe
# direction). Target prefix ≈1.5k tokens — design §6's headroom math assumes it.
_BUDGET_PINNED = 240         # user-pinned items; the last layer ever evicted
_BUDGET_VERBATIM_HOLD = 320  # corrections / identifiers / numbers / commitments
_BUDGET_TAIL = 700           # the verbatim recent-turns tail
# base prompt + recall declaration + handle line are fixed scaffolding and are
# estimated from their real text, not budgeted.

_JOURNAL_LIMIT = 2000        # full raw journal cap (utterance rows)
_TAIL_MSG_OVERHEAD = 8       # chat-template scaffolding per tail message (est.)

# verbatim-hold extraction (lexical, design §6 — no model):
_CORRECTION_RE = re.compile(r"(不是那个|不是|我说的是|我是说|I meant|I said)", re.IGNORECASE)
_NUMBER_UNIT_RE = re.compile(
    r"\d+(?:\.\d+)?\s?(?:元|块|岁|年|月|日|号|点|分|秒|米|厘米|毫米|公里|千克|克|斤|升|毫升"
    r"|km|cm|mm|kg|mg|GB|MB|TB|mL|ml|°C|°F|%|dollars?|years?|hours?|minutes?|seconds?|meters?|miles?)")
# proper nouns / identifiers: hyphen-joined latin runs (BGE-M3), ALLCAPS+digits
# (FM2), or letter+digit mixes (GPT4) — the things a summarizer paraphrases away
_IDENTIFIER_RE = re.compile(r"[A-Za-z][\w.]*(?:[-_][\w.]+)+|[A-Z]{2,}\d+|\b[A-Za-z]+\d+\b")
_COMMITMENT_RE = re.compile(r"(我会|我帮你|我这就|我来帮|I will|I'll|let me)", re.IGNORECASE)

_SUMMARY_PROMPT = """把下面的实时对话压缩成一段简短摘要,按时间顺序保留事实、决定、待办和专有名称;只输出摘要正文,不要编号或解释。
Summarize the realtime conversation below, chronological; keep facts, decisions, todos and proper names; output ONLY the summary text, no numbering or commentary.

对话 / Conversation:
{journal}"""

# mirror of mossvl_patches.DEFAULT_BOARD_REALTIME_SYSTEM_PROMPT — used only
# when the lazy import fails (a slim gateway venv without torch); keep in sync
_DEFAULT_SYSTEM_PROMPT_MIRROR = (
    "You are a helpful AI assistant specializing in real-time video analysis. "
    "The video streams to you frame by frame. At every frame, you decide independently "
    "whether to respond or stay silent — output `<|silence|>` when nothing relevant has happened, "
    "and respond when the visual content warrants it."
)


def _default_system_prompt() -> str:
    """Lazy import: realtime.mossvl_patches pulls in torch, which the gateway
    process does not otherwise need."""
    try:
        from ..realtime.mossvl_patches import DEFAULT_BOARD_REALTIME_SYSTEM_PROMPT
        return DEFAULT_BOARD_REALTIME_SYSTEM_PROMPT
    except Exception:  # noqa: BLE001
        return _DEFAULT_SYSTEM_PROMPT_MIRROR


class RolloverManager:
    """Per-session compaction planner. Construction is cheap; the orchestrator
    drives `should_rollover` (sync, hot-ish) and `build_prefix` (async, off the
    turn path). Every failure degrades to "no rollover", never a broken turn.
    """

    def __init__(self, settings: Settings, store: MemoryStore, conversation_id: str,
                 *, plane: Any = None, base_system_prompt: str = "",
                 lang_getter: Optional[Callable[[], str]] = None,
                 semaphore: Optional[asyncio.Semaphore] = None) -> None:
        self.settings = settings
        self.store = store
        self.conversation_id = conversation_id
        # offline sglang plane (rt.vlm_offline) or None; the summarizer shares
        # the background-job semaphore with fact extraction (design §7:
        # concurrency 1-2 against the sidecar), falling back to its own
        self.plane = plane
        self.base_system_prompt = base_system_prompt or ""
        self._lang_getter = lang_getter or (lambda: "zh")
        self._semaphore = semaphore or asyncio.Semaphore(
            max(1, int(settings.memory_bg_concurrency)))

    # ------------------------------------------------------------------ trigger

    def enabled(self) -> bool:
        return int(self.settings.memory_rollover_hard_tokens) > 0

    def should_rollover(self, text_tokens: Any, *, idle: bool = False) -> bool:
        """Trigger decision on the worker's exact text-token count.

        The hard trigger fires regardless of idleness; the idle trigger only at
        a `<|silence|>` idle moment. Both are subject to the anti-thrash floor:
        if the rebuilt prefix would carry almost as much as the current one,
        the seam costs more than it reclaims.
        """
        if not self.enabled():
            return False
        try:
            tokens = float(text_tokens)
        except (TypeError, ValueError):
            return False
        if tokens < float(self.settings.memory_rollover_hard_tokens):
            if not idle or tokens < float(self.settings.memory_rollover_idle_tokens):
                return False
        # anti-thrash: estimate what the new prefix will carry from CURRENT
        # journal sizes (fixed scaffolding + the budgeted layers at their real
        # sizes + the summary budget when a summary plane is configured). Only
        # an order-of-magnitude check — the real count is known post-build.
        try:
            _, _, est_prefix = self._assemble(summary=None)
        except Exception as exc:  # noqa: BLE001 — a broken estimate skips, never crashes
            log.warning("rollover estimate failed: %s", exc)
            return False
        if self._summary_configured():
            est_prefix += max(0, int(self.settings.memory_summary_max_tokens))
        if (tokens - est_prefix) < float(self.settings.memory_rollover_min_progress) * tokens:
            log.info("rollover skipped (anti-thrash): tokens=%d est_prefix=%d",
                     int(tokens), est_prefix)
            return False
        return True

    # ------------------------------------------------------------------ prefix build

    async def build_prefix(self) -> Tuple[List[dict], List[int], int]:
        """(prefill_messages, kept_item_ids, est_prefix_tokens).

        `kept_item_ids` are EXACTLY the journal ids whose content went into the
        new prefix — the orchestrator hands them to `MemorySession.note_rollover`
        so the injected set is recomputed from reality (design §5), not cleared.
        """
        journal, pinned, tail = await asyncio.to_thread(self._collect)
        summary = await self._summarize(journal)
        return self._assemble(summary, collected=(journal, pinned, tail))

    # ------------------------------------------------------------------ internals

    def _lang(self) -> str:
        try:
            return (self._lang_getter() or "zh").lower()
        except Exception:  # noqa: BLE001
            return "zh"

    def _base_prompt(self) -> str:
        return self.base_system_prompt.strip() or _default_system_prompt()

    def _collect(self) -> Tuple[List[MemoryItem], List[MemoryItem], List[MemoryItem]]:
        """(journal, pinned, tail), all chronological. Blocking sqlite — the
        async callers go through to_thread; `should_rollover` calls it directly
        from the orchestrator's status/idle hooks (a bounded indexed read)."""
        journal = list(reversed(self.store.recent(
            self.conversation_id, [KIND_UTTERANCE], limit=_JOURNAL_LIMIT)))
        pinned = list(reversed(self.store.recent(
            self.conversation_id, [KIND_PINNED], limit=64)))
        return journal, pinned, journal[self._tail_start(journal):]

    def _tail_start(self, journal: Sequence[MemoryItem]) -> int:
        """Boundary rule (design §6): complete QA pairs only — the kept tail
        must not begin with an assistant reply whose user turn is being
        compacted away, so extend the cut backwards past leading replies."""
        n = max(1, int(self.settings.memory_rollover_tail_turns))
        start = max(0, len(journal) - n)
        while start > 0 and journal[start].role == "assistant":
            start -= 1
        return start

    def _budget_lines(self, items: Sequence[MemoryItem], budget: int
                      ) -> Tuple[List[str], List[int]]:
        """Newest-first admission = LRU eviction of the oldest once the layer's
        token budget is spent (design §6); returned chronological."""
        lines: List[str] = []
        ids: List[int] = []
        used = 0
        for item in reversed(list(items)):
            text = inject_mod.sanitize_model_text(item.text).replace("\n", " ").strip()
            if not text:
                continue
            line = f"[{inject_mod.format_stamp(item.session_ts)}] {text}"
            cost = inject_mod.estimate_tokens(line)
            if used + cost > budget:
                continue  # keep scanning: a short older line may still fit
            lines.append(line)
            ids.append(item.id)
            used += cost
        lines.reverse()
        ids.reverse()
        return lines, ids

    def _verbatim_hold(self, journal: Sequence[MemoryItem]) -> Tuple[List[str], List[int]]:
        """Copied character-exact (the summarizer never sees these): user
        corrections, identifiers/proper nouns, numbers with units, assistant
        commitments. Lexical only — correction detection has no model (§7)."""
        picked: List[MemoryItem] = []
        for item in reversed(list(journal)):  # newest first → LRU eviction
            text = (item.text or "").strip()
            if not text:
                continue
            if item.role == "user" and _CORRECTION_RE.search(text):
                picked.append(item)
            elif item.role == "assistant" and _COMMITMENT_RE.search(text):
                picked.append(item)
            elif _NUMBER_UNIT_RE.search(text) or _IDENTIFIER_RE.search(text):
                picked.append(item)
        return self._budget_lines(list(reversed(picked)), _BUDGET_VERBATIM_HOLD)

    def _assemble(self, summary: Optional[str],
                  collected: Optional[Tuple[List[MemoryItem], List[MemoryItem], List[MemoryItem]]] = None,
                  ) -> Tuple[List[dict], List[int], int]:
        """System message = base prompt + recall declaration + pinned +
        verbatim-hold + summary + handle line, in THAT order (design §6); then
        the verbatim tail as real user/assistant messages. The summary is never
        an assistant-role message (voice drift) — it lives inside the system
        layer like every other compacted block."""
        journal, pinned, tail = collected if collected is not None else self._collect()
        lang = self._lang()
        zh = lang.startswith("zh")
        sections = [inject_mod.augment_system_prompt(self._base_prompt(), lang)]
        kept: List[int] = []

        pinned_lines, pinned_ids = self._budget_lines(pinned, _BUDGET_PINNED)
        if pinned_lines:
            header = "置顶记忆 / Pinned:" if zh else "Pinned memories:"
            sections.append(header + "\n" + "\n".join(pinned_lines))
            kept.extend(pinned_ids)

        hold_lines, hold_ids = self._verbatim_hold(journal)
        if hold_lines:
            header = "逐字保留的记忆 / Verbatim:" if zh else "Verbatim memories (exact):"
            sections.append(header + "\n" + "\n".join(hold_lines))
            kept.extend(hold_ids)

        if summary:
            header = "此前对话摘要 / Summary:" if zh else "Conversation summary so far:"
            sections.append(header + "\n" + summary)

        max_ts = max((it.session_ts or 0.0) for it in journal) if journal else 0.0
        stamp = inject_mod.format_stamp(max_ts)
        sections.append(f"记忆库覆盖 t=0…{stamp}" if zh else f"Memory bank covers t=0…{stamp}")

        system_text = "\n\n".join(s for s in sections if s)
        messages: List[dict] = [{"role": "system", "content": system_text}]
        est = inject_mod.estimate_tokens(system_text) + _TAIL_MSG_OVERHEAD

        tail_items = list(tail)
        # tail budget: drop the OLDEST turns first, then re-assert the QA
        # boundary so a shrunk tail still starts on a user turn
        while tail_items and sum(
                inject_mod.estimate_tokens(inject_mod.sanitize_model_text(it.text))
                for it in tail_items) > _BUDGET_TAIL:
            tail_items.pop(0)
        while tail_items and tail_items[0].role == "assistant":
            tail_items.pop(0)

        for item in tail_items:
            text = inject_mod.sanitize_model_text(item.text)
            if not text:
                continue
            role = "user" if item.role == "user" else "assistant"
            messages.append({"role": role, "content": text})
            kept.append(item.id)
            est += inject_mod.estimate_tokens(text) + _TAIL_MSG_OVERHEAD
        return messages, kept, est

    # ------------------------------------------------------------------ summary

    def _summary_configured(self) -> bool:
        return (self.settings.memory_summary_provider or "") == "offline"

    async def _summarize(self, journal: Sequence[MemoryItem]) -> Optional[str]:
        """One summary re-derived from the FULL raw journal, every rollover
        (never summary-of-summary, design §6). Any provider/plane/generation
        problem returns None → verbatim-tail-only, never an exception."""
        if not self._summary_configured():
            return None
        plane = self.plane
        if plane is None:
            return None
        try:
            if not plane.is_loaded():
                return None
        except Exception:  # noqa: BLE001
            return None
        # journal text is user-authored: sanitize before it reaches ANY model
        # (split_special_tokens=False makes special-token strings live control
        # tokens), and cap the prompt so a huge journal can't stall the sidecar
        lines = []
        used = 0
        for item in journal:
            text = inject_mod.sanitize_model_text(item.text).replace("\n", " ").strip()
            if not text:
                continue
            who = "用户" if item.role == "user" else "助手"
            line = f"{who}: {text}"
            used += inject_mod.estimate_tokens(line)
            if used > 3000:  # ~2x the design prefix; plenty for a 200-token summary
                lines.append("…")
                break
            lines.append(line)
        if not lines:
            return None
        req = ChatRequest(
            messages=[ChatMessage(role="user", content=_SUMMARY_PROMPT.format(
                journal="\n".join(lines)))],
            params=GenerationParams(
                max_new_tokens=max(16, int(self.settings.memory_summary_max_tokens)),
                temperature=0.0))
        try:
            async with self._semaphore:
                parts: List[str] = []
                async for delta in plane.generate_stream(req):
                    parts.append(str(delta))
        except Exception as exc:  # noqa: BLE001 — degrade to verbatim-tail-only
            log.warning("rollover summary failed: %s", exc)
            return None
        text = inject_mod.sanitize_model_text("".join(parts))
        return text or None
