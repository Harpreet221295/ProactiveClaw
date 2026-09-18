"""care_patterns.json — what this user actually cares about, learned over time.

Observations are recorded per key (`sender:maya@x.com`, `topic:investor`,
`category:newsletter`, `source:notion`) with counts of how the user reacted:
acted / dismissed / deferred. Explicit rules (mute / boost) come from things
the user says ("ignore anything from X", "always flag term-sheet emails").

`score(item)` returns an urgency adjustment (-1, 0, +1) and whether the item
should be filtered outright. Learned counts decay so stale habits fade.
"""
from __future__ import annotations

import json
import math
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from core.paths import CARE_PATTERNS_FILE
from core.config import effective_settings, load_config

KINDS = ("sender", "topic", "category", "source")
SIGNALS = ("acted", "dismissed", "deferred")
RULE_ACTIONS = ("mute", "boost")

_lock = threading.RLock()


def _now() -> datetime:
    return datetime.now().astimezone()


def _key(kind: str, key: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    return f"{kind}:{key.strip().lower()}"


class CarePatterns:
    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path else Path(CARE_PATTERNS_FILE)
        self._data: dict[str, Any] | None = None

    # ── persistence ─────────────────────────────────────────────────────────

    def load(self) -> dict[str, Any]:
        with _lock:
            if self._data is None:
                if self.path.exists():
                    try:
                        self._data = json.loads(self.path.read_text(encoding="utf-8")) or {}
                    except json.JSONDecodeError:
                        self._data = {}
                else:
                    self._data = {}
                self._data.setdefault("observations", {})
                self._data.setdefault("rules", [])
            return self._data

    def save(self) -> None:
        with _lock:
            data = self.load()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(tmp, self.path)

    # ── recording ───────────────────────────────────────────────────────────

    def observe(self, kind: str, key: str, signal: str, note: str = "") -> dict[str, Any]:
        if signal not in SIGNALS:
            raise ValueError(f"signal must be one of {SIGNALS}")
        k = _key(kind, key)
        with _lock:
            obs = self.load()["observations"].setdefault(k, {
                "kind": kind, "key": key.strip().lower(),
                "acted": 0.0, "dismissed": 0.0, "deferred": 0.0,
                "first_seen": _now().isoformat(), "last_seen": None, "notes": [],
            })
            self._decay(obs)
            obs[signal] = float(obs.get(signal, 0.0)) + 1.0
            obs["last_seen"] = _now().isoformat()
            if note:
                obs["notes"] = (obs.get("notes") or [])[-4:] + [note[:200]]
            self.save()
            return obs

    def observe_item(self, item: dict[str, Any], signal: str, note: str = "") -> list[str]:
        """Record a reaction to a registry item across all its pattern keys."""
        keys: list[str] = []
        if item.get("sender"):
            self.observe("sender", item["sender"], signal, note); keys.append(f"sender:{item['sender']}")
        for t in item.get("topics") or []:
            self.observe("topic", t, signal, note); keys.append(f"topic:{t}")
        if item.get("source"):
            self.observe("source", item["source"], signal, note); keys.append(f"source:{item['source']}")
        return keys

    def add_rule(self, kind: str, key: str, action: str, note: str = "") -> dict[str, Any]:
        if action not in RULE_ACTIONS:
            raise ValueError(f"action must be one of {RULE_ACTIONS}")
        k = _key(kind, key)
        with _lock:
            rules = self.load()["rules"]
            rules[:] = [r for r in rules if r.get("id") != k]
            rule = {"id": k, "kind": kind, "key": key.strip().lower(), "action": action,
                    "note": note[:200], "created_at": _now().isoformat()}
            rules.append(rule)
            self.save()
            return rule

    def remove_rule(self, kind: str, key: str) -> bool:
        k = _key(kind, key)
        with _lock:
            rules = self.load()["rules"]
            before = len(rules)
            rules[:] = [r for r in rules if r.get("id") != k]
            self.save()
            return len(rules) < before

    # ── decay ───────────────────────────────────────────────────────────────

    def _decay(self, obs: dict[str, Any], cfg: dict[str, Any] | None = None) -> None:
        eff = effective_settings(cfg or load_config())
        decay_days = max(1, int(eff["pattern_learning"].get("decay_days", 30)))
        last = obs.get("last_seen")
        if not last:
            return
        try:
            elapsed = (_now() - datetime.fromisoformat(last)).total_seconds() / 86400
        except ValueError:
            return
        if elapsed <= 0:
            return
        factor = math.exp(-elapsed / decay_days)
        for s in SIGNALS:
            obs[s] = round(float(obs.get(s, 0.0)) * factor, 3)

    # ── scoring ─────────────────────────────────────────────────────────────

    def _rule_for(self, kind: str, key: str) -> dict[str, Any] | None:
        k = _key(kind, key)
        return next((r for r in self.load()["rules"] if r.get("id") == k), None)

    def _learned_signal(self, kind: str, key: str, min_obs: int) -> str | None:
        obs = self.load()["observations"].get(_key(kind, key))
        if not obs:
            return None
        self._decay(obs)
        total = obs["acted"] + obs["dismissed"] + obs["deferred"]
        if total < min_obs:
            return None
        if obs["dismissed"] >= 0.7 * total:
            return "dismissed"
        if obs["acted"] >= 0.6 * total:
            return "acted"
        if obs["deferred"] >= 0.7 * total:
            return "deferred"
        return None

    def score(self, item: dict[str, Any], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return {"adjust": -1|0|1, "filter": bool, "reasons": [...]}."""
        cfg = cfg or load_config()
        eff = effective_settings(cfg)
        reasons: list[str] = []
        adjust = 0
        filtered = False

        text = " ".join(str(item.get(k, "")) for k in ("title", "details", "sender")).lower()
        topics = [t.lower() for t in (item.get("topics") or [])]

        # explicit config topic lists
        for t in eff.get("boosted_topics", []):
            if t.lower() in text or t.lower() in topics:
                adjust = max(adjust, 1); reasons.append(f"boosted topic '{t}'"); break
        for t in eff.get("muted_topics", []):
            if t.lower() in text or t.lower() in topics:
                filtered = True; reasons.append(f"muted topic '{t}'"); break
        cat = (item.get("category") or "").lower()
        if cat and cat in [c.lower() for c in eff.get("auto_filter_categories", [])]:
            filtered = True; reasons.append(f"auto-filtered category '{cat}'")

        # explicit user rules
        candidates: list[tuple[str, str]] = []
        if item.get("sender"):
            candidates.append(("sender", item["sender"]))
        candidates += [("topic", t) for t in topics]
        if item.get("source"):
            candidates.append(("source", item["source"]))
        if cat:
            candidates.append(("category", cat))
        for kind, key in candidates:
            rule = self._rule_for(kind, key)
            if rule:
                if rule["action"] == "mute":
                    filtered = True; reasons.append(f"rule: mute {kind} '{key}'")
                else:
                    adjust = 1; reasons.append(f"rule: boost {kind} '{key}'")
        # rules also match substrings of sender/title (e.g. domain or partial name)
        for r in self.load()["rules"]:
            if r["kind"] in ("sender", "topic") and r["key"] and r["key"] in text:
                if r["action"] == "mute" and not filtered:
                    filtered = True; reasons.append(f"rule: mute {r['kind']} '{r['key']}'")
                elif r["action"] == "boost" and adjust < 1:
                    adjust = 1; reasons.append(f"rule: boost {r['kind']} '{r['key']}'")

        # learned behaviour
        if eff["pattern_learning"].get("enabled", True):
            min_obs = int(eff["pattern_learning"].get("min_observations", 3))
            for kind, key in candidates:
                sig = self._learned_signal(kind, key, min_obs)
                if sig == "dismissed":
                    adjust = min(adjust, -1); reasons.append(f"you usually dismiss {kind} '{key}'")
                    if kind == "sender":
                        filtered = True
                elif sig == "acted":
                    adjust = max(adjust, 1); reasons.append(f"you usually act on {kind} '{key}'")
                elif sig == "deferred":
                    reasons.append(f"you usually defer {kind} '{key}'")

        return {"adjust": adjust, "filter": filtered, "reasons": reasons}

    def apply(self, urgency: str, item: dict[str, Any], cfg: dict[str, Any] | None = None) -> tuple[str, bool, list[str]]:
        """Apply learned adjustment to an urgency level. Returns (urgency, filtered, reasons)."""
        order = ["low", "medium", "high"]
        s = self.score(item, cfg)
        idx = order.index(urgency if urgency in order else "medium")
        idx = max(0, min(2, idx + s["adjust"]))
        return order[idx], s["filter"], s["reasons"]

    # ── views ───────────────────────────────────────────────────────────────

    def digest(self, limit: int = 12) -> str:
        data = self.load()
        lines: list[str] = []
        if data["rules"]:
            lines.append("Explicit rules:")
            for r in data["rules"][:limit]:
                lines.append(f"  • {r['action']} {r['kind']} '{r['key']}'" + (f" — {r['note']}" if r.get("note") else ""))
        strong = []
        for k, obs in data["observations"].items():
            total = obs.get("acted", 0) + obs.get("dismissed", 0) + obs.get("deferred", 0)
            if total >= 2:
                dom = max(("acted", "dismissed", "deferred"), key=lambda s: obs.get(s, 0))
                strong.append((total, f"  • {k}: mostly {dom} ({obs.get('acted',0):.0f} acted / {obs.get('dismissed',0):.0f} dismissed / {obs.get('deferred',0):.0f} deferred)"))
        if strong:
            strong.sort(reverse=True)
            lines.append("Learned tendencies:")
            lines += [s for _, s in strong[:limit]]
        return "\n".join(lines) if lines else "No patterns learned yet."
