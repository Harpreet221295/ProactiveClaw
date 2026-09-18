# Configuration reference

Two files, both in the project root:

- **`.env`** — secrets and process settings (API keys, Slack tokens, `AGENT_TIMEOUT`, `HOST`/`PORT`).
- **`config/config.json`** — behaviour. Copy from `config/config.example.json`. Edited by the Settings panel, the setup wizard, or by hand (changes are picked up live).

```json
{
  "user": {"name": "Ada Lovelace", "timezone": "Europe/London"},
  "llm": {"model": "gpt-4o-mini"},
  "proactiveness": {"level": "balanced"},
  "care_mode": "normal",
  "connectors": {"gmail": false, "calendar": false, "notion": false, "web_search": true, "browser": false},
  "notion": {"tasks_database_id": "", "tasks_schema_hint": ""},
  "morning_review": {"time": "07:30", "days": ["monday", "…", "sunday"]},
  "overrides": {}
}
```

## Effective settings

Every behavioural knob is resolved in this order — later wins:

```
EFFECTIVE_DEFAULTS  ←  proactiveness level preset  ←  care mode  ←  overrides
```

`off` is absolute: no override can re-enable nudges, the morning review, re-engagement or commitment capture.

| Key | Meaning | off | minimal | balanced | active | max |
|---|---|---|---|---|---|---|
| `invocations_enabled` | may schedule nudges | ✗ | ✓ | ✓ | ✓ | ✓ |
| `morning_review_enabled` | daily review job | ✗ | ✓ | ✓ | ✓ | ✓ |
| `morning_review_dm` | when the review posts a message | never | if_high_urgency | if_items | always | always |
| `reengagement_enabled` / `reengagement_intervals_hours` | backoff after nudges run out | ✗ | ✗ | 168, 336, 720 | 72, 168, 336, 720 | 24, 72, 168, 336 |
| `max_nudges_per_day` | hard cap (scheduled + delivered) | 0 | 1 | 3 | 6 | 10 |
| `quiet_hours` | nudges are moved to the end of the window | 22–08 | 21–09 | 22–08 | 23–07 | 00–06 |
| `commitment_capture` | off / conservative / aggressive | off | conservative | conservative | aggressive | aggressive |
| `ask_on_borderline` | ask "want me to track that?" | ✗ | ✗ | ✓ | ✗ | ✗ |
| `research_followups` | follow up on info you were given | ✗ | ✗ | ✗ | ✓ | ✓ |
| `surface_care_items_in_chat` | weave tracked items into replies | ✗ | ✓ | ✓ | ✓ | ✓ |

Always-available keys (defaults): `urgency_threshold` (medium), `boosted_topics` ([]), `muted_topics` ([]), `auto_filter_categories` (newsletter, promotion, automated, social, receipt), `resurface_after_hours` (24), `escalate_after_deferrals` (3), `item_max_age_days` (14), `max_brief_items` (8), `pattern_learning` ({enabled, min_observations: 3, decay_days: 30}).

### Care modes

| Mode | Overrides |
|---|---|
| `focus` | `urgency_threshold=high`, nudge cap ≤2, capture ≤conservative, no borderline asks |
| `fundraising` | `urgency_threshold=medium`, boosted topics (investor, series a/b, term sheet, board, legal, due diligence, cap table), nudge cap ≥5, `escalate_after_deferrals=2` |
| `travel` | `urgency_threshold=high`, quiet hours 21–09, capture ≤conservative, nudge cap ≤2 |
| `heads_down` | `urgency_threshold=high`, nudge cap ≤1, capture off, items not surfaced in chat |

### Overrides

Put any of these in `"overrides"` (or use Settings → Fine-tuning / the `care_set_override` tool):
`max_nudges_per_day`, `quiet_hours`, `urgency_threshold`, `commitment_capture`, `ask_on_borderline`, `research_followups`, `boosted_topics`, `muted_topics`, `auto_filter_categories`, `resurface_after_hours`, `escalate_after_deferrals`, `item_max_age_days`, `morning_review_dm`, `surface_care_items_in_chat`.

## Connectors

A connector is **active** when `connectors.<name>` is `true` **and** its credentials exist:

| Connector | Credentials | Tools it unlocks |
|---|---|---|
| `web_search` | `TAVILY_API_KEY` | `tavily_search` |
| `gmail` | `credentials.json` (+ `token.json` after sign-in) | `list_emails`, `read_email`, `send_email`, `reply_to_email` |
| `calendar` | same | `list/create/update/delete_calendar_event` |
| `notion` | `NOTION_API_KEY` | `search_notion`, `read_notion_page`, `create_notion_page`, `update_notion_page`, `query_notion_database`, `create_database_entry` |
| `browser` | `BROWSER_TOKEN` + Chrome extension | `browser_*` |

Inactive connectors' tools are removed from the model's tool list (and from sub-agents), and the system prompt tells the model they're not connected. Google file locations can be overridden with `GOOGLE_CREDENTIALS_FILE` / `GOOGLE_TOKEN_FILE`.

`notion.tasks_database_id` lets the morning review query your task database directly. `tasks_schema_hint` is free text describing its properties (e.g. which ones are `status` vs `select` types).

## Care registry item

```json
{
  "id": "care_1a2b3c", "type": "email|task|commitment|followup|calendar",
  "source": "gmail|notion|calendar|conversation|morning_review|manual", "source_ref": "<gmail id / notion page id>",
  "title": "…", "details": "…", "sender": "…", "topics": ["…"], "category": "…",
  "urgency": "high|medium|low", "confidence": 1.0,
  "status": "new|acknowledged|in_progress|deferred|snoozed|done|dismissed|archived",
  "user_intent": "will reply tonight", "intent_set_at": "…", "snooze_until": null, "deadline": "…",
  "reminder_count": 0, "deferral_count": 0, "last_nudged_at": null,
  "created_at": "…", "last_updated": "…", "resolved_at": null, "history": [ … ]
}
```

`tend()` (run before every morning review) is deterministic:

- `snoozed` past `snooze_until` → `acknowledged` (resurfaced)
- `deferral_count ≥ escalate_after_deferrals` → urgency +1 (once)
- deadline passed, or a "tonight/today" intent still open the next day → flagged `_overdue`, urgency high
- untouched for `item_max_age_days` → archived (low/medium) or flagged stale (high / has deadline)

## Nudge enforcement (in code, not just in the prompt)

`schedule_notifications` applies, for every nudge: level `off` → refused; quiet hours → moved to the end of the window (unless `priority: "critical"`); at least 30 minutes from any other queued nudge; daily cap counting both queued and already-delivered nudges; the morning review may only use `cap − 1` so the pre-exit flow can still add one.

## Agent tools for the registry and config

`care_list`, `care_digest`, `care_find`, `care_add_item`, `care_update_item`, `care_resolve_item`, `care_record_feedback`, `care_patterns`, `care_generate_brief`, `care_mark_source_checked`, `care_get_config`, `care_set_proactiveness`, `care_set_mode`, `care_set_override`. Sub-agents can read/update the registry but cannot change the level or mode.

## HTTP API

| Method & path | Purpose |
|---|---|
| `GET /api/state` | Runtime snapshot (sleeping, level, connectors, queued nudges, counts) |
| `POST /api/chat` `{text}` | Send a message (WebSocket `/ws/ui` is what the UI uses) |
| `GET /api/history`, `DELETE /api/history` | Transcript |
| `GET /api/config`, `PUT /api/config` | Read / update level, mode, connectors, overrides, user, llm, notion, morning_review |
| `GET /api/care?status=open|closed|all` | Registry items |
| `POST /api/care` `{title,type,urgency,deadline}` | Add an item |
| `POST /api/care/{id}` `{action: done|dismiss|defer|snooze|acknowledge|in_progress|reopen|urgency|delete, until?, note?, intent?}` | Act on an item |
| `GET /api/care/patterns`, `POST/DELETE /api/care/patterns/rules…` | Learned patterns and rules |
| `DELETE /api/nudges/{id}` | Cancel a queued nudge or reminder |
| `POST /api/morning-review/run` | Run the review now |
| `POST /api/sleep` | End the session now (pre-exit flow) |
| `POST /api/connectors/google/connect` | Launch Google sign-in |
| `GET /api/health` | Readiness |
