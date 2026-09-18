"""Fixtures: isolate every on-disk path under tmp_path and give tests a config they control."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).parents[2])
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@pytest.fixture(autouse=True)
def isolated_paths(monkeypatch, tmp_path):
    import core.paths as paths
    import core.config as config
    import care.registry as registry
    import care.patterns as patterns
    import care.brief as brief
    import care.nudges as nudges
    import agents.tools._state as state
    import agents.tools.scheduling as scheduling

    eng = tmp_path / "engagement_data"; eng.mkdir()
    afs = tmp_path / "agent_file_system"; afs.mkdir()
    cfgdir = tmp_path / "config"; cfgdir.mkdir()

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(paths, "ENGAGEMENT_DATA_DIR", eng)
    monkeypatch.setattr(paths, "AGENT_FS_DIR", afs)
    monkeypatch.setattr(paths, "CONFIG_FILE", cfgdir / "config.json")
    monkeypatch.setattr(config, "CONFIG_FILE", cfgdir / "config.json")
    monkeypatch.setattr(config, "GOOGLE_CREDENTIALS_FILE", tmp_path / "credentials.json")
    monkeypatch.setattr(config, "GOOGLE_TOKEN_FILE", tmp_path / "token.json")
    monkeypatch.setattr(registry, "CARE_REGISTRY_FILE", eng / "care_registry.json")
    monkeypatch.setattr(patterns, "CARE_PATTERNS_FILE", eng / "care_patterns.json")
    monkeypatch.setattr(brief, "DAILY_BRIEF_FILE", afs / "daily_brief.json")
    monkeypatch.setattr(nudges, "NUDGE_LOG_FILE", eng / "nudge_log.json")
    monkeypatch.setattr(nudges, "QUEUE_FILE", eng / "queue.json")
    monkeypatch.setattr(state, "QUEUE_FILE", str(eng / "queue.json"))
    monkeypatch.setattr(scheduling, "QUEUE_FILE", str(eng / "queue.json"))
    monkeypatch.setattr(scheduling, "REMINDERS_FILE", str(eng / "reminders.json"))
    monkeypatch.setattr(scheduling, "CRON_JOBS_FILE", str(eng / "cron_jobs.json"))

    # fresh config cache
    config._cache = None
    config._cache_mtime = -1.0
    for var in ("NOTION_API_KEY", "TAVILY_API_KEY", "BROWSER_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    yield tmp_path
    config._cache = None
    config._cache_mtime = -1.0


@pytest.fixture()
def write_config(isolated_paths):
    import core.config as config

    def _write(**overrides):
        cfg = json.loads(json.dumps(config.DEFAULT_CONFIG))
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
        config.save_config(cfg)
        return config.load_config(force=True)
    return _write
