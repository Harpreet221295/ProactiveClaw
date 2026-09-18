from __future__ import annotations

from care.patterns import CarePatterns


def test_learned_dismissals_filter_after_min_observations(write_config):
    write_config()
    p = CarePatterns()
    item = {"title": "Weekly digest", "sender": "news@x.com"}
    for _ in range(2):
        p.observe("sender", "news@x.com", "dismissed")
    assert p.score(item)["filter"] is False     # below min_observations=3
    p.observe("sender", "news@x.com", "dismissed")
    s = p.score(item)
    assert s["filter"] is True and s["adjust"] == -1


def test_acted_boosts_and_rules(write_config):
    write_config()
    p = CarePatterns()
    for _ in range(3):
        p.observe("sender", "maya@seq.com", "acted")
    urg, filtered, reasons = p.apply("medium", {"title": "hi", "sender": "maya@seq.com"})
    assert urg == "high" and not filtered

    p.add_rule("topic", "term sheet", "boost")
    assert p.apply("low", {"title": "Term sheet v3 ready"})[0] == "medium"
    p.add_rule("sender", "spam.com", "mute")
    assert p.apply("high", {"title": "Deal", "sender": "bob@spam.com"})[1] is True
    assert p.remove_rule("sender", "spam.com") and not p.remove_rule("sender", "spam.com")


def test_config_topics_and_categories(write_config):
    write_config(overrides={"boosted_topics": ["investor"], "muted_topics": ["webinar"]})
    p = CarePatterns()
    assert p.apply("medium", {"title": "Investor update"})[0] == "high"
    assert p.apply("medium", {"title": "Free webinar tomorrow"})[1] is True
    assert p.apply("medium", {"title": "x", "category": "newsletter"})[1] is True


def test_pattern_learning_can_be_disabled_via_config_file(write_config):
    # Not exposed through set_override (too niche for the UI) but honoured from config.json
    write_config(overrides={"pattern_learning": {"enabled": False}})
    p = CarePatterns()
    for _ in range(3):
        p.observe("sender", "news@x.com", "dismissed")
    assert p.score({"sender": "news@x.com"})["filter"] is False
    # explicit rules still apply even with learning off
    p.add_rule("sender", "news@x.com", "mute")
    assert p.score({"sender": "news@x.com"})["filter"] is True


def test_observe_item_and_digest(write_config):
    write_config()
    p = CarePatterns()
    keys = p.observe_item({"sender": "a@b.com", "topics": ["board"], "source": "gmail"}, "acted")
    assert set(keys) == {"sender:a@b.com", "topic:board", "source:gmail"}
    p.add_rule("topic", "legal", "boost", note="always")
    d = p.digest()
    assert "boost topic 'legal'" in d
