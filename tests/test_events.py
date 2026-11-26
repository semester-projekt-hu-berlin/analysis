"""Automatic events: detection, selection, evidence, LLM output validation, fallback, writing, privacy."""

import json
from dataclasses import replace
from datetime import date, timedelta

import anthropic
import httpx2
import pytest

from analysis import events as ev
from analysis import pipeline
from analysis.textutil import today_berlin
from tests.fakes import FakeFirestore
from tests.test_pipeline import fs as fs

TODAY = today_berlin()
CFG = replace(ev.Config(), data_since="2000-01-01", llm_seconds=30)
SECRET_COMMENT = "Geheimer Kommentartext von Nutzerin XY 4711"


def day(n: int) -> str:
    return (date.fromisoformat(TODAY) - timedelta(days=n)).isoformat()


def series(base: int, base_avg: float, last: int, last_avg: float, days: int = 20) -> dict:
    """recentDays: `base` mentions a day for `days` days, then `last` mentions today."""
    out = {day(k): {"mentions": base, "scoreSum": round(base * base_avg, 4)} for k in range(1, days + 1) if base}
    out[TODAY] = {"mentions": last, "scoreSum": round(last * last_avg, 4)}
    return out


# --------------------------------------------------------------------------- detection


def test_spike_threshold():
    s = series(10, -0.3, 30, -0.3)  # mean 10/day: threshold max(25, 30) = 30
    assert ev.check_day(s, TODAY, CFG)[0] == "spike"
    s[TODAY]["mentions"] = 29
    assert ev.check_day(s, TODAY, CFG) is None
    # small entities: at least 25 mentions even from nothing
    assert ev.check_day(series(0, 0, 24, 0), TODAY, CFG) is None
    assert ev.check_day(series(0, 0, 25, 0), TODAY, CFG)[0] == "spike"


def test_rise_and_fall_thresholds():
    assert ev.check_day(series(10, -0.5, 20, -0.2), TODAY, CFG)[0] == "rise"
    assert ev.check_day(series(10, -0.5, 20, -0.8), TODAY, CFG)[0] == "fall"
    assert ev.check_day(series(10, -0.5, 20, -0.3), TODAY, CFG) is None  # delta 0.2 < 0.25
    assert ev.check_day(series(10, -0.5, 19, -0.1), TODAY, CFG) is None  # too few mentions that day
    s = series(2, -0.5, 20, -0.1, days=14)  # 28 mentions before < 30
    assert ev.check_day(s, TODAY, CFG) is None


def test_no_baseline_before_data_collection():
    cfg = replace(CFG, data_since=day(5))  # only 5 days of data before today
    assert ev.check_day(series(1, 0, 60, 0), TODAY, cfg) is None
    cfg = replace(CFG, data_since=day(7))
    assert ev.check_day(series(1, 0, 60, 0), TODAY, cfg)[0] == "spike"


def test_detect_skips_non_allowlisted_and_ranks():
    docs = {
        "a": {"name": "A", "recentDays": series(10, 0, 100, 0)},
        "b": {"name": "B", "recentDays": series(10, 0, 40, 0)},
        "c": {"name": "C", "allowlisted": False, "recentDays": series(10, 0, 100, 0)},
    }
    got = ev.detect(docs, [TODAY], CFG)
    assert [c.entity_id for c in got] == ["a", "b"]


def cand(eid, d, kind="spike", strength=1.0, mentions=50):
    return ev.Candidate(eid, eid, d, kind, mentions, 10.0, -0.3, -0.4, strength)


def test_select_curated_wins_gap_and_caps():
    existing = {
        "curated-1": {"day": day(0), "entityIds": ["a"]},  # no `source`: curated
        "auto-x": {"day": day(3), "entityIds": ["b"], "source": "auto"},
    }
    cands = [
        cand("a", day(0), strength=5),  # curated that day
        cand("b", day(1)),  # within 3 days of an existing auto event
        cand("b", day(5), strength=2),  # 2 days before it: also blocked
        cand("c", day(2), strength=1),
        cand("c", day(1), strength=9),  # a stronger day later; the earlier one wins
        cand("d", day(0)),
        cand("e", day(6)),  # same day as an existing auto event of another entity
    ]
    chosen, skipped = ev.select(cands, existing, CFG, limit=10)
    assert {(c.entity_id, c.day) for c in chosen} == {("c", day(2)), ("d", day(0)), ("e", day(6))}
    assert skipped == {"curated": 1, "gap": 3, "cap": 0}
    # the auto event of the same day does not block (that is the event a re-run updates)
    chosen, _ = ev.select([cand("b", day(3))], existing, CFG, limit=10)
    assert len(chosen) == 1
    many = [cand(f"e{i}", day(0), strength=i) for i in range(10)]
    chosen, skipped = ev.select(many, {}, CFG, limit=4)
    assert [c.entity_id for c in chosen] == ["e9", "e8", "e7", "e6"] and skipped["cap"] == 6


# --------------------------------------------------------------------------- validation


def test_validate_llm_items():
    g = (["Rockstar veröffentlicht GTA 6", "Andere Meldung", "Dritte Meldung"], ["GTA 6"])
    assert ev.validate(
        {"headline": 0, "title": "Release von GTA 6", "description": "Rockstar hat GTA 6 veröffentlicht."}, *g
    ) == (
        0,
        "Release von GTA 6",
        "Rockstar hat GTA 6 veröffentlicht.",
    )
    assert ev.validate({"headline": None}, *g) == "decline"
    bad = [
        {"headline": 3, "title": "Release von GTA 6", "description": "Rockstar hat GTA 6 veröffentlicht."},
        {"headline": True, "title": "Release von GTA 6", "description": "Rockstar hat GTA 6 veröffentlicht."},
        {"headline": "0", "title": "Release von GTA 6", "description": "Rockstar hat GTA 6 veröffentlicht."},
        {"headline": 0, "title": "x" * 81, "description": "Rockstar hat GTA 6 veröffentlicht."},
        {
            "headline": 0,
            "title": "eins zwei drei vier fünf sechs sieben acht neun zehn elf",
            "description": "Ein Satz hier.",
        },
        {"headline": 0, "title": "Release von GTA 6", "description": "x" * 221},
        {"headline": 0, "title": "Release von GTA 6", "description": "Mehr unter https://example.org/gta."},
        {"headline": 0, "title": "Skandal um Bundeskanzlerin", "description": "Ein erfundener Zusammenhang hier."},
        {"headline": 0, "title": 5, "description": "Rockstar hat GTA 6 veröffentlicht."},
        "kein dict",
    ]
    for item in bad:
        assert ev.validate(item, *g) is None, item


def test_clean_and_shorten_headlines():
    assert ev.clean_headline("GTA 6 ist da | ComputerBase", "ComputerBase") == "GTA 6 ist da"
    assert ev.clean_headline("Merz - Kanzler unter Druck", "DER SPIEGEL") == "Merz - Kanzler unter Druck"
    s = ev.shorten("wort " * 40, 80)
    assert len(s) <= 80 and s.endswith("…")
    assert ev.mentions_entity("Rockstar veröffentlicht GTA 6", ["GTA 6"])
    assert not ev.mentions_entity("Agentur meldet", ["GTA 6"])


# --------------------------------------------------------------------------- end to end on a fake Firestore


class FakeLLM:
    """The Messages API behind the real SDK client: records the requests, answers with `answer(case)`, an
    HTTP status or another stop reason."""

    def __init__(self, answer=None, status=200, stop_reason="end_turn"):
        self.requests: list[dict] = []
        self.answer = answer
        self.status = status
        self.stop_reason = stop_reason

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append({"url": str(request.url), "headers": dict(request.headers), "body": body})
        if self.status != 200:
            err = {"type": "error", "error": {"type": "permission_error", "message": "no access to this model"}}
            return httpx2.Response(self.status, json=err, headers={"retry-after-ms": "1"})
        cases = json.loads(body["messages"][0]["content"])["cases"]
        text = json.dumps({"results": [self.answer(c) for c in cases]}, ensure_ascii=False)
        msg = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": [{"type": "text", "text": text}],
            "stop_reason": self.stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        return httpx2.Response(200, json=msg)


def gta_answer(case):
    hl = case["headlines"]
    if case["kind"] != "spike":
        return {"id": case["id"], "headline": None}
    return {
        "id": case["id"],
        "headline": 0,
        "title": "Release von GTA 6",
        "description": f"Laut {hl[0]['source']} ist GTA 6 erschienen.",
    }


def provider(llm, name="claude"):
    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(llm))
    return ev.ClaudeProvider(name, ev.DEFAULT_MODEL, anthropic.Anthropic(api_key="tok", http_client=http))


@pytest.fixture
def world():
    docs = {
        "sources/computerbase": {"name": "ComputerBase"},
        "sources/spiegel": {"name": "DER SPIEGEL"},
        "sources/mastodon": {"name": "norden.social", "sourceType": "social"},
        "articles/a1": {
            "url": "https://cb.example/gta6",
            "title": "GTA 6 ist da: Rockstar startet den Verkauf | ComputerBase",
            "sourceId": "computerbase",
        },
        "articles/a2": {
            "url": "https://spiegel.example/x",
            "title": "Wie gut ist GTA 6 wirklich geworden?",
            "sourceId": "spiegel",
        },
        "articles/a3": {
            "url": "https://norden.social/public/local?day=x",
            "title": "norden.social · lokale Timeline",
            "sourceId": "mastodon",
            "sourceType": "social",
        },
        "articles/b1": {
            "url": "https://spiegel.example/merz",
            "title": "Merz kündigt Rentenreform an",
            "sourceId": "spiegel",
        },
        "comments/k1": {"articleId": "a1", "text": SECRET_COMMENT},
        "events/2000-curated": {
            "day": day(0),
            "title": "Kuratiert",
            "description": "x",
            "entityIds": ["cdu"],
            "url": None,
        },
    }
    top = {"a3": {"mentions": 50}, "a1": {"mentions": 30}, "a2": {"mentions": 10}}
    for e in ("gta-vi", "rockstar-games", "cdu"):
        docs[f"entities/{e}/days/{TODAY}"] = {"topArticles": top, "mentions": 80}
    docs[f"entities/friedrich-merz/days/{TODAY}"] = {"topArticles": {"b1": {"mentions": 12}}}
    entity_docs = {
        "gta-vi": {"name": "GTA 6", "aliases": ["GTA VI"], "recentDays": series(5, 0, 80, 0)},
        "rockstar-games": {"name": "Rockstar Games", "aliases": ["Rockstar"], "recentDays": series(3, 0, 40, 0)},
        "cdu": {"name": "CDU", "recentDays": series(5, 0, 80, 0)},
        "friedrich-merz": {"name": "Friedrich Merz", "recentDays": series(40, -0.4, 30, -0.75)},
        "quiet": {"name": "Quiet", "recentDays": series(10, 0, 12, 0)},
    }
    return FakeFirestore(docs), entity_docs


def test_events_end_to_end(world):
    fs, entity_docs = world
    llm = FakeLLM(gta_answer)
    out = ev.process(fs, entity_docs, day(1), TODAY, cfg=CFG, providers=[provider(llm)])
    autos = {k: v for k, v in fs.docs.items() if k.startswith("events/auto-")}
    # GTA 6 and Rockstar share their top headline: one event; CDU has a curated event that day;
    # Merz (fall) was declined by the model: no event
    assert list(autos) == [f"events/auto-{TODAY}-gta-vi"]
    e = autos[f"events/auto-{TODAY}-gta-vi"]
    assert e["entityIds"] == ["gta-vi", "rockstar-games"]
    assert e["source"] == "auto" and e["kind"] == "spike"
    assert e["title"] == "Release von GTA 6" and e["description"] == "Laut ComputerBase ist GTA 6 erschienen."
    assert e["url"] == "https://cb.example/gta6" and e["sourceName"] == "ComputerBase"
    assert e["generator"] == {"model": "claude-haiku-5-5", "version": ev.GENERATOR_VERSION}
    assert e["stats"]["gta-vi"] == {
        "mentions": 80,
        "mentionsBefore": 5.0,
        "avgSentiment": 0.0,
        "avgSentimentBefore": 0.0,
    }
    assert set(e["stats"]) == {"gta-vi", "rockstar-games"}
    assert e["createdAt"] == e["updatedAt"]
    assert out["skipped"]["curated"] == 1 and out["llm"]["declined"] == 1
    # the social pseudo-article is never a headline; the suffix of the source is removed
    cases = json.loads(llm.requests[0]["body"]["messages"][0]["content"])["cases"]
    gta = next(c for c in cases if c["entities"][0] == "GTA 6")
    assert [h["title"] for h in gta["headlines"]] == [
        "GTA 6 ist da: Rockstar startet den Verkauf",
        "Wie gut ist GTA 6 wirklich geworden?",
    ]
    assert gta["entities"] == ["GTA 6", "Rockstar Games"]


def test_privacy_no_comment_text_in_requests_or_reads(world):
    fs, entity_docs = world
    llm = FakeLLM(gta_answer)
    ev.process(fs, entity_docs, day(1), TODAY, cfg=CFG, providers=[provider(llm)])
    blob = json.dumps(llm.requests, ensure_ascii=False)
    assert SECRET_COMMENT not in blob and "Nutzerin" not in blob
    assert not fs.reads_by.get("comments") and not fs.reads_by.get("mentions")


def test_rerun_is_idempotent_and_keeps_the_explanation(world):
    fs, entity_docs = world
    llm = FakeLLM(gta_answer)
    ev.process(fs, entity_docs, day(1), TODAY, cfg=CFG, providers=[provider(llm)])
    before = {k: v for k, v in fs.docs.items() if k.startswith("events/")}
    n_req, w0 = len(llm.requests), fs.writes
    out = ev.process(fs, entity_docs, day(1), TODAY, cfg=CFG, providers=[provider(llm)])
    assert {k: v for k, v in fs.docs.items() if k.startswith("events/")} == before
    assert fs.writes == w0 and out["llm"]["kept"] == 1
    # only the declined Merz case is asked again, the GTA explanation is kept
    cases = [c for r in llm.requests[n_req:] for c in json.loads(r["body"]["messages"][0]["content"])["cases"]]
    assert [c["entities"] for c in cases] == [["Friedrich Merz"]]
    # an extractive run (no LLM) never overwrites the model's text, but updates the numbers
    entity_docs["gta-vi"]["recentDays"][TODAY]["mentions"] = 90
    ev.process(fs, entity_docs, day(1), TODAY, cfg=CFG, providers=[])
    e = fs.docs[f"events/auto-{TODAY}-gta-vi"]
    assert e["title"] == "Release von GTA 6" and e["generator"]["model"] == "claude-haiku-5-5"
    assert (
        e["stats"]["gta-vi"]["mentions"] == 90 and e["createdAt"] == before[f"events/auto-{TODAY}-gta-vi"]["createdAt"]
    )


def test_regrouping_reuses_the_id(world):
    fs, entity_docs = world
    # an earlier run wrote the event under the Rockstar id
    fs.docs[f"events/auto-{TODAY}-rockstar-games"] = {"day": TODAY, "source": "auto", "entityIds": ["rockstar-games"]}
    ev.process(fs, entity_docs, day(1), TODAY, cfg=CFG, providers=[])
    autos = sorted(k for k in fs.docs if k.startswith("events/auto-"))
    assert autos == [f"events/auto-{TODAY}-rockstar-games"]
    assert fs.docs[autos[0]]["entityIds"] == ["gta-vi", "rockstar-games"]


def test_extractive_fallback_when_the_llm_is_unavailable(world):
    fs, entity_docs = world
    llm = FakeLLM(status=403)
    out = ev.process(fs, entity_docs, day(1), TODAY, cfg=CFG, providers=[provider(llm)])
    assert len(llm.requests) == 1  # 403: the provider is not asked again in this run
    assert out["llm"]["llmErrors"] == 1
    e = fs.docs[f"events/auto-{TODAY}-gta-vi"]
    assert e["generator"]["model"] == "extractive"
    assert e["title"] == "GTA 6 ist da: Rockstar startet den Verkauf"
    assert e["description"].startswith("Viele Erwähnungen am ") and e["description"].endswith("“ (ComputerBase).")
    assert len(e["description"]) <= ev.DESCRIPTION_MAX
    # the sentiment shift of Merz is never explained without a model
    assert not any("friedrich-merz" in k for k in fs.docs if k.startswith("events/"))


def test_invalid_llm_output_falls_back_and_chain_continues(world):
    fs, entity_docs = world

    def broken(case):
        return {"id": case["id"], "headline": 0, "title": "Mehr unter www.example.org", "description": "kurz"}

    out = ev.process(fs, entity_docs, day(1), TODAY, cfg=CFG, providers=[provider(FakeLLM(broken))])
    assert out["llm"]["invalid"] == 2
    assert fs.docs[f"events/auto-{TODAY}-gta-vi"]["generator"]["model"] == "extractive"


def test_a_failing_provider_hands_over_to_the_next(world):
    fs, entity_docs = world
    first, second = FakeLLM(status=500), FakeLLM(gta_answer)
    ev.process(fs, entity_docs, day(1), TODAY, cfg=CFG, providers=[provider(first), provider(second, "second")])
    assert len(first.requests) == 3 and second.requests  # 3 attempts with backoff, then the next provider
    assert fs.docs[f"events/auto-{TODAY}-gta-vi"]["title"] == "Release von GTA 6"


def test_spike_declined_without_entity_in_headline_gives_no_event(world):
    fs, entity_docs = world
    fs.docs["articles/a1"]["title"] = "Agenturmeldung zum Wochenende ohne Namen"
    fs.docs["articles/a2"]["title"] = "Noch eine Meldung ohne den Namen"
    llm = FakeLLM(lambda c: {"id": c["id"], "headline": None})
    ev.process(fs, entity_docs, day(1), TODAY, cfg=CFG, providers=[provider(llm)])
    assert not [k for k in fs.docs if k.startswith("events/auto-")]


def test_dry_run_writes_nothing(world):
    fs, entity_docs = world
    out = ev.process(fs, entity_docs, day(1), TODAY, cfg=CFG, providers=[], dry_run=True)
    assert out["events"] == 1 and out["wouldWrite"] == 1 and out["written"] == 0
    assert fs.writes == 0


def test_run_events_never_raises(world, monkeypatch):
    fs, entity_docs = world

    def boom(*a, **k):
        raise RuntimeError("Firestore 500")

    monkeypatch.setattr(ev, "process", boom)
    assert ev.run_events(fs, entity_docs) == {"error": "RuntimeError"}
    monkeypatch.setenv("EVENTS", "0")
    assert ev.run_events(fs, entity_docs) == {"skipped": "disabled"}


def test_providers_from_env(monkeypatch):
    assert ev.providers_from_env() == []
    monkeypatch.setenv("GITHUB_TOKEN", "t")  # GitHub Models is retired: the job token is never used
    assert ev.providers_from_env() == []
    monkeypatch.setenv("EVENTS_LLM_API_KEY", "sk-ant-test-key")
    ps = ev.providers_from_env()
    assert [(p.name, p.model) for p in ps] == [("claude", "claude-haiku-5-5")]
    assert "sk-ant-test-key" not in repr(ps[0])  # the key never ends up in a log line
    monkeypatch.setenv("EVENTS_LLM_MODEL", "claude-sonnet-5-5")
    assert ev.providers_from_env()[0].model == "claude-sonnet-5-5"
    monkeypatch.setenv("EVENTS_LLM_PROVIDERS", "none")
    assert ev.providers_from_env() == []


def test_backfill_reads_older_day_docs(world, monkeypatch):
    fs, entity_docs = world
    for eid, d in entity_docs.items():
        fs.docs[f"entities/{eid}"] = {**d, "mentions": sum(v["mentions"] for v in d["recentDays"].values()) + 100}
    fs.docs["meta/active"] = {"ids": sorted(entity_docs)}
    old = (date.fromisoformat(TODAY) - timedelta(days=31)).isoformat()
    fs.docs[f"entities/gta-vi/days/{old}"] = {"day": old, "mentions": 100, "scoreSum": 0.0}
    monkeypatch.setenv("EVENTS_DATA_SINCE", "2000-01-01")
    monkeypatch.setenv("EVENTS_LLM_PROVIDERS", "none")
    out = ev.backfill(fs, (date.fromisoformat(TODAY) - timedelta(days=20)).isoformat(), dry_run=True)
    assert out["events"] >= 1 and out["written"] == 0
    assert fs.reads_by.get("entities/days", 0) >= 1
    assert fs.docs["meta/budget"]["reads"]["analysis"] > 0


def test_run_reports_events(fs):
    out = pipeline.run(fs, limit=100, use_wikidata=False)
    assert out["events"]["candidates"] == 0  # a handful of mentions is never an event


def test_request_shape_and_unusable_answers():
    llm = FakeLLM(gta_answer)
    user = [{"role": "user", "content": json.dumps({"cases": []})}]
    assert json.loads(provider(llm).complete(ev.SYSTEM_PROMPT, user)) == {"results": []}
    body = llm.requests[0]["body"]
    assert llm.requests[0]["url"].endswith("/v1/messages")
    assert body["model"] == "claude-haiku-5-5" and body["system"] == ev.SYSTEM_PROMPT and body["messages"] == user
    assert body["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": ev.RESULTS_SCHEMA}}
    assert "temperature" not in body  # rejected by current models
    for stop, msg in (("refusal", "declined"), ("max_tokens", "cut off")):
        with pytest.raises(ev.ProviderError, match=msg) as e:
            provider(FakeLLM(gta_answer, stop_reason=stop)).complete(ev.SYSTEM_PROMPT, user)
        assert not e.value.fatal
    with pytest.raises(ev.ProviderError, match="HTTP 402: no access to this model") as e:
        provider(FakeLLM(status=402)).complete(ev.SYSTEM_PROMPT, user)
    assert e.value.fatal  # no credit left: not asked again in this run
