"""Offline Antigravity accounting boundaries; synthetic events mirror the verified 1.5.18 schemas."""
import json

import pytest

from howhard.accounting import CATEGORIES, collect
from howhard.validation import HowHardError

MODEL = "agy/gemini-3-pro"
SLUG = "gemini-3-pro"


def counts(**values):
    return {k: values.get(k, 0) for k in CATEGORIES}


def card(**changes):
    return {"format": "howhard-rate-card-v1", "source": "fixture://agy", "version": "fixture-v1",
            "effective_at": "2026-11-07T00:00:00+00:00", "provider": "fixture-provider",
            "model": MODEL, "service_tier": None, "basis": "estimate",
            "rates": counts(input=2, output=8, cache_read=.5, cache_write=None, cache_write_5m=None, cache_write_1h=None),
            **changes}


def write(tmp_path, events, name="owned-agy.jsonl"):
    path = tmp_path / name
    path.write_bytes(b"".join(json.dumps(e).encode() + b"\n" for e in events))
    return path


def manifest(tmp_path, events, *, adapter="agy-print-v1", **changes):
    path = write(tmp_path, events)
    return {"format": "howhard-usage-manifest-v1", "execution_id": "execution", "host_version": "1.5.18",
            "adapter": adapter, "source": str(path), "session_id": "conversation", "scope": "self", "start": 0,
            "end": path.stat().st_size, "completed": True, "provider": "fixture-provider",
            "model": MODEL, "service_tier": None, **changes}


def usage(**changes):
    return {"input_tokens": 100, "output_tokens": 30, "thinking_tokens": 10, "cache_read_tokens": 40,
            "total_tokens": 130, **changes}


def init(**changes):
    return {"event": "init", "conversation_id": "conversation", "init": {"model": SLUG, **changes}}


def step(**changes):
    return {"event": "step_update", "step_update": {"conversation_id": "conversation", "step_index": 1, **changes}}


def result(*, status="SUCCESS", num_turns=1, **usage_changes):
    return {"event": "result", "result": {"conversation_id": "conversation", "status": status, "num_turns": num_turns,
                                          "duration_seconds": 2.5, "usage": usage(**usage_changes)}}


def fresh(**changes):
    return [init(), step(), result(**changes)]


def test_agy_print_fresh_single_turn_counts_cache_and_thinking_once(tmp_path):
    out = collect(manifest(tmp_path, fresh()))
    assert out["status"] == "complete" and out["provenance"]["completed"] is True
    [segment] = out["segments"]
    assert segment["tokens"] == counts(input=100, output=30, cache_read=40)
    assert segment["reported_tokens"] == 170
    assert segment["model"] == MODEL
    assert segment["source_id"] == "conversation:agy-print-v1:0:result"
    assert "baseline_sha256" not in out["provenance"]


def test_agy_print_resumed_span_subtracts_explicit_baseline(tmp_path):
    prior = [init(), result(num_turns=1, input_tokens=60, output_tokens=10, thinking_tokens=4,
                            cache_read_tokens=20, total_tokens=70)]
    start = len(b"".join(json.dumps(e).encode() + b"\n" for e in prior))
    base = {"conversation_id": "conversation", "usage": prior[1]["result"]["usage"]}
    out = collect(manifest(tmp_path, prior + [step(), result(num_turns=2)], start=start, baseline=base))
    assert out["status"] == "complete"
    assert out["segments"][0]["tokens"] == counts(input=40, output=20, cache_read=20)
    assert out["segments"][0]["source_id"] == f"conversation:agy-print-v1:{start}:result"
    assert len(out["provenance"]["baseline_sha256"]) == 64


def test_agy_print_empty_baseline_starts_at_zero(tmp_path):
    out = collect(manifest(tmp_path, fresh(num_turns=3), baseline={"conversation_id": "conversation", "usage": {}}))
    assert out["status"] == "complete"
    assert out["segments"][0]["tokens"] == counts(input=100, output=30, cache_read=40)


@pytest.mark.parametrize("events", [
    [step(), result()],                 # no init: not a verified fresh session
    [init(), step(), result(num_turns=2)],  # multi-turn cumulative totals
    [init(), step(), {"event": "result", "result": {k: v for k, v in result()["result"].items() if k != "num_turns"}}],
])
def test_agy_print_unverified_cumulative_totals_without_baseline_are_unavailable(tmp_path, events):
    out = collect(manifest(tmp_path, events))
    assert out["status"] == "unavailable" and out["segments"] == []


def test_agy_print_resumed_span_without_baseline_is_unavailable(tmp_path):
    prior = [init(), result()]
    start = len(b"".join(json.dumps(e).encode() + b"\n" for e in prior))
    out = collect(manifest(tmp_path, prior + [init(), result(num_turns=1)], start=start))
    assert out["status"] == "unavailable" and out["segments"] == []


def test_agy_print_negative_delta_is_a_cumulative_reset(tmp_path):
    base = {"conversation_id": "conversation", "usage": usage(input_tokens=200, total_tokens=230)}
    with pytest.raises(HowHardError, match="reset"):
        collect(manifest(tmp_path, fresh(), baseline=base))


@pytest.mark.parametrize("base", [
    {"conversation_id": "other", "usage": {}},
    {"conversation_id": "conversation"},
    {"conversation_id": "conversation", "usage": {}, "extra": 1},
    {"conversation_id": "conversation", "usage": usage(thinking_tokens=99)},
])
def test_agy_print_malformed_baseline_is_refused(tmp_path, base):
    with pytest.raises(HowHardError):
        collect(manifest(tmp_path, fresh(), baseline=base))


@pytest.mark.parametrize("status", ["ERROR", "CANCELLED", None])
def test_agy_print_non_success_result_is_partial(tmp_path, status):
    out = collect(manifest(tmp_path, fresh(status=status)))
    assert out["status"] == "partial" and out["provenance"]["completed"] is False
    assert out["segments"][0]["tokens"] == counts(input=100, output=30, cache_read=40)


def test_agy_print_trailing_step_after_result_is_partial(tmp_path):
    out = collect(manifest(tmp_path, fresh() + [step(step_index=2)]))
    assert out["status"] == "partial"


@pytest.mark.parametrize("changes", [{"thinking_tokens": 31}, {"total_tokens": 170}, {"input_tokens": -1},
                                      {"cache_read_tokens": True}, {"output_tokens": None}, {"num_turns": "1"},
                                      {"num_turns": -1}])
def test_agy_print_token_invariants_are_refused(tmp_path, changes):
    turns = changes.pop("num_turns", 1)
    with pytest.raises(HowHardError):
        collect(manifest(tmp_path, [init(), result(num_turns=turns, **changes)]))


def test_agy_print_thinking_and_total_are_optional(tmp_path):
    event = result()
    del event["result"]["usage"]["thinking_tokens"], event["result"]["usage"]["total_tokens"]
    out = collect(manifest(tmp_path, [init(), event]))
    assert out["status"] == "complete" and out["segments"][0]["reported_tokens"] == 170


@pytest.mark.parametrize("events", [
    [{"event": "init", "conversation_id": "other", "init": {"model": SLUG}}, result()],
    [init(), {"event": "step_update", "step_update": {"conversation_id": "other"}}, result()],
    [init(), {"event": "result", "result": {**result()["result"], "conversation_id": "other"}}],
    [init(model="other-model"), result()],
    [{"event": "init", "conversation_id": "conversation", "init": "bad"}, result()],
    [init(), {"event": "step_update", "step_update": "bad"}, result()],
    [init(), {"event": "result", "result": {**result()["result"], "usage": "bad"}}],
    [init(), {"event": 3}, result()],
    [init(), result(), result()],
])
def test_agy_print_identity_and_schema_mismatches_are_refused(tmp_path, events):
    with pytest.raises(HowHardError):
        collect(manifest(tmp_path, events))


def test_agy_print_bare_and_colon_manifest_models_match_init(tmp_path):
    assert collect(manifest(tmp_path, fresh(), model=SLUG))["segments"][0]["model"] == SLUG
    assert collect(manifest(tmp_path, fresh(), model="agy:" + SLUG))["segments"][0]["model"] == "agy:" + SLUG


def test_agy_print_rate_card_prices_qualified_model(tmp_path):
    out = collect(manifest(tmp_path, fresh(), rate_card=card()))
    estimate = out["segments"][0]["estimated_cost"]
    assert estimate["usd"] == pytest.approx((100 * 2 + 30 * 8 + 40 * .5) / 1e6)
    assert estimate["source"] == "fixture://agy" and estimate["rate_card"]["model"] == MODEL
    partial = collect(manifest(tmp_path, fresh(status="ERROR"), rate_card=card()))
    assert partial["segments"][0]["estimated_cost"] is None
    with pytest.raises(HowHardError, match="rate_card"):
        collect(manifest(tmp_path, fresh(), rate_card=card(model=SLUG)))


def test_agy_print_span_cannot_be_claimed_by_two_executions(tmp_path):
    from howhard import Engine
    engine = Engine(directory=tmp_path / "engine")
    m = manifest(tmp_path, fresh())
    engine.observe("task", MODEL, "high", "clean", native=True, id="execution", execution_usage=collect(m))
    engine.observe("task", MODEL, "high", "clean", native=True, id="other")
    with pytest.raises(HowHardError, match="overlap|belongs|claim|source"):
        engine.attach_usage("other", {**m, "execution_id": "other"}, id="other-usage")


def planner(index, *, status="DONE", model=SLUG, **changes):
    return {"step_index": index, "type": "PLANNER_RESPONSE", "status": status, "model": model,
            "input_tokens": 100, "cache_read_tokens": 40, "output_tokens": 30, "thinking_tokens": 10, **changes}


def transcript(*steps):
    return [{"step_index": 0, "type": "USER_INPUT", "status": "DONE"}, *steps]


def test_agy_transcript_aggregates_planner_steps_with_qualified_model_and_price(tmp_path):
    events = transcript(planner(1), {"step_index": 2, "type": "GENERIC", "status": "DONE"},
                        planner(3, input_tokens=10, cache_read_tokens=0, output_tokens=5, thinking_tokens=0))
    out = collect(manifest(tmp_path, events, adapter="agy-transcript-v1", rate_card=card()))
    assert out["status"] == "complete"
    assert [s["source_id"] for s in out["segments"]] == ["conversation:agy-transcript-v1:step-1",
                                                          "conversation:agy-transcript-v1:step-3"]
    assert [s["tokens"] for s in out["segments"]] == [counts(input=100, output=30, cache_read=40),
                                                       counts(input=10, output=5)]
    assert all(s["model"] == MODEL for s in out["segments"])
    assert sum(s["estimated_cost"]["usd"] for s in out["segments"]) == pytest.approx(
        (110 * 2 + 35 * 8 + 40 * .5) / 1e6)


def test_agy_transcript_revised_step_replaces_earlier_record(tmp_path):
    events = transcript(planner(1, status="RUNNING", output_tokens=5, thinking_tokens=0), planner(1, output_tokens=50))
    out = collect(manifest(tmp_path, events, adapter="agy-transcript-v1"))
    assert out["status"] == "complete" and len(out["segments"]) == 1
    assert out["segments"][0]["tokens"] == counts(input=100, output=50, cache_read=40)


@pytest.mark.parametrize("events", [
    transcript(planner(1, model="other-model")),
    transcript(planner(1, thinking_tokens=31)),
    transcript(planner(1, input_tokens=None)),
    transcript(planner(1, status=None)),
    transcript({**planner(1), "step_index": -1}),
    transcript({**planner(1), "model": None}),
    [{"type": "USER_INPUT"}],
    [{"step_index": 0}],
    transcript(planner(3), planner(2)),
])
def test_agy_transcript_malformed_or_foreign_steps_are_refused(tmp_path, events):
    with pytest.raises(HowHardError):
        collect(manifest(tmp_path, events, adapter="agy-transcript-v1"))


@pytest.mark.parametrize("events", [
    transcript(planner(1), planner(2, status="RUNNING")),
    transcript(planner(1, status="ERROR"), planner(2)),
])
def test_agy_transcript_unfinished_generation_is_partial(tmp_path, events):
    out = collect(manifest(tmp_path, events, adapter="agy-transcript-v1", rate_card=card()))
    assert out["status"] == "partial"
    assert all(s["estimated_cost"] is None for s in out["segments"])


def test_agy_transcript_without_generation_is_unavailable(tmp_path):
    out = collect(manifest(tmp_path, transcript(), adapter="agy-transcript-v1"))
    assert out["status"] == "unavailable"


def test_agy_transcript_refuses_shared_parent_agents_and_unsupported_versions(tmp_path):
    with pytest.raises(HowHardError, match="included_agents"):
        collect(manifest(tmp_path, transcript(planner(1)), adapter="agy-transcript-v1", scope="inclusive",
                         included_agents=["agent"]))
    with pytest.raises(HowHardError, match="host_version"):
        collect(manifest(tmp_path, transcript(planner(1)), adapter="agy-transcript-v1", host_version="1.5.17"))
