"""Offline public accounting boundaries; synthetic events are not provider claims."""
from copy import deepcopy
import json

import pytest

from howhard.accounting import CATEGORIES, MAX_SOURCE, MAX_SPAN, collect, price, validate
from howhard.validation import HowHardError


def counts(**values):
    return {k: values.get(k, 0) for k in CATEGORIES}


def card(**changes):
    return {"format": "howhard-rate-card-v1", "source": "fixture://contract", "version": "fixture-v1",
            "effective_at": "2026-11-07T00:00:00+00:00", "provider": "fixture-provider",
            "model": "fixture-model", "service_tier": None, "basis": "estimate",
            "rates": counts(input=2, output=8, cache_read=.2, cache_write=3, cache_write_5m=2.5, cache_write_1h=4), **changes}


def artifact(tmp_path, events, *, adapter="codex-exec-v1", **changes):
    path = tmp_path / "owned-events.jsonl"
    path.write_bytes(b"".join(json.dumps(e).encode() + b"\n" for e in events))
    return {"format": "howhard-usage-manifest-v1", "execution_id": "execution", "host_version": "0.166.0" if adapter.startswith("codex") else "2.1.318",
            "adapter": adapter, "source": str(path), "session_id": "session", "scope": "self", "start": 0,
            "end": path.stat().st_size, "completed": True, "provider": "fixture-provider",
            "model": "fixture-model", "service_tier": None, **changes}


def codex_usage(**changes):
    return {"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 20,
            "reasoning_output_tokens": 8, **changes}


def codex_events(**changes):
    return [{"type": "thread.started", "thread_id": "session"},
            {"type": "turn.completed", "usage": codex_usage(**changes)}]


def sdk_model(**changes):
    return {"inputTokens": 100, "outputTokens": 20, "cacheReadInputTokens": 40,
            "cacheCreationInputTokens": 30, "costUSD": .12, **changes}


def sdk_event(**changes):
    return {"type": "result", "uuid": "result-1", "session_id": "session", "subtype": "success",
            "is_error": False, "total_cost_usd": 12, "modelUsage": {"fixture-model": sdk_model()}, **changes}


def sdk_manifest(tmp_path, event=None, **changes):
    return artifact(tmp_path, [event or sdk_event()], adapter="claude-sdk-v1", scope="inclusive",
                    included_sessions=[], baseline={"session_id": "session", "modelUsage": {}}, **changes)


def claude_message(ident="message-1", **changes):
    usage = {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 40,
             "cache_creation_input_tokens": 30, "cache_creation": {
                 "ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 15}}
    return {"type": "assistant", "sessionId": "session", "message": {
        "id": ident, "model": "fixture-model", "stop_reason": "end_turn", "usage": usage, **changes}}


def test_codex_cached_input_and_reasoning_are_counted_once_without_prices(tmp_path):
    result = collect(artifact(tmp_path, codex_events()))
    assert result["status"] == "complete"
    assert result["segments"][0]["tokens"] == counts(input=60, output=20, cache_read=40)
    assert result["segments"][0]["estimated_cost"] is None
    assert result["segments"][0]["billed_cost"] is None
    assert "source" not in result["provenance"]
    assert len(result["provenance"]["artifact_sha256"]) == 64


@pytest.mark.parametrize("changes", [{"cached_input_tokens": 101}, {"reasoning_output_tokens": 21},
                                      {"input_tokens": -1}, {"output_tokens": True}])
def test_impossible_codex_subsets_and_counts_are_refused(tmp_path, changes):
    with pytest.raises(HowHardError):
        collect(artifact(tmp_path, codex_events(**changes)))


def test_missing_category_stays_unknown_and_incomplete_boundary_stays_partial(tmp_path):
    manifest = artifact(tmp_path, codex_events(cached_input_tokens=None), completed=False)
    result = collect(manifest)
    assert result["status"] == "partial"
    assert result["segments"][0]["tokens"]["input"] is None
    assert result["segments"][0]["tokens"]["cache_read"] is None
    manifest = artifact(tmp_path, codex_events() + [{"type": "turn.started"}])
    assert collect(manifest)["status"] == "partial"


def test_claude_ttl_buckets_partition_creation_and_streamed_message_replaces(tmp_path):
    first = claude_message(stop_reason=None)
    final = claude_message()
    final["message"]["usage"]["output_tokens"] = 25
    result = collect(artifact(tmp_path, [first, final], adapter="claude-session-v1"))
    assert result["status"] == "complete" and len(result["segments"]) == 1
    assert result["segments"][0]["tokens"] == counts(input=100, output=25, cache_read=40,
                                                      cache_write=5, cache_write_5m=10, cache_write_1h=15)


def test_claude_invalid_ttl_partition_is_refused(tmp_path):
    event = claude_message()
    event["message"]["usage"]["cache_creation_input_tokens"] = 20
    with pytest.raises(HowHardError, match="TTL"):
        collect(artifact(tmp_path, [event], adapter="claude-session-v1"))


def test_claude_sdk_model_breakdown_is_estimated_never_billed(tmp_path):
    event = sdk_event(modelUsage={"fixture-model": sdk_model(), "other-model": sdk_model(inputTokens=10, costUSD=.02)})
    result = collect(sdk_manifest(tmp_path, event))
    assert result["status"] == "complete"
    assert [s["model"] for s in result["segments"]] == ["fixture-model", "other-model"]
    assert sum(s["estimated_cost"]["usd"] for s in result["segments"]) == pytest.approx(.14)
    assert all(s["billed_cost"] is None for s in result["segments"])
    assert result["segments"][0]["tokens"] == counts(input=100, output=20, cache_read=40, cache_write=30)


def test_sdk_resumed_usage_subtracts_explicit_session_baseline(tmp_path):
    manifest = sdk_manifest(tmp_path)
    manifest["baseline"]["modelUsage"] = {"fixture-model": sdk_model(inputTokens=80, outputTokens=15,
                                                   cacheReadInputTokens=30, cacheCreationInputTokens=25, costUSD=.10)}
    result = collect(manifest)
    assert result["segments"][0]["tokens"] == counts(input=20, output=5, cache_read=10, cache_write=5)
    assert result["segments"][0]["estimated_cost"]["usd"] == pytest.approx(.02)
    assert len(result["provenance"]["baseline_sha256"]) == 64


def test_sdk_unknown_baseline_never_relabels_inherited_totals(tmp_path):
    manifest = sdk_manifest(tmp_path)
    manifest.pop("baseline")
    assert collect(manifest)["status"] == "unavailable"


@pytest.mark.parametrize("field", ["session_id", "inputTokens"])
def test_sdk_mismatched_or_decreasing_baseline_is_refused(tmp_path, field):
    manifest = sdk_manifest(tmp_path)
    if field == "session_id":
        manifest["baseline"][field] = "other-session"
    else:
        manifest["baseline"]["modelUsage"] = {"fixture-model": sdk_model(inputTokens=101)}
    with pytest.raises(HowHardError):
        collect(manifest)


def test_sdk_requires_explicit_inclusive_child_scope(tmp_path):
    manifest = sdk_manifest(tmp_path)
    manifest.pop("included_sessions")
    with pytest.raises(HowHardError, match="included_sessions|child"):
        collect(manifest)
    manifest = sdk_manifest(tmp_path)
    manifest["scope"] = "self"
    with pytest.raises(HowHardError, match="inclusive"):
        collect(manifest)


def test_claude_session_uses_real_model_and_refuses_sidechain_identity(tmp_path):
    first, second = claude_message(), claude_message("message-2", model="other-model")
    result = collect(artifact(tmp_path, [first, second], adapter="claude-session-v1"))
    assert [s["model"] for s in result["segments"]] == ["fixture-model", "other-model"]
    second["isSidechain"] = True
    with pytest.raises(HowHardError, match="scope|session"):
        collect(artifact(tmp_path, [first, second], adapter="claude-session-v1"))


def token_event(n, *, timestamp="2026-11-07T01:00:00Z"):
    return {"timestamp": timestamp, "type": "event_msg", "payload": {"type": "token_count", "info": {
        "total_token_usage": codex_usage(input_tokens=n), "last_token_usage": codex_usage()}}}


def test_codex_session_repeated_cumulative_event_deduplicates_last_usage(tmp_path):
    event = token_event(100)
    events = [{"type": "session_meta", "payload": {"id": "session"}}, event, deepcopy(event),
              {"type": "event_msg", "payload": {"type": "task_complete"}}]
    result = collect(artifact(tmp_path, events, adapter="codex-session-v1"))
    assert result["status"] == "complete" and len(result["segments"]) == 1
    assert result["segments"][0]["tokens"] == counts(input=60, output=20, cache_read=40)


def test_codex_session_cumulative_reset_is_refused(tmp_path):
    with pytest.raises(HowHardError, match="reset"):
        collect(artifact(tmp_path, [token_event(200), token_event(100)], adapter="codex-session-v1"))


@pytest.mark.parametrize("adapter,event", [("codex-exec-v1", {"type": "thread.started", "thread_id": "other"}),
    ("codex-session-v1", {"type": "session_meta", "payload": {"id": "other"}}),
    ("claude-session-v1", {"type": "assistant", "sessionId": "other"})])
def test_bound_source_session_identity_is_checked(tmp_path, adapter, event):
    with pytest.raises(HowHardError, match="session"):
        collect(artifact(tmp_path, [event], adapter=adapter))


@pytest.mark.parametrize("change", [{"start": 1}, {"end": 1}, {"end": MAX_SPAN + 1}, {"completed": "true"}])
def test_span_and_completion_are_explicit_and_bounded(tmp_path, change):
    with pytest.raises(HowHardError):
        collect(artifact(tmp_path, codex_events(), **change))


def test_source_size_limit_missing_file_and_malformed_records(tmp_path):
    manifest = artifact(tmp_path, codex_events())
    source = tmp_path / "owned-events.jsonl"
    with source.open("ab") as stream:
        stream.truncate(MAX_SOURCE + 1)
    with pytest.raises(HowHardError, match="oversized"):
        collect(manifest)
    source.unlink()
    with pytest.raises(HowHardError, match="read failed"):
        collect(manifest)
    source.write_bytes(b"{invalid}\n")
    manifest["end"] = source.stat().st_size
    with pytest.raises(HowHardError, match="JSON"):
        collect(manifest)


def test_category_prices_use_each_nonoverlapping_rate_and_unknown_coverage(tmp_path):
    tokens = counts(input=100, output=20, cache_read=40, cache_write=5, cache_write_5m=10, cache_write_1h=15)
    rates = card()
    assert price(tokens, rates) == pytest.approx((200 + 160 + 8 + 15 + 25 + 60) / 1e6)
    rates["rates"]["cache_write"] = None
    assert price(tokens, rates) is None
    assert price({**tokens, "cache_write": 0}, rates) is not None
    assert price({**tokens, "cache_read": None}, card()) is None
    manifest = artifact(tmp_path, codex_events(), rate_card=card(basis="charge"))
    result = collect(manifest)
    assert result["segments"][0]["estimated_cost"]["usd"] == pytest.approx((120 + 160 + 8) / 1e6)
    assert result["segments"][0]["billed_cost"] is None


def test_rate_card_provider_model_tier_must_match_artifact(tmp_path):
    with pytest.raises(HowHardError, match="mismatch"):
        collect(artifact(tmp_path, codex_events(), rate_card=card(model="other-model")))


def test_incomplete_boundary_cannot_claim_complete_estimate_or_rate_card_billing(tmp_path):
    usage = collect(artifact(tmp_path, codex_events(), rate_card=card()))
    invalid = deepcopy(usage)
    invalid["status"] = "partial"
    invalid["provenance"]["completed"] = False
    with pytest.raises(HowHardError, match="incomplete|boundary"):
        validate(invalid)
    invalid = deepcopy(usage)
    invalid["segments"][0]["billed_cost"] = invalid["segments"][0]["estimated_cost"]
    with pytest.raises(HowHardError, match="estimate|billed"):
        validate(invalid)


def test_attach_persists_once_preserves_outcome_and_retries_after_later_outcome(tmp_path):
    from howhard import Engine
    engine = Engine(directory=tmp_path)
    engine.observe("task", "fixture-model", "high", "clean", native=True, id="execution")
    original = deepcopy(engine.evidence("execution", raw=True))
    manifest = artifact(tmp_path, codex_events())
    attached = engine.attach_usage("execution", manifest, id="usage-1")
    assert len(engine.raw_rows) == 2 and len(engine.rows) == 1
    assert engine.evidence("execution", raw=True) == original
    assert engine.evidence("execution")["execution_usage"] == collect(manifest)
    engine.observe("task", "fixture-model", "high", "loud", amends="usage-1", id="outcome-1")
    assert engine.attach_usage("execution", manifest, id="usage-1") == attached
    assert len(engine.raw_rows) == 3
    assert engine.evidence("execution")["outcome"] == "loud"
    restored = Engine(directory=tmp_path)
    assert restored.evidence("execution")["execution_usage"] == collect(manifest)
    writer = restored.memory_learning_context("execution")["request"]
    assert writer["observation"]["execution_usage"] == collect(manifest)
    assert len(writer["outcome_history"]) == 3


def test_attach_failure_keeps_outcome_and_conflicting_id_never_appends(tmp_path):
    from howhard import Engine
    engine = Engine(directory=tmp_path)
    engine.observe("task", "fixture-model", "high", "clean", native=True, id="execution")
    manifest = artifact(tmp_path, codex_events())
    missing = {**manifest, "source": str(tmp_path / "missing.jsonl")}
    with pytest.raises(HowHardError, match="read failed"):
        engine.attach_usage("execution", missing, id="usage-1")
    assert len(Engine(directory=tmp_path).raw_rows) == 1
    engine.attach_usage("execution", manifest, id="usage-1")
    changed = artifact(tmp_path, codex_events(output_tokens=21))
    with pytest.raises(HowHardError, match="conflict"):
        engine.attach_usage("execution", changed, id="usage-1")
    assert len(engine.raw_rows) == 2


def test_wrong_owner_clear_allows_truthful_reattachment_and_preserves_raw_history(tmp_path):
    from howhard import Engine
    engine = Engine(directory=tmp_path)
    manifest = artifact(tmp_path, codex_events())
    usage = collect(manifest)
    engine.observe("task", "fixture-model", "high", "clean", native=True, id="execution", execution_usage=usage)
    engine.observe("task", "fixture-model", "high", "clean", amends="execution", id="clear",
                   measurement_corrections={"execution_usage": None})
    assert engine.evidence("execution")["execution_usage"] is None
    assert engine.evidence("execution", raw=True)["execution_usage"] == usage
    engine.observe("task", "fixture-model", "high", "clean", native=True, id="second")
    other = {**manifest, "execution_id": "second"}
    restored = Engine(directory=tmp_path)
    restored.attach_usage("second", other, id="second-usage")
    assert len(restored.raw_rows) == 4 and len(restored.rows) == 2
    assert restored.evidence("second")["execution_usage"]["execution_id"] == "second"
    assert restored.evidence("execution")["execution_usage"] is None
    assert restored.evidence("execution", raw=True)["execution_usage"] == usage


def test_partial_span_overlap_and_child_inclusion_are_not_double_counted(tmp_path):
    from howhard import Engine
    engine = Engine(directory=tmp_path)
    manifest = artifact(tmp_path, codex_events(), scope="inclusive", included_sessions=["child-session"])
    engine.observe("task", "fixture-model", "high", "clean", native=True, id="execution", execution_usage=collect(manifest))
    engine.observe("task", "fixture-model", "high", "clean", native=True, id="child")
    child_events = [{"type": "thread.started", "thread_id": "child-session"}, *codex_events()[1:]]
    child = artifact(tmp_path, child_events, execution_id="child", session_id="child-session")
    with pytest.raises(HowHardError, match="overlap|belongs|claim|source"):
        engine.attach_usage("child", child, id="child-usage")
    assert len(engine.rows) == 2 and len(engine.raw_rows) == 2


def test_usage_correction_can_replace_envelope_without_altering_other_measurements(tmp_path):
    from howhard import Engine
    engine = Engine(directory=tmp_path)
    manifest = artifact(tmp_path, codex_events())
    original = collect(manifest)
    engine.observe("task", "fixture-model", "high", "clean", native=True, id="execution",
                   execution_usage=original, seconds=7, tokens=120)
    replacement = collect(artifact(tmp_path, codex_events(output_tokens=21)))
    engine.observe("task", "fixture-model", "high", "clean", amends="execution", id="replacement",
                   measurement_corrections={"execution_usage": replacement})
    current = engine.evidence("execution")
    assert current["execution_usage"] == replacement and current["seconds"] == 7 and current["tokens"] == 121
    assert engine.evidence("execution", raw=True)["execution_usage"] == original


def test_usage_cli_collect_attach_and_original_envelope(tmp_path, monkeypatch, capsys):
    from howhard import Engine
    from howhard.cli import main
    monkeypatch.chdir(tmp_path)
    manifest = artifact(tmp_path, codex_events())
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert main(["usage", "collect", "--manifest", str(path)]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope == collect(manifest)
    assert not (tmp_path / ".howhard").exists()
    usage_path = tmp_path / "usage.json"
    usage_path.write_text(json.dumps(envelope))
    assert main(["observe", "task", "fixture-model", "high", "clean", "--native", "--id", "execution",
                 "--execution-usage", str(usage_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["observed"]["execution_usage"] == envelope
    # Attach already-known evidence is idempotent and does not create a second run.
    assert main(["usage", "attach", "execution", "--manifest", str(path), "--id", "usage-1", "--json"]) == 0
    capsys.readouterr()
    assert len(Engine(directory=tmp_path).raw_rows) == 1


def decide_categories(engine, candidates, *, token_usage=None, scalar=120):
    prepared = engine.prepare("task", "Fixture category pricing", native_candidates=candidates)
    usage = counts(input=60, output=20, cache_read=40) if token_usage is None else token_usage
    response = {"forecasts": [{"family": c["family"], "effort": c["effort"], "clean": .99,
                               "loud": .009, "silent": .001, "tokens": scalar, "seconds": 1,
                               "token_usage": usage} for c in prepared["baseline_request"]["cells"]],
                "memory_forecasts": [], "reason": "Synthetic fixture"}
    return engine.decide(prepared["request_id"], response, seed=7)


def test_category_forecast_pricing_and_explicit_cost_basis(tmp_path):
    from howhard import Engine
    engine = Engine(directory=tmp_path)
    decision = decide_categories(engine, [{"model": "fixture-model", "effort": "high", "rate_card": card()}])
    assert decision["cost_policy"]["worker_money_included"]
    assert decision["cells"][0]["C_run"] == pytest.approx((120 + 160 + 8) / 1e6)


@pytest.mark.parametrize("usage,scalar", [(counts(input=60, output=20, cache_read=40), 121),
    (counts(input=float("inf")), 120), (counts(input=-1), 120), (counts(input=None), 120)])
def test_invalid_category_forecasts_refused_before_decision(tmp_path, usage, scalar):
    from howhard import Engine
    with pytest.raises(HowHardError, match="token|forecast|nonnegative|finite"):
        decide_categories(Engine(directory=tmp_path), [{"model": "fixture-model", "effort": "high", "rate_card": card()}],
                          token_usage=usage, scalar=scalar)


def test_category_rate_card_requires_category_forecast(tmp_path):
    from howhard import Engine
    engine = Engine(directory=tmp_path)
    prepared = engine.prepare("task", native_candidates=[{"model": "fixture-model", "effort": "high", "rate_card": card()}])
    cell = prepared["baseline_request"]["cells"][0]
    response = {"forecasts": [{"family": cell["family"], "effort": cell["effort"], "clean": .99,
                               "loud": .009, "silent": .001, "tokens": 120, "seconds": 1}],
                "memory_forecasts": [], "reason": "Synthetic fixture"}
    with pytest.raises(HowHardError, match="token_usage|category"):
        engine.decide(prepared["request_id"], response, seed=7)


@pytest.mark.parametrize("second", [
    {"model": "other-model", "effort": "high", "usd_per_mtok": 1},
    {"model": "other-model", "effort": "high", "rate_card": card(model="other-model", basis="charge")},
    {"model": "other-model", "effort": "high"},
])
def test_unknown_or_mixed_money_basis_excludes_worker_money_uniformly(tmp_path, second):
    from howhard import Engine
    decision = decide_categories(Engine(directory=tmp_path), [
        {"model": "fixture-model", "effort": "high", "rate_card": card()}, second])
    assert not decision["cost_policy"]["worker_money_included"]
    # Equal probabilities/time give equal objectives despite different/unknown prices.
    assert decision["cells"][0]["J"] == pytest.approx(decision["cells"][1]["J"])


def test_candidate_refuses_blended_and_category_prices_together(tmp_path):
    from howhard import Engine
    with pytest.raises(HowHardError, match="exclusive"):
        Engine(directory=tmp_path).prepare("task", native_candidates=[{
            "model": "fixture-model", "effort": "high", "rate_card": card(), "usd_per_mtok": 1}])


def test_unsupported_host_schema_is_refused_instead_of_guessed(tmp_path):
    with pytest.raises(HowHardError, match="version|schema"):
        collect(artifact(tmp_path, codex_events(), host_version="codex-future-unknown"))


def test_empty_or_only_unsupported_events_leave_usage_unavailable(tmp_path):
    result = collect(artifact(tmp_path, [{"type": "future.unsupported", "usage": codex_usage()}]))
    assert result["status"] == "unavailable" and result["segments"] == []


def test_missing_token_category_keeps_complete_boundary_partial(tmp_path):
    result = collect(artifact(tmp_path, codex_events(cached_input_tokens=None)))
    assert result["status"] == "partial"
    assert result["segments"][0]["estimated_cost"] is None


def test_codex_reasoning_null_stays_unknown_without_crashing(tmp_path):
    result = collect(artifact(tmp_path, codex_events(reasoning_output_tokens=None)))
    assert result["segments"][0]["tokens"]["output"] == 20


def test_cross_coordinate_claim_cannot_double_count_same_session(tmp_path):
    from howhard import Engine
    engine = Engine(directory=tmp_path)
    manifest = artifact(tmp_path, codex_events(), coordinate_id="owned-coordinate")
    engine.observe("task", "fixture-model", "high", "clean", native=True, id="execution", execution_usage=collect(manifest))
    engine.observe("task", "fixture-model", "high", "clean", native=True, id="other")
    other = {**manifest, "execution_id": "other", "coordinate_id": "unrelated-coordinate"}
    with pytest.raises(HowHardError, match="coordinate|claim|source|overlap"):
        engine.attach_usage("other", other, id="other-usage")


def test_claude_exact_sidechain_binding_can_be_collected(tmp_path):
    event = claude_message()
    event.update(isSidechain=True, agentId="owned-agent")
    result = collect(artifact(tmp_path, [event], adapter="claude-session-v1", agent_id="owned-agent"))
    assert result["status"] == "complete"
    with pytest.raises(HowHardError, match="agent|scope|session"):
        collect(artifact(tmp_path, [event], adapter="claude-session-v1", agent_id="other-agent"))


def test_sdk_requires_result_uuid_and_refuses_multiple_cumulative_results(tmp_path):
    event = sdk_event()
    event.pop("uuid")
    with pytest.raises(HowHardError, match="uuid|identity|ID"):
        collect(sdk_manifest(tmp_path, event))
    manifest = sdk_manifest(tmp_path)
    path = tmp_path / "owned-events.jsonl"
    path.write_text(json.dumps(sdk_event()) + "\n" + json.dumps(sdk_event(uuid="result-2")) + "\n")
    manifest["end"] = path.stat().st_size
    with pytest.raises(HowHardError, match="one|result|cumulative"):
        collect(manifest)


def test_rate_card_calculation_tampering_and_unauthoritative_bill_refused(tmp_path):
    usage = collect(artifact(tmp_path, codex_events(), rate_card=card()))
    usage["segments"][0]["estimated_cost"]["usd"] += 1
    with pytest.raises(HowHardError, match="amount|price|rate|cost"):
        validate(usage)
    usage = collect(artifact(tmp_path, codex_events()))
    usage["segments"][0]["billed_cost"] = {"usd": 1, "source": "client", "version": "fixture"}
    with pytest.raises(HowHardError, match="authoritative|receipt|execution|billed|missing"):
        validate(usage)


def test_known_codex_aggregate_survives_unknown_cache_partition_but_not_incomplete_boundary(tmp_path):
    from howhard import Engine
    engine = Engine(directory=tmp_path)
    manifest = artifact(tmp_path, codex_events(cached_input_tokens=None))
    usage = collect(manifest)
    assert usage["status"] == "partial" and usage["segments"][0]["reported_tokens"] == 120
    engine.observe("task", "fixture-model", "high", "clean", native=True, id="execution", execution_usage=usage)
    assert engine.evidence("execution")["tokens"] == 120
    incomplete_events = [{"type": "thread.started", "thread_id": "other-session"}, *codex_events(cached_input_tokens=None)[1:]]
    other = artifact(tmp_path, incomplete_events, completed=False, execution_id="incomplete", session_id="other-session")
    engine.observe("task", "fixture-model", "high", "clean", native=True, id="incomplete", execution_usage=collect(other))
    assert engine.evidence("incomplete")["tokens"] is None


def test_codex_session_null_cumulative_counts_remain_partial_without_comparison_crash(tmp_path):
    first = token_event(100)
    second = token_event(None, timestamp="2026-11-07T01:00:01Z")
    events = [first, second, {"type": "event_msg", "payload": {"type": "task_complete"}}]
    result = collect(artifact(tmp_path, events, adapter="codex-session-v1"))
    assert result["status"] == "partial"


def test_report_separates_estimate_bill_legacy_and_unknown_coverage(tmp_path):
    from howhard import Engine
    from howhard.report import report
    engine = Engine(directory=tmp_path)
    usage = collect(artifact(tmp_path, codex_events(), rate_card=card()))
    usage["segments"][0]["billed_cost"] = {"usd": .03, "source": "fixture://invoice", "version": "fixture-v1",
        "authoritative": True, "execution_id": "execution", "receipt_id": "fixture-receipt-1"}
    engine.observe("task", "fixture-model", "high", "clean", native=True, id="execution", execution_usage=usage, cost_usd=.5)
    engine.observe("task", "fixture-model", "high", "clean", native=True, id="unknown")
    stats = report(engine)
    accounting = stats["execution_accounting"]
    assert stats["runs"] == accounting["runs"] == 2
    assert accounting["statuses"] == {"complete": 1, "partial": 0, "unavailable": 0, "missing": 1}
    assert accounting["tokens"]["input"] == {"known": 60, "unknown_segments": 0, "unknown_runs": 1, "total": None}
    assert accounting["estimated_cost"]["known_usd"] == pytest.approx(.000288)
    assert accounting["billed_cost"]["known_usd"] == .03
    assert accounting["estimated_cost"]["unknown_runs"] == accounting["billed_cost"]["unknown_runs"] == 1
    assert accounting["estimated_cost"]["total_usd"] is None and accounting["billed_cost"]["total_usd"] is None
    assert stats["worker_cost"] == {"basis": "legacy-unspecified", "known_usd": .5, "unknown_runs": 1, "total_usd": None}


def test_billed_receipt_must_identify_same_execution(tmp_path):
    usage = collect(artifact(tmp_path, codex_events()))
    usage["segments"][0]["billed_cost"] = {"usd": .03, "source": "fixture://invoice", "version": "fixture-v1",
        "authoritative": True, "execution_id": "another-execution", "receipt_id": "fixture-receipt-1"}
    with pytest.raises(HowHardError, match="execution"):
        validate(usage)


def test_nonregular_source_is_refused_before_opening_fifo(tmp_path):
    import os
    manifest = artifact(tmp_path, codex_events())
    path = tmp_path / "owned-events.jsonl"
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(HowHardError, match="regular|source|file"):
        collect(manifest)


def test_codex_resumed_session_requires_starting_cumulative_baseline(tmp_path):
    baseline = codex_usage()
    event = token_event(200)
    event["payload"]["info"]["total_token_usage"] = codex_usage(input_tokens=200, cached_input_tokens=80, output_tokens=40,
                                                               reasoning_output_tokens=16)
    events = [{"type": "session_meta", "payload": {"id": "session"}}, event,
              {"type": "event_msg", "payload": {"type": "task_complete"}}]
    manifest = artifact(tmp_path, events, adapter="codex-session-v1")
    path = tmp_path / "owned-events.jsonl"
    prefix = b'{"type":"already-owned-history"}\n'
    path.write_bytes(prefix + path.read_bytes())
    manifest.update(start=len(prefix), end=path.stat().st_size)
    assert collect(manifest)["status"] == "partial"
    measured = collect({**manifest, "baseline": baseline})
    assert measured["status"] == "complete"
    assert measured["segments"][0]["tokens"] == counts(input=60, output=20, cache_read=40)
    assert len(measured["provenance"]["baseline_sha256"]) == 64


def test_codex_missing_usage_frame_stays_partial_despite_terminal_event(tmp_path):
    first, last = token_event(100), token_event(300, timestamp="2026-11-07T01:00:02Z")
    # The cumulative delta contains an omitted turn in addition to last_token_usage.
    last["payload"]["info"]["total_token_usage"] = codex_usage(input_tokens=300, cached_input_tokens=120, output_tokens=60)
    result = collect(artifact(tmp_path, [first, last, {"type": "event_msg", "payload": {"type": "task_complete"}}],
                              adapter="codex-session-v1"))
    assert result["status"] == "partial" and not result["provenance"]["completed"]


def test_multimodel_sdk_card_mismatch_can_retry_without_prices(tmp_path):
    event = sdk_event(modelUsage={"fixture-model": sdk_model(), "other-model": sdk_model(costUSD=.02)})
    manifest = sdk_manifest(tmp_path, event, rate_card=card())
    with pytest.raises(HowHardError, match="mismatch"):
        collect(manifest)
    manifest.pop("rate_card")
    result = collect(manifest)
    assert result["status"] == "complete" and len(result["segments"]) == 2
    assert all(s["billed_cost"] is None for s in result["segments"])


def test_authoritative_invoice_line_cannot_be_reused_between_executions(tmp_path):
    from howhard import Engine
    engine = Engine(directory=tmp_path)
    usage = collect(artifact(tmp_path, codex_events()))
    receipt = {"usd": .03, "source": "fixture://invoice", "version": "fixture-v1", "authoritative": True,
               "execution_id": "execution", "receipt_id": "fixture-line-1"}
    usage["segments"][0]["billed_cost"] = receipt
    engine.observe("task", "fixture-model", "high", "clean", native=True, id="execution", execution_usage=usage)
    other_events = [{"type": "thread.started", "thread_id": "other-session"}, *codex_events()[1:]]
    other = collect(artifact(tmp_path, other_events, session_id="other-session", execution_id="other"))
    other["segments"][0]["billed_cost"] = {**receipt, "execution_id": "other"}
    with pytest.raises(HowHardError, match="receipt|authoritative|billing"):
        engine.observe("task", "fixture-model", "high", "clean", native=True, id="other", execution_usage=other)
    assert len(engine.rows) == len(engine.raw_rows) == 1


@pytest.mark.parametrize("subtype", ["error_max_turns", "error_max_budget_usd"])
def test_sdk_bounded_failure_has_complete_usage_and_estimate_independent_of_outcome(tmp_path, subtype):
    from howhard import Engine
    manifest = sdk_manifest(tmp_path, sdk_event(subtype=subtype, is_error=True))
    usage = collect(manifest)
    assert usage["status"] == "complete" and usage["provenance"]["completed"] is True
    assert usage["segments"][0]["estimated_cost"]["usd"] == .12
    assert usage["segments"][0]["billed_cost"] is None
    engine = Engine(directory=tmp_path)
    engine.observe("task", "fixture-model", "high", "loud", native=True, id="execution", execution_usage=usage)
    effective = engine.evidence("execution")
    assert effective["outcome"] == "loud" and effective["tokens"] == 190
    assert effective["execution_usage"]["status"] == "complete"


@pytest.mark.parametrize("subtype,is_error", [("success", True), ("error_max_turns", False),
    ("error_max_budget_usd", False), ("success", "false"), ("error_max_turns", 1)])
def test_sdk_contradictory_or_untyped_terminal_error_flag_refused(tmp_path, subtype, is_error):
    with pytest.raises(HowHardError, match="contradictory|is_error|boolean|failure"):
        collect(sdk_manifest(tmp_path, sdk_event(subtype=subtype, is_error=is_error)))


def test_sdk_uncertain_crash_result_stays_partial_without_complete_estimate(tmp_path):
    usage = collect(sdk_manifest(tmp_path, sdk_event(subtype="error_during_execution", is_error=True)))
    assert usage["status"] == "partial" and usage["provenance"]["completed"] is False
    assert usage["segments"][0]["estimated_cost"] is None
    assert usage["segments"][0]["billed_cost"] is None


def test_sdk_completed_boundary_preserves_client_estimate_with_unknown_token_category(tmp_path):
    from howhard import Engine
    from howhard.report import report
    event = sdk_event(modelUsage={"fixture-model": sdk_model(cacheReadInputTokens=None)})
    manifest = sdk_manifest(tmp_path, event, rate_card=card())
    usage = collect(manifest)
    assert usage["status"] == "partial" and usage["provenance"]["completed"] is True
    segment = usage["segments"][0]
    assert segment["tokens"]["cache_read"] is None and segment["tokens"]["input"] == 100
    assert segment["estimated_cost"] == {"usd": .12, "source": "claude-client", "version": "2.1.318"}
    assert segment["billed_cost"] is None
    engine = Engine(directory=tmp_path)
    engine.observe("task", "fixture-model", "high", "clean", native=True, id="execution", execution_usage=usage)
    accounting = report(engine)["execution_accounting"]
    assert accounting["estimated_cost"] == {"known_usd": .12, "unknown_runs": 0, "total_usd": .12}
    assert accounting["tokens"]["cache_read"] == {"known": 0, "unknown_segments": 1, "unknown_runs": 1, "total": None}
    assert accounting["tokens"]["input"] == {"known": 100, "unknown_segments": 0, "unknown_runs": 0, "total": 100}
    assert engine.evidence("execution")["tokens"] is None


@pytest.mark.parametrize("completed", [False, True])
def test_authoritative_billing_independent_of_partial_tokens_and_boundary(tmp_path, completed):
    from howhard import Engine
    from howhard.report import report
    usage = collect(artifact(tmp_path, codex_events(cached_input_tokens=None), completed=completed))
    assert usage["status"] == "partial"
    usage["segments"][0]["billed_cost"] = {"usd": .03, "source": "fixture://invoice", "version": "fixture-v1",
        "authoritative": True, "execution_id": "execution", "receipt_id": "fixture-line-1"}
    assert validate(usage) == usage
    engine = Engine(directory=tmp_path)
    engine.observe("task", "fixture-model", "high", "loud", native=True, id="execution", execution_usage=usage)
    accounting = report(engine)["execution_accounting"]
    assert accounting["statuses"]["partial"] == 1
    assert accounting["billed_cost"] == {"known_usd": .03, "unknown_runs": 0, "total_usd": .03}
    assert accounting["estimated_cost"] == {"known_usd": 0, "unknown_runs": 1, "total_usd": None}
    assert engine.evidence("execution")["outcome"] == "loud"


def test_envelope_primary_claim_and_completed_boundary_must_match_provenance(tmp_path):
    usage = collect(artifact(tmp_path, codex_events()))
    wrong_claim = deepcopy(usage)
    wrong_claim["claims"][0]["coordinate_id"] = "other-coordinate"
    with pytest.raises(HowHardError, match="claim|provenance"):
        validate(wrong_claim)
    incomplete = deepcopy(usage)
    incomplete["provenance"]["completed"] = False
    with pytest.raises(HowHardError, match="complete|boundary"):
        validate(incomplete)
