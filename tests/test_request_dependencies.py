"""Concurrent request validity at public boundaries (issue #69)."""

from copy import deepcopy
from contextlib import ExitStack
import json
import subprocess

import pytest

from howhard import Engine
from howhard.decisions import load
from howhard.validation import HowHardError
from test_capability_filter import filtering, forecasts, inventory, ProtocolDouble
from test_write_queue import client, receive


class RelevantOnly:
    """Select only explicit rollback fixtures; never load model weights."""
    @property
    def reranker(self):
        return self

    def score(self, query, documents):
        return [8.] * len(documents)

    def search(self, records, query, limit=8, **options):
        selected = [r for r in records if 'rollback' in r['text'].lower()][:limit]
        return {'records': deepcopy(selected), 'trace': {'seconds': 0.,
                'selected': [{'key': r['key'], 'score': 8.} for r in selected]}}


def engine_at(directory):
    class WriterDouble(ProtocolDouble):
        def complete(self, request):
            if request['kind'] == 'learn':
                return {'operations': [{'op': 'add', 'title': 'Unrelated lesson',
                        'body': 'Synthetic unrelated writer output'}], 'reason': 'Synthetic writer'}
            return super().complete(request)
    return Engine(directory=directory, retrieval=RelevantOnly(), llm=WriterDouble())


def prepare(engine):
    return engine.prepare('task', 'Repair rollback', native_candidates=inventory(2))


def finish(engine, pending):
    while 'decision_id' not in pending:
        response = (filtering(pending['baseline_request'], 0, 1)
                    if pending['stage'] == 'filter' else forecasts(pending['baseline_request']))
        pending = engine.decide(pending['request_id'], response, seed=42)
    return pending


def _write(directory, operation, options, connection):
    engine = engine_at(directory)
    if operation == 'unrelated':
        engine.observe('other', 'fixture/other', 'high', 'clean', native=True,
                       task_text='Synthetic unrelated task', id=options['id'])
        engine.memory.add('Unrelated lesson', 'Synthetic unrelated memory')
    elif operation == 'edit':
        engine.memory.edit(options['id'], body='Changed rollback guidance')
    elif operation == 'amend':
        engine.observe('task', 'fixture/model-000', 'high', 'silent',
                       amends=options['id'], id='synthetic-correction')
    elif operation == 'score':
        engine.observe_decision(options['decision_id'], 'silent', id='synthetic-scored-run')
    connection.send('written')


def write(directory, operation, **options):
    with client(_write, directory, operation, options) as (process, connection):
        assert receive(connection) == 'written'
        process.join(10)
        assert process.exitcode == 0


def test_unrelated_process_writes_do_not_invalidate_either_stage(tmp_path):
    engine = engine_at(tmp_path)
    pending = prepare(engine)
    write(tmp_path, 'unrelated', id='synthetic-first')
    pending = engine.decide(pending['request_id'], filtering(pending['baseline_request'], 0, 1))
    write(tmp_path, 'unrelated', id='synthetic-second')
    result = finish(engine, pending)
    receipt = load(engine.config, result['decision_id'])
    assert receipt['request_snapshot']['weights_policy'] == 'snapshot'
    assert receipt['request_snapshot']['dependencies']['evidence'] == {}
    assert len(engine_at(tmp_path).rows) == 2


def test_surfaced_memory_change_names_versions_and_returns_refresh(tmp_path):
    engine = engine_at(tmp_path)
    memory = engine.memory.add('Rollback lesson', 'Original rollback guidance')
    pending = prepare(engine)
    write(tmp_path, 'edit', id=memory['id'])
    with pytest.raises(HowHardError) as caught:
        engine.decide(pending['request_id'], filtering(pending['baseline_request'], 0, 1))
    assert type(caught.value).__name__ == 'StaleRequest'
    error = caught.value.as_dict()
    assert error['code'] == 'stale_request'
    change = next(c for c in error['changes'] if c['id'] == memory['id'])
    assert (change['old_version'], change['new_version']) == (1, 2)
    refreshed = error['refreshed_request']
    assert refreshed['request_id'] != pending['request_id']
    assert refreshed['baseline_request'] == pending['baseline_request']
    assert error['baseline_unchanged']
    assert finish(engine, refreshed)['decision_id']


def test_surfaced_evidence_amendment_refuses_only_that_dependency(tmp_path):
    engine = engine_at(tmp_path)
    engine.observe('task', 'fixture/model-000', 'high', 'clean', native=True,
                   task_text='Synthetic rollback precedent', id='synthetic-precedent')
    pending = prepare(engine)
    write(tmp_path, 'amend', id='synthetic-precedent')
    with pytest.raises(HowHardError) as caught:
        engine.decide(pending['request_id'], filtering(pending['baseline_request'], 0, 1))
    assert type(caught.value).__name__ == 'StaleRequest'
    change = next(c for c in caught.value.as_dict()['changes']
                  if c['kind'] == 'evidence' and c['id'] == 'synthetic-precedent')
    assert change['old_digest'] != change['new_digest']
    assert caught.value.as_dict()['refreshed_request']['stage'] == 'filter'


def test_expert_weights_are_frozen_when_a_concurrent_run_is_scored(tmp_path):
    engine = engine_at(tmp_path)
    memory = engine.memory.add('Rollback lesson', 'Synthetic rollback forecasts')
    warmup = prepare(engine)
    f = filtering(warmup['baseline_request'], 0, 1)
    f['memory_filters'] = [{'id': memory['id'], 'version': 1, 'filter': deepcopy(f['filter'])}]
    band = engine.decide(warmup['request_id'], f)
    response = forecasts(band['baseline_request'])
    response['memory_forecasts'] = [{'id': memory['id'], 'version': 1,
        'forecasts': [{**c, 'clean': .5, 'loud': .2, 'silent': .3} for c in response['forecasts']]}]
    previous = engine.decide(band['request_id'], response, seed=42)
    pending = prepare(engine)
    original = deepcopy(pending['request']['beliefs'])
    write(tmp_path, 'score', decision_id=previous['decision_id'])
    f = filtering(pending['baseline_request'], 0, 1)
    f['memory_filters'] = [{'id': memory['id'], 'version': 1, 'filter': deepcopy(f['filter'])}]
    band = engine.decide(pending['request_id'], f)
    response = forecasts(band['baseline_request'])
    response['memory_forecasts'] = [{'id': memory['id'], 'version': 1,
                                    'forecasts': deepcopy(response['forecasts'])}]
    result = engine.decide(band['request_id'], response, seed=42)
    old = {e['key'].removeprefix('filter:'): e['log_score'] for e in original['experts']}
    frozen = result['semantic']['beliefs']['experts']
    assert all(e['log_score'] == old[e['key']] for e in frozen)
    current = {e['key']: e['log_score'] for e in engine_at(tmp_path).memory_beliefs()['experts']}
    assert current[f"memory:{memory['id']}:1"] != old[f"memory:{memory['id']}:1"]


def _prepare_slowly(directory, connection):
    class SlowRetrieval(RelevantOnly):
        def search(self, *args, **kwargs):
            connection.send('retrieving')
            connection.recv()
            return super().search(*args, **kwargs)
    engine = engine_at(directory)
    engine.retrieval = SlowRetrieval()
    try:
        connection.send({'prepared': prepare(engine)})
    except Exception as exc:
        connection.send({'error': str(exc)})


def test_prepare_publishes_a_consistent_snapshot_despite_intervening_writes(tmp_path):
    engine = engine_at(tmp_path)
    memory = engine.memory.add('Rollback lesson', 'Original rollback guidance')
    with client(_prepare_slowly, tmp_path) as (process, connection):
        assert receive(connection) == 'retrieving'
        engine.memory.edit(memory['id'], body='Changed rollback guidance')
        connection.send('continue')
        result = receive(connection)
        assert 'prepared' in result, result
        pending = result['prepared']
        assert pending['request']['memory_revision'] == 1
        assert pending['request']['memories'][0]['version'] == 1
        process.join(10)
        assert process.exitcode == 0


def _flow(directory, connection):
    engine = engine_at(directory)
    pending = prepare(engine)
    while True:
        connection.send(pending)
        if 'decision_id' in pending:
            return
        connection.recv()
        response = (filtering(pending['baseline_request'], 0, 1)
                    if pending['stage'] == 'filter' else forecasts(pending['baseline_request']))
        pending = engine.decide(pending['request_id'], response, seed=42)


def test_two_flows_complete_with_third_process_observations_and_learning(tmp_path):
    with ExitStack() as stack:
        clients = [stack.enter_context(client(_flow, tmp_path)) for _ in range(2)]
        pending = [receive(connection) for _, connection in clients]
        for stage in ('filter', 'forecast'):
            assert all(p['stage'] == stage for p in pending)
            write(tmp_path, 'unrelated', id='synthetic-interleaved-' + stage)
            for _, connection in clients:
                connection.send('advance')
            pending = [receive(connection) for _, connection in clients]
        assert all(p['decision_id'] for p in pending)
        for process, _ in clients:
            process.join(10)
            assert process.exitcode == 0
    engine = engine_at(tmp_path)
    assert len(engine.rows) == 2
    assert len(engine.memory.list()) == 4  # Explicit writes plus the actual fixture writer steps.


def test_widening_preserves_snapshot_across_unrelated_writes(tmp_path):
    engine = engine_at(tmp_path)
    pending = engine.prepare('task', 'Repair rollback', native_candidates=inventory(4))
    band = engine.decide(pending['request_id'], filtering(pending['baseline_request'], 0, 0))
    write(tmp_path, 'unrelated', id='synthetic-before-widening')
    wider = engine.decide(band['request_id'], forecasts(band['baseline_request'], silent=.8))
    assert wider['stage'] == 'forecast' and len(wider['capability']['widening']) == 1
    write(tmp_path, 'unrelated', id='synthetic-after-widening')
    result = engine.decide(wider['request_id'], forecasts(wider['baseline_request']), seed=42)
    assert result['cap_met'] and len(result['cells']) == 2
    assert result['request_snapshot']['dependencies']['evidence'] == {}


def test_unsurfaced_memory_edit_does_not_change_an_inflight_request(tmp_path):
    engine = engine_at(tmp_path)
    hidden = engine.memory.add('Unrelated', 'Not selected')
    pending = prepare(engine)
    write(tmp_path, 'edit', id=hidden['id'])
    result = finish(engine, pending)
    assert result['request_snapshot']['dependencies']['memories'] == {}


def test_additional_retrieval_uses_the_snapshot_and_tracks_new_dependencies(tmp_path):
    engine = engine_at(tmp_path)
    hidden = engine.memory.add('Unrelated', 'Original content')
    pending = prepare(engine)
    looked_up = engine.retrieve(pending['request_id'], [{'tool': 'memory.read', 'id': hidden['id']}])
    write(tmp_path, 'edit', id=hidden['id'])
    with pytest.raises(HowHardError) as caught:
        finish(engine, looked_up)
    assert caught.value.as_dict()['changes'][0]['id'] == hidden['id']


def test_retrieval_cannot_silently_read_newer_unsurfaced_content(tmp_path):
    engine = engine_at(tmp_path)
    hidden = engine.memory.add('Unrelated', 'Original content')
    pending = prepare(engine)
    write(tmp_path, 'edit', id=hidden['id'])
    with pytest.raises(HowHardError) as caught:
        engine.retrieve(pending['request_id'], [{'tool': 'memory.read', 'id': hidden['id']}])
    error = caught.value.as_dict()
    assert error['request_id'] == pending['request_id']
    assert error['changes'][0]['old_version'] == 1
    assert error['changes'][0]['new_version'] == 2


def test_dependency_is_checked_again_after_selection_before_receipt_commit(tmp_path, monkeypatch):
    import howhard.engine as module
    engine = engine_at(tmp_path)
    memory = engine.memory.add('Rollback lesson', 'Original guidance')
    pending = prepare(engine)
    band = engine.decide(pending['request_id'], filtering(pending['baseline_request'], 0, 1))
    choose = module.choose
    def concurrent_change(*args, **kwargs):
        result = choose(*args, **kwargs)
        write(tmp_path, 'edit', id=memory['id'])
        return result
    monkeypatch.setattr(module, 'choose', concurrent_change)
    with pytest.raises(HowHardError) as caught:
        engine.decide(band['request_id'], forecasts(band['baseline_request']), seed=42)
    assert caught.value.as_dict()['request_id'] == band['request_id']
    assert caught.value.as_dict()['refreshed_request']['stage'] == 'filter'
    assert not (engine.config.state_root / 'decisions').exists()


def test_cli_returns_structured_changes_and_refreshed_request(tmp_path, monkeypatch, capsys):
    from howhard.cli import main
    engine = engine_at(tmp_path)
    memory = engine.memory.add('Rollback lesson', 'Original rollback guidance')
    pending = prepare(engine)
    write(tmp_path, 'edit', id=memory['id'])
    response = tmp_path / 'response.json'
    response.write_text(json.dumps(filtering(pending['baseline_request'], 0, 1)))
    monkeypatch.chdir(tmp_path)
    assert main(['decide', pending['request_id'], '--file', str(response), '--json']) == 2
    output = capsys.readouterr()
    assert not output.out
    error = json.loads(output.err)['error']
    assert error['code'] == 'stale_request' and error['refreshed_request']['request_id']
    assert error['changes'][0]['id'] == memory['id']


def test_shared_worktree_writes_allow_owner_request_but_preserve_binding(tmp_path):
    project, sibling = tmp_path / 'project', tmp_path / 'sibling'
    project.mkdir()
    def git(*args):
        subprocess.run(['git', '-C', str(project), *args], check=True, capture_output=True)
    git('init', '-q')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '--allow-empty', '-m', 'base')
    git('worktree', 'add', '-b', 'sibling', str(sibling))
    engine = engine_at(project)
    pending = prepare(engine)
    write(sibling, 'unrelated', id='synthetic-sibling')
    with pytest.raises(HowHardError, match='another worktree'):
        finish(engine_at(sibling), pending)
    assert finish(engine, pending)['decision_id']


def test_receipt_rejects_invalid_snapshot_policy(tmp_path):
    engine = engine_at(tmp_path)
    result = finish(engine, prepare(engine))
    receipt = load(engine.config, result['decision_id'])
    receipt['request_snapshot']['weights_policy'] = 'live'
    with pytest.raises(HowHardError, match='weights policy'):
        load(engine.config, receipt)


def test_capability_recency_weights_do_not_move_with_unsurfaced_appends(tmp_path):
    engine = engine_at(tmp_path)
    engine.observe('task', 'fixture/model-000', 'high', 'clean', native=True,
                   task_text='Synthetic rollback precedent', id='prior-precedent')
    pending = prepare(engine)
    response = filtering(pending['baseline_request'], 0, 1)
    original = engine.decide(pending['request_id'], response)['capability']
    write(tmp_path, 'unrelated', id='synthetic-later')
    frozen = engine.decide(pending['request_id'], response)['capability']
    assert frozen['evidence'] == original['evidence']
    fresh = prepare(engine)
    updated = engine.decide(fresh['request_id'], filtering(fresh['baseline_request'], 0, 1))['capability']
    assert updated['evidence'][0]['weight'] != original['evidence'][0]['weight']


def test_legacy_v5_requests_keep_conservative_validation(tmp_path):
    from howhard.context import canonical, digest
    engine = engine_at(tmp_path)
    pending = prepare(engine)
    directory = engine.config.state_root / 'requests'
    bundle = json.loads((directory / (pending['request_id'] + '.json')).read_bytes())
    for name in ('read_state', 'dependencies', 'learned_state', 'inventory_source'):
        bundle.pop(name)
    request_id = digest(bundle)
    (directory / (request_id + '.json')).write_bytes(canonical(bundle))
    write(tmp_path, 'unrelated', id='synthetic-legacy-change')
    with pytest.raises(HowHardError, match='stale evidence or memory') as caught:
        engine.decide(request_id, filtering(pending['baseline_request'], 0, 1))
    assert type(caught.value).__name__ == 'HowHardError'


def test_existing_v6_receipts_without_snapshot_metadata_remain_readable(tmp_path):
    engine = engine_at(tmp_path)
    result = finish(engine, prepare(engine))
    receipt = load(engine.config, result['decision_id'])
    receipt.pop('request_snapshot')
    assert load(engine.config, receipt)['chosen'] == result['chosen']
