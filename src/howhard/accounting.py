"""Versioned, offline execution usage collection from explicitly owned artifacts."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
import stat
from pathlib import Path

from .validation import HowHardError, keys, number, text

CATEGORIES = ('input', 'output', 'cache_read', 'cache_write', 'cache_write_5m', 'cache_write_1h')
USAGE_FORMAT = 'howhard-execution-usage-v1'
RATE_FORMAT = 'howhard-rate-card-v1'
ADAPTERS = ('codex-exec-v1', 'codex-session-v1', 'claude-sdk-v1', 'claude-session-v1')
SUPPORTED_HOSTS = {'codex-exec-v1': ('0.166.0',), 'codex-session-v1': ('0.166.0',), 'claude-sdk-v1': ('2.1.318', '0.3.239'), 'claude-session-v1': ('2.1.318',)}
MAX_SOURCE = 32 * 1024 * 1024
MAX_SPAN = 16 * 1024 * 1024


def tokens(value, *, forecast=False):
    keys(value, CATEGORIES, where='token_usage')
    for key, count in value.items():
        if count is None and not forecast:
            continue
        if type(count) is int and count > 2**63 - 1:
            raise HowHardError('token_usage: token count exceeds supported integer range')
        number(count, 'token_usage.' + key, nonnegative=True)
        if not forecast and type(count) is not int:
            raise HowHardError('token_usage: measured tokens must be integers or null')
    return value


def total(value):
    return sum(value.values()) if all(v is not None for v in value.values()) else None


def rate_card(value):
    keys(value, ['format', 'source', 'version', 'effective_at', 'provider', 'model', 'service_tier', 'basis', 'rates'], where='rate_card')
    if value['format'] != RATE_FORMAT or value['basis'] not in ('estimate', 'charge'):
        raise HowHardError('rate_card: unsupported format or basis')
    for key in ('source', 'version', 'provider', 'model'):
        text(value[key], 'rate_card.' + key)
    if value['service_tier'] is not None:
        text(value['service_tier'], 'rate_card.service_tier')
    try:
        if datetime.fromisoformat(value['effective_at']).utcoffset() is None:
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise HowHardError('rate_card.effective_at: expected timezone-aware ISO timestamp') from exc
    keys(value['rates'], CATEGORIES, where='rate_card.rates')
    for key, amount in value['rates'].items():
        if amount is not None:
            number(amount, 'rate_card.rates.' + key, nonnegative=True)
    return value


def price(counts, card):
    rate_card(card)
    if any(counts[k] is None or (counts[k] != 0 and card['rates'][k] is None) for k in CATEGORIES):
        return None
    return sum(counts[k] * (card['rates'][k] or 0) for k in CATEGORIES) / 1e6


def money(value, *, billed=False):
    if value is None:
        return
    keys(value, ['usd', 'source', 'version'], ['rate_card', 'authoritative', 'execution_id', 'receipt_id'], where='usage money')
    number(value['usd'], 'usage money.usd', nonnegative=True)
    text(value['source'], 'usage money.source')
    text(value['version'], 'usage money.version')
    if billed and (value.get('authoritative') is not True or not value.get('receipt_id') or not value.get('execution_id')):
        raise HowHardError('billed_cost: explicit authoritative execution receipt required')
    if not billed and any(k in value for k in ('authoritative', 'execution_id', 'receipt_id')):
        raise HowHardError('estimated_cost: billed receipt fields are not estimates')
    if billed:
        for k in ('receipt_id', 'execution_id'):
            text(value[k], 'billed_cost.' + k)
    if 'rate_card' in value:
        if billed:
            raise HowHardError('billed_cost: a rate card is an estimate, not a billed receipt')
        rate_card(value['rate_card'])


def validate(value):
    keys(value, ['format', 'execution_id', 'status', 'provenance', 'claims', 'segments'], where='execution_usage')
    if value['format'] != USAGE_FORMAT or value['status'] not in ('complete', 'partial', 'unavailable'):
        raise HowHardError('execution_usage: unsupported format or status')
    text(value['execution_id'], 'execution_usage.execution_id')
    p = value['provenance']
    keys(p, ['adapter', 'host_version', 'artifact_sha256', 'session_id', 'start', 'end', 'scope', 'coordinate_id'], ['baseline_sha256', 'agent_id', 'completed'], where='usage provenance')
    if p['adapter'] not in ADAPTERS or p['scope'] not in ('self', 'inclusive'):
        raise HowHardError('usage provenance: unsupported adapter or scope')
    if p['host_version'] not in SUPPORTED_HOSTS[p['adapter']]:
        raise HowHardError('usage provenance.host_version: unsupported host/schema combination')
    if 'agent_id' in p:
        text(p['agent_id'], 'usage provenance.agent_id')
    if 'baseline_sha256' in p and (not isinstance(p['baseline_sha256'], str) or len(p['baseline_sha256']) != 64 or any(c not in '0123456789abcdef' for c in p['baseline_sha256'])):
        raise HowHardError('usage provenance.baseline_sha256: invalid SHA-256')
    if 'completed' in p and type(p['completed']) is not bool:
        raise HowHardError('usage provenance.completed: expected explicit boolean')
    if value['status'] == 'complete' and p.get('completed') is not True:
        raise HowHardError('execution_usage: complete requires a completed source boundary')
    for key in ('host_version', 'session_id', 'coordinate_id'):
        text(p[key], 'usage provenance.' + key)
    if not isinstance(p['artifact_sha256'], str) or len(p['artifact_sha256']) != 64 or any(c not in '0123456789abcdef' for c in p['artifact_sha256']):
        raise HowHardError('usage provenance: invalid artifact hash')
    if any(type(p[k]) is not int or p[k] < 0 for k in ('start', 'end')) or p['end'] < p['start']:
        raise HowHardError('usage provenance: invalid byte span')
    if not isinstance(value['claims'], list) or not value['claims']:
        raise HowHardError('execution_usage.claims: explicit source claims required')
    for c in value['claims']:
        keys(c, ['session_id', 'start', 'end', 'coordinate_id'], where='usage claim')
        text(c['session_id'], 'usage claim.session_id')
        text(c['coordinate_id'], 'usage claim.coordinate_id')
        if type(c['start']) is not int or c['start'] < 0 or (c['end'] is not None and (type(c['end']) is not int or c['end'] <= c['start'])):
            raise HowHardError('usage claim: invalid exclusive span')
    expected_session = p['session_id'] + ':agent:' + p['agent_id'] if 'agent_id' in p else p['session_id']
    expected = dict(session_id=expected_session, start=p['start'], end=p['end'], coordinate_id=p['coordinate_id'])
    if value['claims'][0] != expected:
        raise HowHardError('execution_usage.claims: primary claim must match source provenance')
    if p['scope'] == 'self' and len(value['claims']) != 1:
        raise HowHardError('execution_usage.claims: self scope cannot claim child sessions')
    for claim in value['claims'][1:]:
        if claim['start'] != 0 or claim['end'] is not None or claim['coordinate_id'] != '*':
            raise HowHardError('execution_usage.claims: included child requires a whole-session claim')
    if not isinstance(value['segments'], list):
        raise HowHardError('execution_usage.segments: expected array')
    if value['status'] == 'complete' and not value['segments']:
        raise HowHardError('execution_usage: complete requires measured segments')
    seen, receipts = set(), set()
    for segment in value['segments']:
        keys(segment, ['source_id', 'provider', 'model', 'service_tier', 'tokens', 'estimated_cost', 'billed_cost'], ['reported_tokens'], where='usage segment')
        for key in ('source_id', 'provider', 'model'):
            text(segment[key], 'usage segment.' + key)
        if segment['source_id'] in seen:
            raise HowHardError('usage segment: duplicate source identity')
        seen.add(segment['source_id'])
        if segment['service_tier'] is not None:
            text(segment['service_tier'], 'usage segment.service_tier')
        tokens(segment['tokens'])
        if segment.get('reported_tokens') is not None and (type(segment['reported_tokens']) is not int or segment['reported_tokens'] < 0):
            raise HowHardError('usage segment.reported_tokens: expected nonnegative integer or null')
        if segment.get('reported_tokens') is not None and total(segment['tokens']) is not None and segment['reported_tokens'] != total(segment['tokens']):
            raise HowHardError('usage segment.reported_tokens: does not match token categories')
        if value['status'] == 'complete' and total(segment['tokens']) is None:
            raise HowHardError('execution_usage: complete requires every measured category')
        money(segment['estimated_cost'])
        money(segment['billed_cost'], billed=True)
        if segment['billed_cost'] is not None:
            receipt = segment['billed_cost']['source'], segment['billed_cost']['receipt_id']
            if receipt in receipts:
                raise HowHardError('billed_cost: duplicate authoritative receipt line identity')
            receipts.add(receipt)
        if segment['billed_cost'] is not None and segment['billed_cost']['execution_id'] != value['execution_id']:
            raise HowHardError('billed_cost.execution_id: receipt does not match execution')
        estimate = segment['estimated_cost']
        if estimate is not None and 'rate_card' in estimate:
            card = estimate['rate_card']
            if (card['provider'], card['model'], card['service_tier']) != (segment['provider'], segment['model'], segment['service_tier']) or estimate['source'] != card['source'] or estimate['version'] != card['version']:
                raise HowHardError('estimated_cost: rate-card identity/provenance mismatch')
            amount = price(segment['tokens'], card)
            if amount is None or abs(amount - estimate['usd']) > 1e-9:
                raise HowHardError('estimated_cost: rate-card amount does not match measured categories')
        if not p.get('completed') and segment['estimated_cost'] is not None:
            raise HowHardError('execution_usage: incomplete source boundary cannot claim complete estimate')
    return value


def duplicates(rows):
    """Check effective executions, so replacing an execution's own claims is permitted."""
    owners = []
    for row in rows:
        usage = row.get('execution_usage')
        if usage is None:
            continue
        validate(usage)
        if usage['execution_id'] != row['id']:
            raise HowHardError('execution_usage.execution_id: must identify the original observation')
        for old, other in owners:
            if usage['execution_id'] == old['execution_id']:
                continue
            bills = {(s['billed_cost']['source'], s['billed_cost']['receipt_id']) for s in usage['segments'] if s['billed_cost'] is not None}
            old_bills = {(s['billed_cost']['source'], s['billed_cost']['receipt_id']) for s in old['segments'] if s['billed_cost'] is not None}
            if bills & old_bills:
                raise HowHardError('billed_cost: authoritative receipt already belongs to another execution')
            if any(a['source_id'] == b['source_id'] for a in usage['segments'] for b in old['segments']):
                raise HowHardError('execution_usage: source identity already belongs to another execution')
            for a in usage['claims']:
                for b in old['claims']:
                    if a['session_id'] == b['session_id'] and (a['coordinate_id'] != b['coordinate_id'] or ((a['end'] is None or b['start'] < a['end']) and (b['end'] is None or a['start'] < b['end']))):
                        raise HowHardError('execution_usage: source span overlaps another execution')
        owners.append((usage, row['id']))


def _count(value, key, default=None):
    if not isinstance(value, dict):
        raise HowHardError('source usage: expected object')
    count = value.get(key, default)
    if count is not None and (type(count) is not int or count < 0 or count > 2**63 - 1):
        raise HowHardError('source usage.' + key + ': expected nonnegative integer')
    return count


def _codex(u):
    inp, cached, out = (_count(u, k) for k in ('input_tokens', 'cached_input_tokens', 'output_tokens'))
    writes = _count(u, 'cache_write_input_tokens', 0)
    if inp is not None and cached is not None and ((cached + writes if writes is not None else cached) > inp):
        raise HowHardError('Codex cached input exceeds inclusive input')
    reasoning = _count(u, 'reasoning_output_tokens', 0)
    if out is not None and reasoning is not None and reasoning > out:
        raise HowHardError('Codex reasoning exceeds inclusive output')
    return dict(input=None if inp is None or cached is None or writes is None else inp - cached - writes,
                output=out, cache_read=cached, cache_write=writes, cache_write_5m=0, cache_write_1h=0)


def _codex_total(u):
    inp, out = _count(u, 'input_tokens'), _count(u, 'output_tokens')
    count = _count(u, 'total_tokens')
    if count is not None and inp is not None and out is not None and count != inp + out:
        raise HowHardError('Codex total_tokens: does not match inclusive input/output')
    return count if count is not None else inp + out if inp is not None and out is not None else None


def _claude(u, *, sdk=False):
    names = ('inputTokens', 'outputTokens', 'cacheReadInputTokens', 'cacheCreationInputTokens') if sdk else ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens')
    inp, out, read, write = (_count(u, k) for k in names)
    ttl = u.get('cache_creation')
    if ttl is not None:
        keys(ttl, ['ephemeral_5m_input_tokens', 'ephemeral_1h_input_tokens'], where='Claude cache creation')
        five, hour = (_count(ttl, k) for k in ('ephemeral_5m_input_tokens', 'ephemeral_1h_input_tokens'))
        if write is None or five is None or hour is None or five + hour > write:
            raise HowHardError('Claude cache TTL counts exceed or lack total')
        write -= five + hour
    else:
        five, hour = 0, 0
    return dict(input=inp, output=out, cache_read=read, cache_write=write, cache_write_5m=five, cache_write_1h=hour)


def collect(manifest):
    keys(manifest, ['format', 'execution_id', 'host_version', 'adapter', 'source', 'session_id', 'scope', 'start', 'end', 'completed', 'provider', 'model', 'service_tier'], ['included_sessions', 'rate_card', 'baseline', 'coordinate_id', 'agent_id'], where='usage manifest')
    if manifest['format'] != 'howhard-usage-manifest-v1' or manifest['adapter'] not in ADAPTERS:
        raise HowHardError('usage manifest: unsupported format/adapter')
    if manifest['host_version'] not in SUPPORTED_HOSTS[manifest['adapter']]:
        raise HowHardError('usage manifest.host_version: unsupported host/schema combination')
    if type(manifest['completed']) is not bool or manifest['scope'] not in ('self', 'inclusive'):
        raise HowHardError('usage manifest: explicit completion and scope required')
    for key in ('execution_id', 'provider', 'model', 'session_id'):
        text(manifest[key], 'usage manifest.' + key)
    for key in ('agent_id', 'coordinate_id', 'service_tier'):
        if key in manifest and manifest[key] is not None:
            text(manifest[key], 'usage manifest.' + key)
    if ('agent_id' in manifest and manifest['agent_id'] is None) or ('coordinate_id' in manifest and manifest['coordinate_id'] is None):
        raise HowHardError('usage manifest: agent_id/coordinate_id must be explicit nonempty identities')
    start, end = manifest['start'], manifest['end']
    if type(start) is not int or type(end) is not int or start < 0 or end <= start or end - start > MAX_SPAN:
        raise HowHardError('usage manifest: invalid or oversized byte span')
    path = Path(text(manifest['source'], 'usage manifest.source'))
    try:
        if not stat.S_ISREG(path.stat().st_mode):
            raise HowHardError('usage source: bounded regular file required')
        # Nonblocking open plus fstat also closes a regular-file-to-FIFO replacement race.
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(descriptor, 'rb') as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise HowHardError('usage source: bounded regular file required')
            if not path.is_file() or handle.seek(0, 2) > MAX_SOURCE or handle.tell() < end:
                raise HowHardError('usage source: unavailable span or oversized artifact')
            if start:
                handle.seek(start - 1)
                if handle.read(1) != b'\n':
                    raise HowHardError('usage span: start must be a complete record boundary')
            handle.seek(start)
            blob = handle.read(end - start)
            if len(blob) != end - start:
                raise HowHardError('usage source: source changed during bounded read')
    except OSError as exc:
        raise HowHardError('usage source: read failed; outcome remains independently durable') from exc
    if not blob.endswith(b'\n'):
        raise HowHardError('usage span: end must be a complete record boundary')
    try:
        events = [json.loads(line) for line in blob.splitlines()]
    except (ValueError, UnicodeError) as exc:
        raise HowHardError('usage source: invalid JSON records') from exc
    if any(not isinstance(e, dict) for e in events):
        raise HowHardError('usage source: expected event objects')
    adapter, session = manifest['adapter'], manifest['session_id']
    text(session, 'usage session_id')
    if adapter == 'claude-sdk-v1' and manifest['scope'] != 'inclusive':
        raise HowHardError('Claude SDK result modelUsage is child-inclusive; declare inclusive scope')
    if adapter == 'claude-sdk-v1' and 'included_sessions' not in manifest:
        raise HowHardError('Claude inclusive result requires explicit included_sessions (empty only when no children ran)')
    children = manifest.get('included_sessions', [])
    if not isinstance(children, list) or any(not isinstance(s, str) or not s or s == session for s in children) or len(set(children)) != len(children):
        raise HowHardError('usage manifest.included_sessions: distinct child session IDs required')
    if children and manifest['scope'] != 'inclusive':
        raise HowHardError('usage manifest: child claims require inclusive scope')
    # Coordinates must identify a single append-only artifact across captures. Without
    # host binding, cross-source same-session claims are conservatively incompatible.
    coordinate = manifest.get('coordinate_id', str(path.resolve()) + ':' + adapter)
    source_session = session + ':agent:' + manifest['agent_id'] if 'agent_id' in manifest else session
    claims = [{'session_id': source_session, 'start': start, 'end': end, 'coordinate_id': coordinate}] + [{'session_id': s, 'start': 0, 'end': None, 'coordinate_id': '*'} for s in children]
    result = dict(format=USAGE_FORMAT, execution_id=manifest['execution_id'], status='unavailable',
                  provenance={k: manifest[k] for k in ('adapter', 'host_version', 'session_id', 'start', 'end', 'scope')}, claims=claims, segments=[])
    result['provenance']['coordinate_id'] = coordinate
    if 'agent_id' in manifest:
        result['provenance']['agent_id'] = manifest['agent_id']
    result['provenance']['artifact_sha256'] = hashlib.sha256(blob).hexdigest()
    records, terminal, seen, previous = [], False, {}, None
    baseline = manifest.get('baseline')
    if adapter == 'claude-sdk-v1' and baseline is not None:
        keys(baseline, ['session_id', 'modelUsage'], where='Claude baseline')
        if baseline['session_id'] != session or not isinstance(baseline['modelUsage'], dict):
            raise HowHardError('Claude baseline: same-session measured starting totals required')
        result['provenance']['baseline_sha256'] = hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest()
    if adapter == 'codex-session-v1' and baseline is not None:
        if not isinstance(baseline, dict):
            raise HowHardError('Codex baseline: expected measured cumulative starting usage')
        _codex(baseline)
        previous = baseline
        result['provenance']['baseline_sha256'] = hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest()
    codex_seen_gap = False
    for event in events:
        if adapter == 'codex-exec-v1':
            if event.get('type') == 'thread.started' and event.get('thread_id') != session:
                raise HowHardError('Codex source session identity mismatch')
            if event.get('type') == 'turn.completed':
                u = event.get('usage')
                if not isinstance(u, dict):
                    raise HowHardError('Codex completed turn lacks usage')
                records.append((str(start) + ':' + str(len(records)), manifest['model'], manifest['service_tier'], _codex(u), None, _codex_total(u)))
                terminal = True
            elif event.get('type') in ('turn.started', 'turn.failed', 'error'):
                terminal = False
        elif adapter == 'codex-session-v1':
            p = event.get('payload', {})
            if not isinstance(p, dict):
                raise HowHardError('Codex transcript: invalid payload object')
            if event.get('type') == 'session_meta' and p.get('id') != session:
                raise HowHardError('Codex source session identity mismatch')
            if event.get('type') == 'event_msg' and p.get('type') == 'token_count':
                info = p.get('info')
                if info is None:
                    codex_seen_gap = True
                    continue
                if not isinstance(info, dict) or not isinstance(info.get('total_token_usage'), dict) or not isinstance(info.get('last_token_usage'), dict):
                    raise HowHardError('Codex transcript: unsupported token_count schema')
                current = info['total_token_usage']
                identity = json.dumps(current, sort_keys=True)
                if identity in seen:
                    continue
                if previous is not None and any(_count(current, k, 0) is not None and _count(previous, k, 0) is not None and current.get(k, 0) < previous.get(k, 0) for k in ('input_tokens', 'output_tokens')):
                    raise HowHardError('Codex cumulative usage reset; split at reset with explicit ownership')
                if previous is not None or start == 0:
                    for k in ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens', 'output_tokens'):
                        now, before, last = (_count(d, k, 0) for d in (current, previous if previous is not None else {}, info['last_token_usage']))
                        if None in (now, before, last) or now - before != last:
                            codex_seen_gap = True
                else:
                    codex_seen_gap = True  # A resumed span needs its measured cumulative starting point.
                previous = current
                seen[identity] = True
                stamp = text(event.get('timestamp'), 'Codex token_count timestamp')
                records.append((stamp + ':' + hashlib.sha256(identity.encode()).hexdigest(), manifest['model'], manifest['service_tier'], _codex(info['last_token_usage']), None, _codex_total(info['last_token_usage'])))
                terminal = False
            if event.get('type') == 'event_msg' and p.get('type') == 'task_complete':
                terminal = True
            elif event.get('type') == 'event_msg' and p.get('type') in ('task_started', 'turn_aborted'):
                terminal = False
        elif adapter == 'claude-sdk-v1':
            if event.get('type') != 'result':
                continue
            if event.get('session_id') != session:
                raise HowHardError('Claude result session identity mismatch')
            if records:
                raise HowHardError('Claude result: one bounded invocation result required; do not sum resume totals')
            models = event.get('modelUsage')
            if not isinstance(models, dict) or not models:
                raise HowHardError('Claude result: modelUsage required')
            if baseline is None:
                terminal = False
                continue  # Cumulative totals with unknown starting point are not invocation usage.
            if set(baseline['modelUsage']) - set(models):
                raise HowHardError('Claude cumulative usage reset: baseline model missing')
            for model, u in models.items():
                if not isinstance(u, dict):
                    raise HowHardError('Claude result: invalid model usage')
                estimate = u.get('costUSD')
                if estimate is not None:
                    number(estimate, 'Claude result estimated cost', nonnegative=True)
                counts = _claude(u, sdk=True)
                prior = baseline['modelUsage'].get(model)
                if prior is not None:
                    before = _claude(prior, sdk=True)
                    for key in CATEGORIES:
                        if counts[key] is None or before[key] is None:
                            counts[key] = None
                        else:
                            counts[key] -= before[key]
                            if counts[key] < 0:
                                raise HowHardError('Claude cumulative usage reset; split execution at reset')
                    if estimate is not None:
                        if prior.get('costUSD') is None:
                            estimate = None
                        else:
                            number(prior['costUSD'], 'Claude baseline estimated cost', nonnegative=True)
                            estimate -= prior['costUSD']
                            if estimate < 0:
                                raise HowHardError('Claude cumulative cost reset; split execution at reset')
                result_id = text(event.get('uuid'), 'Claude result UUID')
                records.append((result_id + ':' + model, model, manifest['service_tier'], counts, estimate, total(counts)))
            # Never replay cumulative restored spend as a fresh invocation's estimate.
            if event.get('num_turns') is not None and type(event['num_turns']) is not int:
                raise HowHardError('Claude result: invalid turn count')
            if type(event.get('is_error')) is not bool:
                raise HowHardError('Claude result.is_error: explicit boolean required')
            subtype = event.get('subtype')
            terminal = subtype in ('success', 'error_max_turns', 'error_max_budget_usd')
            if terminal and event.get('is_error', False) != (subtype != 'success'):
                raise HowHardError('Claude result: contradictory terminal failure metadata')
        else:
            if event.get('type') != 'assistant':
                continue
            if event.get('sessionId') != session or event.get('isSidechain', False) != ('agent_id' in manifest) or ('agent_id' in manifest and event.get('agentId') != manifest['agent_id']):
                raise HowHardError('Claude transcript: session/child scope mismatch')
            msg = event.get('message')
            if not isinstance(msg, dict) or not isinstance(msg.get('usage'), dict):
                raise HowHardError('Claude transcript: unsupported assistant usage schema')
            ident = text(msg.get('id'), 'Claude message ID')
            normalized = (ident, text(msg.get('model'), 'Claude message model'), msg['usage'].get('service_tier', manifest['service_tier']), _claude(msg['usage']), None, total(_claude(msg['usage'])))
            if ident in seen:
                # Streaming records can revise usage for the same assistant message.
                records[seen[ident]] = normalized
            else:
                seen[ident] = len(records)
                records.append(normalized)
            terminal = msg.get('stop_reason') == 'end_turn'
    result['provenance']['completed'] = bool(records) and terminal and manifest['completed'] and not codex_seen_gap
    complete = bool(records) and terminal and manifest['completed'] and not codex_seen_gap and all(total(r[3]) is not None for r in records)
    result['status'] = 'complete' if complete else 'partial' if records else 'unavailable'
    for ident, model, tier, counts, estimate, reported in records:
        segment = dict(source_id=source_session + ':' + adapter + ':' + ident, provider=manifest['provider'], model=model,
                       service_tier=tier, tokens=counts, reported_tokens=reported, estimated_cost=None, billed_cost=None)
        if result['provenance']['completed'] and estimate is not None:
            segment['estimated_cost'] = dict(usd=estimate, source='claude-client', version=manifest['host_version'])
        card = manifest.get('rate_card')
        if card is not None:
            rate_card(card)
            if (card['provider'], card['model'], card['service_tier']) != (segment['provider'], model, tier):
                raise HowHardError('rate_card: source provider/model/tier mismatch')
            amount = price(counts, card)
            if result['provenance']['completed'] and amount is not None:
                segment['estimated_cost'] = dict(usd=amount, source=card['source'], version=card['version'], rate_card=deepcopy(card))
        result['segments'].append(segment)
    return validate(result)


def summary(rows):
    """Separate money bases and coverage without turning missing measurements into zero."""
    result = {"runs": len(rows), "statuses": {s: 0 for s in ("complete", "partial", "unavailable", "missing")},
              "tokens": {k: {"known": 0, "unknown_segments": 0, "unknown_runs": 0} for k in CATEGORIES}}
    for field in ("estimated_cost", "billed_cost"):
        known, unknown = 0.0, 0
        for row in rows:
            usage = row.get('execution_usage')
            if not usage or not usage['segments']:
                unknown += 1
                continue
            amounts = [s[field] for s in usage['segments']]
            known += sum(v['usd'] for v in amounts if v is not None)
            unknown += int(any(v is None for v in amounts) or (field == 'estimated_cost' and not usage['provenance'].get('completed')))
        result[field] = dict(known_usd=known, unknown_runs=unknown, total_usd=known if not unknown else None)
    for row in rows:
        usage = row.get('execution_usage')
        result['statuses'][usage['status'] if usage else 'missing'] += 1
        for key in CATEGORIES:
            if not usage or not usage['segments'] or not usage['provenance'].get('completed') or any(s['tokens'][key] is None for s in usage['segments']):
                result['tokens'][key]['unknown_runs'] += 1
        for segment in usage['segments'] if usage else []:
            for key, count in segment['tokens'].items():
                if count is None:
                    result['tokens'][key]['unknown_segments'] += 1
                else:
                    result['tokens'][key]['known'] += count
    for category in result['tokens'].values():
        category['total'] = category['known'] if not category['unknown_runs'] else None
    return result
