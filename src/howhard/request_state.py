"""Pinned request reads and validation of the dependencies actually surfaced."""

from copy import copy, deepcopy
import re

from . import storage
from .context import digest
from .validation import HowHardError, keys


class StaleRequest(HowHardError):
    """A surfaced input changed; forecasts must be reconsidered explicitly."""

    def __init__(self, changes, bundle):
        self.changes, self.bundle = changes, bundle
        self.request_id = digest(bundle)
        self.refreshed_request = None
        self.refresh_error = None
        self.audit_error = None
        self.baseline_unchanged = False
        names = ', '.join(
            f"memory {c['id']} version {c['old_version']} -> {c['new_version']} "
            f"(active {c['old_active']} -> {c['new_active']})" if c['kind'] == 'memory'
            else f"{c['kind']} {c['id']} digest {c['old_digest']} -> {c['new_digest']}"
            for c in changes)
        super().__init__('request_id: stale dependencies: ' + names)

    def __str__(self):
        message = super().__str__()
        if self.refreshed_request is not None:
            message += '; refreshed request: ' + self.refreshed_request['request_id']
        return message

    def as_dict(self):
        result = {'code': 'stale_request', 'message': str(self),
                  'request_id': self.request_id, 'changes': deepcopy(self.changes),
                  'refreshed_request': self.refreshed_request,
                  'baseline_unchanged': self.baseline_unchanged}
        if self.refresh_error is not None:
            result['refresh_error'] = self.refresh_error
        if self.audit_error is not None:
            result['audit_error'] = deepcopy(self.audit_error)
        return result


def configuration(config):
    models = {k: v for k, v in config.data['models'].items() if not v.get('native')}
    return digest({**config.data, 'models': models})


def validate_provenance(value):
    keys(value, ['weights_policy', 'dependencies', 'evidence_key', 'memory_revision'], where='request_snapshot')
    if value['weights_policy'] != 'snapshot' or type(value['memory_revision']) is not int or value['memory_revision'] < 0:
        raise HowHardError('request_snapshot: invalid weights policy or revision')
    def checksum(item):
        if not isinstance(item, str) or not re.fullmatch(r'[0-9a-f]{64}', item):
            raise HowHardError('request_snapshot: expected a SHA-256 dependency digest')
    checksum(value['evidence_key'])
    manifest = value['dependencies']
    keys(manifest, ['configuration', 'memories', 'evidence'], where='request dependencies')
    checksum(manifest['configuration'])
    for name in ('memories', 'evidence'):
        if not isinstance(manifest[name], dict):
            raise HowHardError('request dependencies: expected mappings')
        for id, item in manifest[name].items():
            if not isinstance(id, str) or not id:
                raise HowHardError('request dependencies: expected nonempty IDs')
            if name == 'memories':
                keys(item, ['version', 'active', 'digest'], where='memory dependency')
                if type(item['version']) is not int or item['version'] < 1 or type(item['active']) is not bool:
                    raise HowHardError('request dependencies: invalid memory version or status')
                item = item['digest']
            checksum(item)


class FrozenMemory:
    def __init__(self, snapshot, retrieval):
        self.state, self.retrieval = snapshot, retrieval

    def snapshot(self, *, include_removed=False):
        result = deepcopy(self.state)
        if not include_removed:
            result['memories'] = [m for m in result['memories'] if m['active']]
        return result

    def read(self, memory_id, *, version=None):
        for memory in self.state['memories']:
            if memory['id'] == memory_id and (version is None or version == memory['version']):
                return deepcopy(memory)
        raise HowHardError(f'memory.id: unavailable in request snapshot: {memory_id}')

    def search(self, query, *, limit=8, include_removed=False):
        from .retrieval import records
        memories = self.snapshot(include_removed=include_removed)['memories']
        return [r['value'] for r in self.retrieval.search(records(memories=memories), query, limit=limit)['records']]


def view(engine, state):
    result = copy(engine)
    result.config = copy(engine.config)
    result.config.data = deepcopy(engine.config.data)
    result.config.native_models = deepcopy(engine.config.native_models)
    result.raw_rows = deepcopy(state['raw_rows'])
    result.rows = storage.effective_rows(result.config, result.raw_rows)
    result.memory = FrozenMemory(state['memory'], engine.retrieval)
    result._summaries = {}
    return result


def capture(engine):
    # Initialize before taking shared data locks: initialization itself needs a turn.
    engine.memory.snapshot()
    with storage.reading(engine.config):
        engine.refresh()
        state = {'raw_rows': deepcopy(engine.raw_rows),
                 'memory': engine.memory.snapshot(include_removed=True)}
        return view(engine, state), state


def extend(engine, bundle):
    """Union surfaced dependencies across every stage, using the pinned values."""
    if 'read_state' not in bundle:
        return
    state = bundle['read_state']
    sources = view(engine, state)
    memories = {m['id']: m for m in state['memory']['memories']}
    rows = {r['id']: r for r in sources.rows}
    manifest = bundle.setdefault('dependencies', {'configuration': configuration(engine.config),
                                                'memories': {}, 'evidence': {}})

    def memory(item):
        if item['id'].startswith('execution:'):
            evidence(item['id'][10:])
        elif item['id'] in memories:
            old = memories[item['id']]
            manifest['memories'].setdefault(old['id'], {'version': old['version'],
                'active': old['active'], 'digest': digest(old)})

    def evidence(id):
        # evidence.read may name an amendment; bind its effective original.
        raw = {r['id']: r for r in state['raw_rows']}
        while id in raw and raw[id]['amends'] is not None:
            id = raw[id]['amends']
        if id in rows:
            manifest['evidence'].setdefault(id, digest(rows[id]))

    payload = bundle['payload']
    for item in payload['memories']:
        memory(item)
    for item in payload['evidence']:
        evidence(item['id'])
    for retrieval in payload['retrieval']:
        name, data = retrieval['request']['tool'], retrieval['result']
        if name.startswith(('memory.', 'evidence.')):
            for item in data if isinstance(data, list) else [data]:
                if name.startswith('memory.'):
                    memory(item)
                else:
                    evidence(item['id'])


def changes(engine, bundle):
    manifest = bundle['dependencies']
    result = []
    current_config = configuration(engine.config)
    if manifest['configuration'] != current_config:
        result.append({'kind': 'configuration', 'id': 'configuration',
                       'old_digest': manifest['configuration'], 'new_digest': current_config})
    current = {m['id']: m for m in engine.memory.snapshot(include_removed=True)['memories']}
    for id, old in manifest['memories'].items():
        new = current.get(id)
        if new is None or old['digest'] != digest(new):
            result.append({'kind': 'memory', 'id': id, 'old_version': old['version'],
                           'new_version': new['version'] if new else None,
                           'old_active': old['active'], 'new_active': new['active'] if new else None,
                           'old_digest': old['digest'], 'new_digest': digest(new) if new else None})
    rows = {r['id']: r for r in engine.rows}
    for id, old in manifest['evidence'].items():
        new = digest(rows[id]) if id in rows else None
        if old != new:
            result.append({'kind': 'evidence', 'id': id, 'old_digest': old, 'new_digest': new})
    return result
