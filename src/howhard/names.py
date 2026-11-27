"""Names for runs whose receipt stores none: the host of an older decision and the agent of a family.

A ``howhard-decision-v11`` receipt stores the host that asked for the route and the agent
label of each candidate. Older receipts store neither, and a run recorded with no decision
has no receipt at all; all of them belong to the one unnamed host and the one unnamed agent
until an operator names them with one optional file::

    {"hosts":  {"<decision id>": "<host name>"},
     "agents": {"<family>": "<agent label>"}}

``hosts`` names the host of a stored decision older than v11, and so of its runs. ``agents``
names the agent of a family for every run whose receipt stores no agent label: receipts older
than v11 and runs with no decision. A registered model id stands for its family. The names
live in two tables of the payload store beside the receipts, so no receipt is rewritten (its
identifier is the digest of its content). A v11 decision is never renamed.
"""

import json

from .context import DECISION_FORMAT
from .validation import HowHardError, keys, name

TABLES = ("hosts", "agents")
EMPTY = {"hosts": {}, "agents": {}}


def read(config):
    """The imported names of the store: ``{"hosts": {decision_id: host}, "agents": {family: agent}}``."""
    from .payload_store import imported_names
    return imported_names(config)


def host(receipt, decision_id, names):
    """The host of a stored decision; ``None`` is the one unnamed host.

    A v11 receipt's own name, never an imported one; an older receipt's imported name.
    """
    if receipt.get("format") == DECISION_FORMAT:
        return receipt.get("host")
    return (names or EMPTY)["hosts"].get(decision_id)


def family_agent(family, names):
    """The imported agent of a family, for a run whose receipt stores no agent label."""
    return (names or EMPTY)["agents"].get(family)


def parse(value):
    """Check the file's shape and every name; return its two tables."""
    if not isinstance(value, dict):
        raise HowHardError("names: expected an object with optional hosts and agents tables")
    keys(value, [], TABLES, where="names")
    tables = {}
    for table in TABLES:
        entries = value.get(table, {})
        if not isinstance(entries, dict):
            raise HowHardError(f"names.{table}: expected an object")
        for key, label in entries.items():
            name(label, f"names.{table}.{key}")
        tables[table] = dict(entries)
    return tables


def _stored_format(config, decision_id):
    from .payload_store import ID, get
    if not isinstance(decision_id, str) or not ID.fullmatch(decision_id):
        return None
    blob = get(config, "decisions", decision_id)
    if blob is None:
        return None
    try:
        value = json.loads(blob)
    except (ValueError, UnicodeError) as exc:
        raise HowHardError(f"names.hosts: cannot read stored decision {decision_id}") from exc
    return value.get("format") if isinstance(value, dict) else None


def _families(config):
    """Each registered family, and each registered model id with the family it stands for."""
    models = config.data["models"]
    return {spec["family"] for spec in models.values()}, {model: spec["family"] for model, spec in models.items()}


def _count(wanted, current):
    stored = {k: v for k, v in wanted.items() if k not in current}
    replaced = {k: v for k, v in wanted.items() if k in current and current[k] != v}
    unchanged = [k for k, v in wanted.items() if current.get(k) == v]
    return stored, replaced, unchanged


def import_names(config, value):
    """Store the names of one file under the storage lock and count what happened to each entry.

    An entry for a v11 decision, for an identifier that names no stored decision, or for a
    family or model that is not registered is skipped and reported; the rest are stored. The
    same file twice writes nothing, and a later file replaces names an earlier one gave.
    """
    from .payload_store import imported_names, store_names
    from .storage import locked
    tables = parse(value)
    skipped = []
    with locked(config, operation="names import"):
        current = imported_names(config)
        hosts = {}
        for decision_id, label in sorted(tables["hosts"].items()):
            stored = _stored_format(config, decision_id)
            if stored is None:
                skipped.append({"table": "hosts", "key": decision_id, "reason": "not a stored decision"})
            elif stored == DECISION_FORMAT:
                skipped.append({"table": "hosts", "key": decision_id,
                                "reason": "a howhard-decision-v11 receipt stores its own host"})
            else:
                hosts[decision_id] = label
        families, models = _families(config)
        agents, sources = {}, {}
        for key, label in sorted(tables["agents"].items()):
            family = key if key in families else models.get(key)
            if family is None:
                skipped.append({"table": "agents", "key": key, "reason": "not a registered family or model"})
                continue
            if family in agents and agents[family] != label:
                raise HowHardError(f"names.agents: {sources[family]!r} and {key!r} name family {family!r} differently")
            agents[family], sources[family] = label, key
        plans = {"hosts": _count(hosts, current["hosts"]), "agents": _count(agents, current["agents"])}
        writes = {table: {**plan[0], **plan[1]} for table, plan in plans.items()}
        if any(writes.values()):
            store_names(config, writes["hosts"], writes["agents"])
    result = {table: {"stored": len(plan[0]), "unchanged": len(plan[2]), "replaced": len(plan[1]),
                      "skipped": sum(item["table"] == table for item in skipped)}
              for table, plan in plans.items()}
    result["skipped"] = skipped
    return result
