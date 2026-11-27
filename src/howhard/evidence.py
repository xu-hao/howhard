"""One scored candidate set per decision, shared by the three evidence readers.

A candidate is a recorded run with outcome clean, loud or silent that belongs to at least
one prototype the new task matches. Prototypes only choose the candidates: no prototype
score and no membership weight enters a weight. Each candidate is compared with the new
task once, when the request is prepared, and every reader weighs it from that one score::

    sim = 1 / (1 + exp(-r))                r: reranker score of (new task text, the run's task text)
    a   = 0.5 ** (days / half_life_days)   days: from the run's observed_at to the prepare time
    w   = m * sim * factor * a             m: the run's weight on record; factor: the reader's own
    w  <- w / max(1, W_t)                  W_t: the total w of the counted runs of one ticket

The outcome reader's factor is the relatedness discount ``delta``: a run by a related cell
counts as a fraction of a run by the judged cell itself (``relation`` and ``discount``).
"""

from datetime import datetime, timezone
import math

from .storage import effective_active_seconds
from .validation import HowHardError, keys, name, number, text

OUTCOMES = ("clean", "loud", "silent")
CANDIDATE_FIELDS = ("id", "family", "effort", "agent", "outcome", "active_seconds", "tokens", "token_usage", "ticket",
                    "run_weight", "score", "similarity", "days", "age_factor")
TOLERANCE = 1e-10
# The sampling-policy key that holds the discount of each relation between the cell that ran
# and the cell being judged. The fourth relation, ``cell``, is the cell's own run and counts in full.
DISCOUNT_KEYS = {"family": "effort_discount", "agent": "model_discount", "other": "agent_discount"}


def similarity(score):
    """The reranker's score on a 0-to-1 scale. There is no cut-off: a negative score still counts."""
    score = float(score)
    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))
    return math.exp(score) / (1.0 + math.exp(score))


def age_factor(days, half_life_days):
    """Half as much for every ``half_life_days`` between the run and the request."""
    return 0.5 ** (max(0.0, float(days)) / float(half_life_days))


def relation(run_cell, cell):
    """How the cell that ran relates to the cell being judged; each is ``(family, effort, agent)``.

    The first line that fits is used: ``cell`` for the same family and effort, ``family`` for
    the same family at another effort, ``agent`` for another family of the same agent, and
    ``other`` for another agent. An agent of ``None`` is the one unnamed agent.
    """
    if run_cell[0] == cell[0]:
        return "cell" if run_cell[1] == cell[1] else "family"
    return "agent" if run_cell[2] == cell[2] else "other"


def discount(related, policy):
    """The relatedness discount ``delta`` of a relation under the sampling policy."""
    return 1.0 if related == "cell" else float(policy[DISCOUNT_KEYS[related]])


def weight(candidate, factor):
    """One run's weight before the per-ticket limit; ``factor`` is the reader's own multiplier."""
    return candidate["run_weight"] * candidate["similarity"] * factor * candidate["age_factor"]


def ticket_shares(entries):
    """The factor ``1 / max(1, W_t)`` for each ``(ticket, weight)``: one ticket counts for at most one run.

    Retries follow failures, so the runs of one ticket are not independent. A run with no
    ticket is not limited.
    """
    grouped = {}
    for ticket, value in entries:
        if ticket:
            grouped.setdefault(ticket, []).append(value)
    totals = {ticket: math.fsum(values) for ticket, values in grouped.items()}
    return [1.0 / max(1.0, totals[ticket]) if ticket else 1.0 for ticket, _ in entries]


def _observed(row):
    return datetime.fromisoformat(row["observed_at"])


def collect(engine, payload, proto_state, receipts=None):
    """Choose and score the candidates once, on the caller's captured rows and outside any storage lock.

    ``receipts`` is an optional cache of the decisions the candidate rows name, shared with
    the capability reader so that each receipt is loaded once per request.
    """
    from . import prototypes
    from .decisions import run_agent
    from .accounting import usage_by_kind
    from .retrieval import records, task_text
    policy = engine.config.data["sampling"]
    half_life, cap = policy["half_life_days"], policy["comparison_cap"]
    prepared = datetime.now(timezone.utc)
    members = prototypes.matched_members(engine, proto_state)
    rows = [(index, row) for index, row in enumerate(engine.rows)
            if row["outcome"] in OUTCOMES and row["id"] in members]
    # Most recent first; the journal position orders runs recorded at the same instant.
    rows.sort(key=lambda item: (_observed(item[1]), item[0]), reverse=True)
    matched, rows = len(rows), [row for _, row in rows[:cap]]
    scores = []
    if rows:
        query = task_text(payload["task_type"], payload["context"])
        try:
            # Each run is represented by its own task text, as for the precedents shown to the reader.
            scores = list(engine.retrieval.reranker.score(query, [doc["text"] for doc in records(rows=rows)]))
        except HowHardError:
            raise
        except Exception as exc:
            raise HowHardError(f"retrieval failed ({type(exc).__name__}): {str(exc)[:300]}") from exc
        if len(scores) != len(rows) or any(not math.isfinite(float(v)) for v in scores):
            raise HowHardError("evidence similarity: invalid reranker scores")
    candidates = []
    for row, score in zip(rows, scores):
        days = max(0.0, (prepared - _observed(row)).total_seconds() / 86400.0)
        # What the run recorded for the beliefs of its own cell: active time, total tokens, and
        # usage by kind when its execution usage is complete.
        # The run's agent is the label its decision stored for the chosen cell; a run with no
        # decision, an older receipt or an unlabeled candidate belongs to the one unnamed agent.
        candidates.append({"id": row["id"], "family": row["family"], "effort": row["resolved_effort"],
                           "agent": run_agent(engine.config, row, receipts), "outcome": row["outcome"], "active_seconds": effective_active_seconds(row),
                           "tokens": row.get("tokens"), "token_usage": usage_by_kind(row.get("execution_usage")),
                           "ticket": row["ticket"], "run_weight": float(row["weight"]),
                           "score": float(score), "similarity": similarity(score),
                           "days": days, "age_factor": age_factor(days, half_life)})
    return {"prepared_at": prepared.isoformat(), "half_life_days": half_life, "comparison_cap": cap,
            "matched": matched, "candidates": candidates}


def close(actual, expected):
    """Whether a stored number equals its recomputed value."""
    return (type(actual) in (int, float) and math.isfinite(actual)
            and abs(actual - expected) <= TOLERANCE * max(1.0, abs(expected)))


def validate(state, *, where="decision.semantic.evidence_candidates"):
    """Check a stored candidate set, recomputing each similarity and age factor."""
    keys(state, ["prepared_at", "half_life_days", "comparison_cap", "matched", "candidates"], where=where)
    try:
        if not isinstance(state["prepared_at"], str) or datetime.fromisoformat(state["prepared_at"]).utcoffset() is None:
            raise ValueError
    except ValueError as exc:
        raise HowHardError(f"{where}.prepared_at: expected ISO timestamp with timezone") from exc
    half_life = number(state["half_life_days"], f"{where}.half_life_days", positive=True)
    cap, matched, candidates = state["comparison_cap"], state["matched"], state["candidates"]
    if type(cap) is not int or cap < 1 or type(matched) is not int or matched < 0 or not isinstance(candidates, list):
        raise HowHardError(f"{where}: expected a positive comparison_cap, a candidate count and an array")
    if len(candidates) > cap:
        raise HowHardError(f"{where}: more than comparison_cap candidates")
    if len(candidates) != min(matched, cap):
        raise HowHardError(f"{where}: candidate count does not match the scored candidates")
    seen = set()
    for item in candidates:
        keys(item, CANDIDATE_FIELDS, where=f"{where}.candidate")
        for field in ("id", "family", "effort"):
            text(item[field], f"{where}.candidate.{field}")
        if item["id"] in seen or item["outcome"] not in OUTCOMES:
            raise HowHardError(f"{where}.candidate: duplicate ID or unknown outcome")
        seen.add(item["id"])
        if item["agent"] is not None:
            name(item["agent"], f"{where}.candidate.agent")
        if item["active_seconds"] is not None:
            number(item["active_seconds"], f"{where}.candidate.active_seconds", nonnegative=True)
        if item["tokens"] is not None:
            number(item["tokens"], f"{where}.candidate.tokens", nonnegative=True)
        if item["token_usage"] is not None:
            # Usage by kind is kept only from a complete execution usage, which measures every kind.
            from .accounting import tokens as token_counts
            try:
                counts = token_counts(item["token_usage"])
            except HowHardError as exc:
                raise HowHardError(f"{where}.candidate.{exc}") from exc
            if any(count is None for count in counts.values()):
                raise HowHardError(f"{where}.candidate.token_usage: expected a count for every kind")
        if item["ticket"] is not None and not isinstance(item["ticket"], str):
            raise HowHardError(f"{where}.candidate.ticket: expected string or null")
        number(item["run_weight"], f"{where}.candidate.run_weight", positive=True)
        score = number(item["score"], f"{where}.candidate.score")
        days = number(item["days"], f"{where}.candidate.days", nonnegative=True)
        if not close(item["similarity"], similarity(score)):
            raise HowHardError(f"{where}.candidate: similarity does not match its score")
        if not close(item["age_factor"], age_factor(days, half_life)):
            raise HowHardError(f"{where}.candidate: age factor does not match its days")
    return state


def same(actual, expected, where):
    """Refuse stored reader rows that differ from the rows recomputed from the candidates."""
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise HowHardError(f"{where}: does not match the scored candidates")
    for stored, wanted in zip(actual, expected):
        keys(stored, list(wanted), where=where)
        for field, value in wanted.items():
            matches = close(stored[field], value) if type(value) is float else stored[field] == value
            if not matches:
                raise HowHardError(f"{where}: {field} does not match the scored candidates")
    return actual
