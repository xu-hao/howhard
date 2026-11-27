"""Dynamic synthesized (task, context) prototypes with materialized per-cell priors and stats."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import re
import sqlite3

import numpy as np

from .accounting import CATEGORIES, total as token_total
from .context import PROJECT_FORMAT, digest
from .llm import SETTINGS
from .storage import effective_active_seconds, locked
from .validation import HowHardError

OUTCOMES = ("clean", "loud", "silent")
# What a stored Stage 2 forecast keeps: its values, every stated range, and (as "host") who made it.
FORECAST_FIELDS = ("clean", "loud", "silent", "tokens", "seconds", "token_usage",
                   "loud_ci", "silent_ci", "seconds_ci", "token_usage_ci")
EMPIRICAL_BASE = (0.88, 0.08, 0.04)
OVERRUN_GRACE_RATIO = 1.5
STAGE1_MIN_CAP_WEIGHT = 0.5
STAGE2_MIN_EMP_WEIGHT = 0.5
BACKFILL_BATCH_THRESHOLD = 32
BACKFILL_COSINE_THRESHOLD = 0.82
SPLIT_MIN_MEMBERS = 12
SPLIT_MIN_CHILD = 3
SPLIT_DURATION_DISPERSION = 0.65
SPLIT_DURATION_MIN_RUNS = 6
SPLIT_OUTCOME_MIX_RATIO = 0.25
SPLIT_OUTCOME_MIN_RUNS = 8
SPLIT_COSINE_DISTANCE = 0.18
# Per-observation decay of a prototype cell's stored outcome counts.
STATS_OUTCOME_DECAY = 0.95

_MATCH_CACHE = {}


def _sigmoid(x):
    x = max(-60.0, min(60.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def canonical_task_text(task_type, raw_text):
    """Deterministically strip one-off ticket/commit/scratch wrappers while keeping task words."""
    value = raw_text if isinstance(raw_text, str) else ""
    value = re.sub(
        r"^\s*You are (?:an|the) assigned (?:[A-Za-z0-9_-]+\s+){0,4}worker"
        r"(?:\s+for\s+[^\n:.—-]{1,80})?[:.\s—-]*",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(r"(?:/var/tmp|/tmp)/[^\s\"'`,;)\]]+", "", value)
    value = re.sub(r"(?<!\w)#\d+\b", "", value)
    value = re.sub(
        r"\b(?=[0-9a-f]{7,64}\b)(?:[0-9a-f]*\d[0-9a-f]*|[0-9a-f]{16,64})\b",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(r"\s+", " ", value).strip(" -:;,. \t\r\n")
    if len(value) > 1024:
        value = value[:1024].rsplit(" ", 1)[0] or value[:1024]
    return value or f"General {task_type} task"


def synthesize_representative(task_type, context):
    """Synthesize a clean reusable (representative_task, representative_context) descriptor."""
    ctx = context if isinstance(context, dict) else {}
    rep_task = canonical_task_text(task_type, ctx.get("task_text", ""))
    proj = ctx.get("project_context")
    if isinstance(proj, dict) and proj.get("format") == PROJECT_FORMAT:
        raw_text = proj.get("text", "")
        excerpt = raw_text[:2048]
        rep_context = {
            "format": PROJECT_FORMAT,
            "parameters": deepcopy(proj["parameters"]),
            "text": excerpt,
            "revision": proj.get("revision"),
            "fingerprint": proj["fingerprint"],
            "truncated": bool(proj.get("truncated") or len(raw_text) > 2048),
        }
    else:
        rep_context = None
    return {
        "task_type": task_type,
        "representative_task": rep_task,
        "representative_context": rep_context,
    }


def prototype_context_for_baseline(context, representative):
    proj = representative.get("representative_context")
    return {
        "encoder": context["encoder"],
        "task_text": representative["representative_task"],
        "project_context": deepcopy(proj if proj is not None else context["project_context"]),
    }


def empty_stats():
    return {
        "counts": [0.0, 0.0, 0.0],
        "capability_weight": 0.0,
        "capability_clean": 0.0,
        "timely_clean": 0.0,
        "duration_w": 0.0,
        "duration_mean_log": 0.0,
        "duration_m2_log": 0.0,
        "tokens_w": 0.0,
        "tokens_mean_log": 0.0,
        "token_shares_w": 0.0,
        "token_usage_shares": None,
        "run_count": 0,
    }


def _row_signature(row):
    return [
        row["id"],
        row["outcome"],
        row.get("cause"),
        row.get("tokens"),
        row.get("seconds"),
        row.get("active_seconds"),
        row.get("execution_usage"),
        row.get("note", ""),
        row.get("weight", 1.0),
    ]


def journal_digest(rows):
    return digest([_row_signature(r) for r in rows])


def _has_tables(db):
    if db is None:
        return False
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prototypes'"
    ).fetchone()
    return row is not None


def _read_meta(config, key):
    from .payload_store import _connect

    db = _connect(config, write=False)
    if db is None:
        return None
    try:
        if not _has_tables(db):
            return None
        row = db.execute("SELECT value FROM prototype_meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None
    finally:
        db.close()


def _load_active_prototypes(db, *, task_type=None):
    if not _has_tables(db):
        return []
    if task_type is None:
        rows = db.execute(
            "SELECT prototype_id, task_type, risk_class, representative_task, representative_context, "
            "active, parent_id, member_count, created_at FROM prototypes WHERE active=1 "
            "ORDER BY created_at ASC, prototype_id ASC"
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT prototype_id, task_type, risk_class, representative_task, representative_context, "
            "active, parent_id, member_count, created_at FROM prototypes WHERE active=1 AND task_type=? "
            "ORDER BY created_at ASC, prototype_id ASC",
            (task_type,),
        ).fetchall()
    cols = (
        "prototype_id",
        "task_type",
        "risk_class",
        "representative_task",
        "representative_context",
        "active",
        "parent_id",
        "member_count",
        "created_at",
    )
    return [dict(zip(cols, r)) for r in rows]


def _load_cells(db, prototype_ids):
    if not prototype_ids or not _has_tables(db):
        return {}
    placeholders = ",".join("?" for _ in prototype_ids)
    rows = db.execute(
        f"SELECT prototype_id, model, family, effort, prior_json, stats_json "
        f"FROM prototype_cells WHERE prototype_id IN ({placeholders})",
        list(prototype_ids),
    ).fetchall()
    result = {pid: {} for pid in prototype_ids}
    for pid, model, family, effort, prior_raw, stats_raw in rows:
        result.setdefault(pid, {})[(model, effort)] = {
            "model": model,
            "family": family,
            "effort": effort,
            "prior": json.loads(prior_raw) if prior_raw is not None else None,
            "stats": json.loads(stats_raw),
        }
    return result


def _extract_category_shares(row):
    usage = row.get("execution_usage")
    if not usage or not usage.get("provenance", {}).get("completed") or not usage.get("segments"):
        return None
    sums = {k: 0.0 for k in CATEGORIES}
    for seg in usage["segments"]:
        tok = seg.get("tokens")
        if not tok or token_total(tok) is None:
            return None
        for k in CATEGORIES:
            sums[k] += float(tok[k])
    tot = math.fsum(sums.values())
    if tot <= 0:
        return None
    return [sums[k] / tot for k in CATEGORIES]


def _apply_member_to_stats(stats, row, member_weight, config, prior=None):
    """Apply one non-interrupted member observation using exact exponential-decay recurrences."""
    if row["outcome"] == "interrupted" or row["outcome"] not in OUTCOMES:
        return stats
    out_recency = STATS_OUTCOME_DECAY
    cap_recency = SETTINGS["belief"]["recency"]
    cap_policy = config.data["capability"]

    u = float(row.get("weight", 1.0)) * float(member_weight)
    if u <= 0:
        return stats

    counts = [c * out_recency for c in stats["counts"]]
    counts[OUTCOMES.index(row["outcome"])] += u
    stats["counts"] = counts
    stats["run_count"] = int(stats["run_count"]) + 1

    strength = 1.0
    if row["outcome"] != "clean":
        strength = cap_policy[row["outcome"] + "_weight"]
        cause = row.get("cause")
        if cause != "capability":
            strength *= cap_policy["unknown_cause_weight"]
            if re.search(r"\bcapability_caused\s*[:=]\s*false\b", row.get("note", ""), re.I):
                strength = 0.0
    w_cap = u * strength
    stats["capability_weight"] = stats["capability_weight"] * cap_recency + w_cap
    stats["capability_clean"] = stats["capability_clean"] * cap_recency + (
        w_cap if row["outcome"] == "clean" else 0.0
    )
    if row["outcome"] == "clean":
        active = effective_active_seconds(row)
        prior_sec = prior.get("seconds") if isinstance(prior, dict) else None
        if (
            active is not None
            and active > 0
            and prior_sec is not None
            and prior_sec > 0
            and max(1.0, active) / max(1.0, float(prior_sec)) > OVERRUN_GRACE_RATIO
        ):
            ratio = max(1.0, active) / max(1.0, float(prior_sec))
            t_str = (OVERRUN_GRACE_RATIO / ratio) ** 2
        else:
            t_str = 1.0
        stats["timely_clean"] = stats["timely_clean"] * cap_recency + w_cap * t_str
    else:
        stats["timely_clean"] = stats["timely_clean"] * cap_recency

    active = effective_active_seconds(row)
    if active is not None:
        y = math.log(max(1.0, float(active)))
        w_old = stats["duration_w"] * cap_recency
        m2_old = stats["duration_m2_log"] * cap_recency
        mu_old = stats["duration_mean_log"] if w_old > 0 else y
        w_new = w_old + u
        delta = y - mu_old
        mu_new = mu_old + (u / w_new) * delta
        m2_new = m2_old + u * delta * (y - mu_new)
        stats["duration_w"] = w_new
        stats["duration_mean_log"] = mu_new
        stats["duration_m2_log"] = max(0.0, m2_new)

    if row.get("tokens") is not None:
        yt = math.log(max(1.0, float(row["tokens"])))
        tw_old = stats["tokens_w"] * cap_recency
        tmu_old = stats["tokens_mean_log"] if tw_old > 0 else yt
        tw_new = tw_old + u
        stats["tokens_w"] = tw_new
        stats["tokens_mean_log"] = tmu_old + (u / tw_new) * (yt - tmu_old)

    shares = _extract_category_shares(row)
    if shares is not None:
        sw_old = stats.get("token_shares_w", 0.0) * cap_recency
        old_shares = stats.get("token_usage_shares")
        sw_new = sw_old + u
        if old_shares is None or sw_old <= 0:
            new_shares = list(shares)
        else:
            new_shares = [(sw_old * a + u * b) / sw_new for a, b in zip(old_shares, shares)]
        tot_s = math.fsum(new_shares)
        if tot_s > 0:
            new_shares = [v / tot_s for v in new_shares]
        stats["token_shares_w"] = sw_new
        stats["token_usage_shares"] = new_shares

    return stats


def _retrieval_signature(retrieval):
    from .retrieval import RetrievalPipeline

    if isinstance(retrieval, RetrievalPipeline):
        return (
            "pipeline",
            getattr(retrieval.embedder, "identity", type(retrieval.embedder).__qualname__),
            getattr(retrieval.lexical, "identity", type(retrieval.lexical).__qualname__),
            getattr(retrieval.dense, "identity", type(retrieval.dense).__qualname__),
            getattr(retrieval.reranker, "identity", type(retrieval.reranker).__qualname__),
        )
    return ("custom", f"{type(retrieval).__module__}.{type(retrieval).__qualname__}", id(retrieval))


def _search_prototypes(retrieval, candidates, canonical_task):
    """Match canonical_task against active prototypes of the same task_type."""
    from .retrieval import BM25Index, RetrievalPipeline

    if not candidates:
        return []
    if isinstance(retrieval, RetrievalPipeline) and isinstance(retrieval.lexical, BM25Index):
        if len(candidates) == 1 and candidates[0]["representative_task"] == canonical_task:
            return [(candidates[0], 10.0, 1.0)]

    docs = [
        {
            "key": "prototype:" + p["prototype_id"],
            "kind": "evidence",
            "value": {**p, "id": f"proto-{idx}"},
            "text": p["representative_task"],
        }
        for idx, p in enumerate(candidates)
    ]
    if isinstance(retrieval, RetrievalPipeline) and not isinstance(retrieval.lexical, BM25Index):
        lex = retrieval.lexical.search(docs, canonical_task, limit=min(200, len(docs)))
        den = retrieval.dense.search(
            docs, canonical_task, embedder=retrieval.embedder, limit=min(200, len(docs))
        )
        if not lex and not den:
            return []

    limit = min(5, len(docs))
    res = retrieval.search(docs, canonical_task, limit=limit)
    selected = res["records"]
    scores = {item["key"]: item["score"] for item in res.get("trace", {}).get("selected", [])}
    missing = [d for d in selected if scores.get(d["key"]) is None]
    if missing:
        try:
            values = list(retrieval.reranker.score(canonical_task, [d["text"] for d in missing]))
        except HowHardError:
            raise
        except Exception as exc:
            raise HowHardError(f"retrieval failed ({type(exc).__name__}): {str(exc)[:300]}") from exc
        if len(values) != len(missing) or any(not math.isfinite(float(v)) for v in values):
            raise HowHardError("prototype similarity: invalid reranker scores")
        scores.update({d["key"]: float(v) for d, v in zip(missing, values)})

    scored = [
        (d["value"], float(scores[d["key"]]))
        for d in selected
        if scores.get(d["key"]) is not None and float(scores[d["key"]]) > 0.0
    ]
    scored.sort(key=lambda item: (-item[1], item[0]["prototype_id"]))
    top = scored[:3]
    if not top:
        return []
    raw_weights = [_sigmoid(score) for _, score in top]
    total_w = math.fsum(raw_weights)
    return [(proto, score, w / total_w) for (proto, score), w in zip(top, raw_weights)]


def _make_prototype_id(task_type, risk_class, rep_task, rep_ctx, salt=""):
    return digest(["prototype-v1", task_type, risk_class, rep_task, rep_ctx, salt])


def _insert_prototype(db, task_type, risk_class, rep, *, parent_id=None, salt=""):
    now = datetime.now(timezone.utc).isoformat()
    rep_ctx_json = json.dumps(rep["representative_context"], sort_keys=True, separators=(",", ":"))
    pid = _make_prototype_id(task_type, risk_class, rep["representative_task"], rep["representative_context"], salt)
    while db.execute("SELECT 1 FROM prototypes WHERE prototype_id=?", (pid,)).fetchone() is not None:
        salt = salt + "+"
        pid = _make_prototype_id(
            task_type, risk_class, rep["representative_task"], rep["representative_context"], salt
        )
    db.execute(
        "INSERT INTO prototypes VALUES (?, ?, ?, ?, ?, 1, ?, 0, ?)",
        (pid, task_type, risk_class, rep["representative_task"], rep_ctx_json, parent_id, now),
    )
    return {
        "prototype_id": pid,
        "task_type": task_type,
        "risk_class": risk_class,
        "representative_task": rep["representative_task"],
        "representative_context": rep_ctx_json,
        "active": 1,
        "parent_id": parent_id,
        "member_count": 0,
        "created_at": now,
    }


def _recompute_prototype(db, proto_id, rows_by_id, row_order, config):
    """Recompute prototype_cells and member_count cleanly in chronological order from effective rows."""
    members = db.execute(
        "SELECT evidence_id, weight FROM prototype_members WHERE prototype_id=?",
        (proto_id,),
    ).fetchall()
    existing_cells = db.execute(
        "SELECT model, family, effort, prior_json FROM prototype_cells WHERE prototype_id=?",
        (proto_id,),
    ).fetchall()
    priors = {(m, e): (f, json.loads(p) if p is not None else None) for m, f, e, p in existing_cells}

    ordered_members = sorted(
        ((eid, float(w)) for eid, w in members if eid in rows_by_id),
        key=lambda item: row_order.get(item[0], 0),
    )
    non_interrupted = 0
    cell_stats = {}
    cell_families = {pair: fam for pair, (fam, _) in priors.items()}

    for eid, w in ordered_members:
        row = rows_by_id[eid]
        if row["outcome"] == "interrupted" or row["outcome"] not in OUTCOMES:
            continue
        non_interrupted += 1
        pair = (row["model"], row["resolved_effort"])
        cell_families[pair] = row["family"]
        st = cell_stats.setdefault(pair, empty_stats())
        prior_obj = priors.get(pair, (None, None))[1]
        _apply_member_to_stats(st, row, w, config, prior=prior_obj)

    now = datetime.now(timezone.utc).isoformat()
    all_pairs = set(priors) | set(cell_stats)
    db.execute("DELETE FROM prototype_cells WHERE prototype_id=?", (proto_id,))
    for pair in sorted(all_pairs):
        model, effort = pair
        family = cell_families[pair]
        prior_obj = priors.get(pair, (None, None))[1]
        st = cell_stats.get(pair, empty_stats())
        if prior_obj is None and st["run_count"] == 0:
            continue
        db.execute(
            "INSERT INTO prototype_cells VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                proto_id,
                model,
                family,
                effort,
                json.dumps(prior_obj, sort_keys=True, separators=(",", ":")) if prior_obj is not None else None,
                json.dumps(st, sort_keys=True, separators=(",", ":")),
                now,
            ),
        )
    db.execute("UPDATE prototypes SET member_count=? WHERE prototype_id=?", (non_interrupted, proto_id))
    return non_interrupted


def _lexical_tf_matrix(texts):
    vocab = {}
    tokenized = []
    for t in texts:
        toks = re.findall(r"\w+", t.casefold())
        tokenized.append(toks)
        for tok in toks:
            if tok not in vocab and len(vocab) < 512:
                vocab[tok] = len(vocab)
    if not vocab:
        return np.ones((len(texts), 1), dtype=np.float32)
    mat = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for idx, toks in enumerate(tokenized):
        for tok in toks:
            col = vocab.get(tok)
            if col is not None:
                mat[idx, col] += 1.0
        norm = float(np.linalg.norm(mat[idx]))
        if norm > 0:
            mat[idx] /= norm
        else:
            mat[idx, 0] = 1.0
    return mat


def _encode_unit_vectors(retrieval, texts):
    from .retrieval import RetrievalPipeline

    unique = list(dict.fromkeys(texts))
    if len(unique) <= 1:
        return np.ones((len(texts), 1), dtype=np.float32), False
    if isinstance(retrieval, RetrievalPipeline) and hasattr(retrieval, "embedder"):
        try:
            raw = np.asarray(retrieval.embedder.encode(unique), dtype=np.float32)
            if raw.ndim == 2 and raw.shape[0] == len(unique) and np.isfinite(raw).all():
                norms = np.linalg.norm(raw, axis=1, keepdims=True)
                if np.all(norms > 0):
                    unit = raw / norms
                    by_text = dict(zip(unique, unit))
                    return np.asarray([by_text[t] for t in texts], dtype=np.float32), True
        except Exception:
            pass
    return _lexical_tf_matrix(texts), False


def _two_means(features):
    """Deterministic 2-means clustering on unit/standardized feature vectors."""
    n = features.shape[0]
    if n < 2 * SPLIT_MIN_CHILD:
        return None
    # Deterministic farthest-point seeding: pick point farthest from centroid, then farthest from c0.
    mean_vec = features.mean(axis=0)
    idx0 = int(np.argmax(np.linalg.norm(features - mean_vec, axis=1)))
    c0 = features[idx0].copy()
    idx1 = int(np.argmax(np.linalg.norm(features - c0, axis=1)))
    if idx1 == idx0:
        return None
    c1 = features[idx1].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(15):
        d0 = np.linalg.norm(features - c0, axis=1)
        d1 = np.linalg.norm(features - c1, axis=1)
        new_labels = (d1 < d0).astype(int)
        if np.all(new_labels == 0) or np.all(new_labels == 1):
            return None
        c0_new = features[new_labels == 0].mean(axis=0)
        c1_new = features[new_labels == 1].mean(axis=0)
        n0, n1 = float(np.linalg.norm(c0_new)), float(np.linalg.norm(c1_new))
        if n0 > 0:
            c0 = c0_new / n0
        if n1 > 0:
            c1 = c1_new / n1
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    else:
        d0 = np.linalg.norm(features - c0, axis=1)
        d1 = np.linalg.norm(features - c1, axis=1)
        labels = (d1 < d0).astype(int)

    g0 = [i for i in range(n) if labels[i] == 0]
    g1 = [i for i in range(n) if labels[i] == 1]
    if len(g0) < SPLIT_MIN_CHILD or len(g1) < SPLIT_MIN_CHILD:
        return None
    return g0, g1


def _needs_split(cell_entries, distinct_tasks):
    if len(distinct_tasks) < 2:
        return False
    for entry in cell_entries.values():
        st = entry["stats"]
        if (
            st["run_count"] >= SPLIT_DURATION_MIN_RUNS
            and st["duration_w"] > 0
            and (st["duration_m2_log"] / st["duration_w"]) >= SPLIT_DURATION_DISPERSION
        ):
            return True
        w_tot = math.fsum(st["counts"])
        if (
            st["run_count"] >= SPLIT_OUTCOME_MIN_RUNS
            and w_tot > 0
            and min(st["counts"][0], st["counts"][1] + st["counts"][2]) / w_tot >= SPLIT_OUTCOME_MIX_RATIO
        ):
            return True
    return False


def _prepare_split_plan(proto, member_rows, cell_entries, retrieval):
    """Compute split sub-clusters outside the write lock when heterogeneity warrants it."""
    if len(member_rows) < SPLIT_MIN_MEMBERS:
        return None
    reps = [synthesize_representative(proto["task_type"], r) for r, _ in member_rows]
    task_texts = [rep["representative_task"] for rep in reps]
    distinct = set(task_texts)
    if len(distinct) < 2:
        return None

    empirical_heterogeneous = _needs_split(cell_entries, distinct)
    vecs, used_dense = _encode_unit_vectors(retrieval, task_texts)
    centroid = vecs.mean(axis=0)
    c_norm = float(np.linalg.norm(centroid))
    mean_cos_dist = (
        float(1.0 - (vecs @ (centroid / c_norm)).mean()) if c_norm > 0 else 0.0
    )
    semantic_heterogeneous = used_dense and mean_cos_dist >= SPLIT_COSINE_DISTANCE
    if not (empirical_heterogeneous or semantic_heterogeneous):
        return None

    partition = _two_means(vecs)
    if partition is None:
        return None
    g0, g1 = partition

    def pick_medoid(indices):
        sub = vecs[indices]
        sub_c = sub.mean(axis=0)
        norm = float(np.linalg.norm(sub_c))
        if norm > 0:
            sub_c = sub_c / norm
        sims = sub @ sub_c
        best_local = int(np.argmax(sims))
        return reps[indices[best_local]], [member_rows[i] for i in indices]

    rep0, members0 = pick_medoid(g0)
    rep1, members1 = pick_medoid(g1)
    if rep0["representative_task"] == rep1["representative_task"]:
        return None
    return (rep0, members0), (rep1, members1)


def _apply_split(db, parent, children_plan, rows_by_id, row_order, config):
    parent_id = parent["prototype_id"]
    parent_cells = _load_cells(db, [parent_id]).get(parent_id, {})
    db.execute("UPDATE prototypes SET active=0 WHERE prototype_id=?", (parent_id,))
    now = datetime.now(timezone.utc).isoformat()
    for idx, (child_rep, child_members) in enumerate(children_plan):
        child = _insert_prototype(
            db,
            parent["task_type"],
            parent["risk_class"],
            child_rep,
            parent_id=parent_id,
            salt=f"split:{parent_id}:{idx}",
        )
        cid = child["prototype_id"]
        # Seed parent cold-start priors as initial fallback so covered cells stay covered without re-querying LLM.
        for (model, effort), entry in parent_cells.items():
            if entry["prior"] is not None:
                db.execute(
                    "INSERT INTO prototype_cells VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        cid,
                        model,
                        entry["family"],
                        effort,
                        json.dumps(entry["prior"], sort_keys=True, separators=(",", ":")),
                        json.dumps(empty_stats(), sort_keys=True, separators=(",", ":")),
                        now,
                    ),
                )
        for row, w in child_members:
            db.execute(
                "INSERT OR REPLACE INTO prototype_members VALUES (?, ?, ?)",
                (row["id"], cid, float(w)),
            )
        _recompute_prototype(db, cid, rows_by_id, row_order, config)


def _cached_or_encode_vectors(engine, texts):
    """Encode unique texts in one batch, reusing/populating retrieval.sqlite3 vectors outside storage.locked."""
    from .retrieval import cache_db

    embedder = engine.retrieval.embedder
    unique = list(dict.fromkeys(texts))
    if not unique:
        return {}
    key_map = {t: digest([embedder.identity, t]) for t in unique}
    path = engine.config.state_root / "retrieval.sqlite3"
    admission = getattr(engine.retrieval.dense, "admission", None)
    stored = {}
    if path.is_file() and admission is not None:
        with cache_db(path, admission) as db:
            db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
            for t, k in key_map.items():
                row = db.execute("SELECT vector FROM vectors WHERE key=?", (k,)).fetchone()
                if row is not None and len(row[0]) == 4 * embedder.dimensions:
                    vec = np.frombuffer(row[0], dtype=np.float32).copy()
                    if np.isfinite(vec).all() and np.linalg.norm(vec) > 0:
                        stored[t] = vec
    missing = [t for t in unique if t not in stored]
    if missing:
        encoded = np.asarray(embedder.encode(missing), dtype=np.float32)
        if (
            encoded.shape != (len(missing), embedder.dimensions)
            or not np.isfinite(encoded).all()
            or not np.all(np.linalg.norm(encoded, axis=1) > 0)
        ):
            raise HowHardError("retrieval.embedder: invalid vectors")
        for t, vec in zip(missing, encoded):
            stored[t] = vec
        if admission is not None:
            with cache_db(path, admission) as db:
                db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
                db.executemany(
                    "INSERT OR REPLACE INTO vectors VALUES (?, ?)",
                    [(key_map[t], stored[t].tobytes()) for t in missing],
                )
    return {t: stored[t] / float(np.linalg.norm(stored[t])) for t in unique}


def _batch_cluster_plan(engine, eligible_rows):
    """Online vectorized cosine clustering for fast >32 historical backfill."""
    by_task = {}
    for row in eligible_rows:
        by_task.setdefault(row["task_type"], []).append(row)

    all_reps = {}
    all_texts = []
    for task_type, group in by_task.items():
        for row in group:
            rep = synthesize_representative(task_type, row)
            all_reps[row["id"]] = rep
            all_texts.append(rep["representative_task"])

    if all(len({all_reps[r["id"]]["representative_task"] for r in group}) <= 1 for group in by_task.values()):
        sim_w = float(_sigmoid(8.0))
        return [
            (task_type, group[0]["risk_class"], all_reps[group[0]["id"]], [(r, sim_w) for r in group])
            for task_type, group in by_task.items()
        ]

    unit_by_text = _cached_or_encode_vectors(engine, all_texts)
    clusters = []
    for task_type, group in by_task.items():
        vecs = np.asarray(
            [unit_by_text[all_reps[r["id"]]["representative_task"]] for r in group],
            dtype=np.float32,
        )
        centroids = []
        counts_per = []
        members = []
        for idx, vec in enumerate(vecs):
            if not centroids:
                centroids.append(vec.copy())
                counts_per.append(1)
                members.append([idx])
                continue
            mat = np.asarray(centroids, dtype=np.float32)
            sims = mat @ vec
            best = int(np.argmax(sims))
            if float(sims[best]) >= BACKFILL_COSINE_THRESHOLD or len(centroids) >= 256:
                members[best].append(idx)
                n_c = counts_per[best]
                updated = centroids[best] * n_c + vec
                norm = float(np.linalg.norm(updated))
                if norm > 0:
                    centroids[best] = updated / norm
                counts_per[best] = n_c + 1
            else:
                centroids.append(vec.copy())
                counts_per.append(1)
                members.append([idx])

        for centroid, idxs in zip(centroids, members):
            sub_vecs = vecs[idxs]
            sims = sub_vecs @ centroid
            medoid_local = int(np.argmax(sims))
            medoid_row = group[idxs[medoid_local]]
            rep = all_reps[medoid_row["id"]]
            member_pairs = [
                (group[i], float(_sigmoid((float(vecs[i] @ centroid) - 0.5) * 16.0)))
                for i in idxs
            ]
            clusters.append((task_type, medoid_row["risk_class"], rep, member_pairs))
    return clusters


def sync_observations(engine):
    """Synchronize prototype tables with engine.rows in O(1) when unchanged."""
    from .payload_store import _connect
    from .retrieval import RetrievalPipeline

    rows = engine.rows
    target_digest = journal_digest(rows)
    if getattr(engine, "_prototype_journal_digest", None) == target_digest:
        return target_digest

    current_meta = _read_meta(engine.config, "journal_digest")
    if current_meta == target_digest:
        engine._prototype_journal_digest = target_digest
        return target_digest
    if not rows and current_meta is None:
        return target_digest

    rows_by_id = {r["id"]: r for r in rows}
    row_order = {r["id"]: idx for idx, r in enumerate(rows)}
    row_digests = {r["id"]: digest(_row_signature(r)) for r in rows}

    # Inspect current DB state (read-only) to plan any heavy embedding/search work OUTSIDE storage.locked.
    db_ro = _connect(engine.config, write=False)
    existing_row_digests = {}
    proto_count = 0
    if db_ro is not None:
        try:
            if _has_tables(db_ro):
                proto_count = db_ro.execute("SELECT COUNT(*) FROM prototypes").fetchone()[0]
                existing_row_digests = dict(
                    db_ro.execute("SELECT evidence_id, row_digest FROM prototype_row_state").fetchall()
                )
        finally:
            db_ro.close()

    eligible_all = [r for r in rows if r["outcome"] in OUTCOMES]
    use_batch_backfill = (
        proto_count == 0
        and not existing_row_digests
        and len(eligible_all) > BACKFILL_BATCH_THRESHOLD
        and isinstance(engine.retrieval, RetrievalPipeline)
        and hasattr(engine.retrieval, "embedder")
    )

    batch_plan = _batch_cluster_plan(engine, eligible_all) if use_batch_backfill else None

    with locked(engine.config, operation="prototype sync"):
        db = _connect(engine.config, write=True)
        try:
            stored_meta = db.execute(
                "SELECT value FROM prototype_meta WHERE key='journal_digest'"
            ).fetchone()
            if stored_meta is not None and stored_meta[0] == target_digest:
                engine._prototype_journal_digest = target_digest
                return target_digest

            current_proto_count = db.execute("SELECT COUNT(*) FROM prototypes").fetchone()[0]
            current_state = dict(
                db.execute("SELECT evidence_id, row_digest FROM prototype_row_state").fetchall()
            )

            with db:
                if batch_plan is not None and current_proto_count == 0 and not current_state:
                    now_ts = datetime.now(timezone.utc).isoformat()
                    proto_inserts = []
                    member_inserts = []
                    cell_inserts = []
                    split_candidates = []
                    used_pids = set()
                    for c_idx, (task_type, risk_class, rep, member_pairs) in enumerate(batch_plan):
                        rep_ctx_json = json.dumps(rep["representative_context"], sort_keys=True, separators=(",", ":"))
                        pid = _make_prototype_id(
                            task_type, risk_class, rep["representative_task"], rep["representative_context"], f"bf:{c_idx}"
                        )
                        while pid in used_pids:
                            pid = _make_prototype_id(
                                task_type, risk_class, rep["representative_task"], rep["representative_context"], pid
                            )
                        used_pids.add(pid)
                        ordered_m = sorted(member_pairs, key=lambda it: row_order.get(it[0]["id"], 0))
                        cell_stats = {}
                        cell_fams = {}
                        n_mem = 0
                        for m_row, sim_w in ordered_m:
                            member_inserts.append((m_row["id"], pid, float(sim_w)))
                            if m_row["outcome"] in OUTCOMES:
                                n_mem += 1
                                pair = (m_row["model"], m_row["resolved_effort"])
                                cell_fams[pair] = m_row["family"]
                                _apply_member_to_stats(
                                    cell_stats.setdefault(pair, empty_stats()), m_row, sim_w, engine.config
                                )
                        proto_dict = {
                            "prototype_id": pid,
                            "task_type": task_type,
                            "risk_class": risk_class,
                            "representative_task": rep["representative_task"],
                            "representative_context": rep_ctx_json,
                            "active": 1,
                            "parent_id": None,
                            "member_count": n_mem,
                            "created_at": now_ts,
                        }
                        proto_inserts.append(
                            (pid, task_type, risk_class, rep["representative_task"], rep_ctx_json, 1, None, n_mem, now_ts)
                        )
                        for pair, st in cell_stats.items():
                            cell_inserts.append(
                                (
                                    pid,
                                    pair[0],
                                    cell_fams[pair],
                                    pair[1],
                                    None,
                                    json.dumps(st, sort_keys=True, separators=(",", ":")),
                                    now_ts,
                                )
                            )
                        if n_mem >= SPLIT_MIN_MEMBERS:
                            split_candidates.append((proto_dict, ordered_m))
                    db.executemany("INSERT INTO prototypes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", proto_inserts)
                    db.executemany("INSERT OR REPLACE INTO prototype_members VALUES (?, ?, ?)", member_inserts)
                    db.executemany("INSERT INTO prototype_cells VALUES (?, ?, ?, ?, ?, ?, ?)", cell_inserts)
                    db.executemany(
                        "INSERT OR REPLACE INTO prototype_row_state VALUES (?, ?, ?)",
                        [(r["id"], row_digests[r["id"]], idx) for idx, r in enumerate(rows)],
                    )
                    for proto_dict, ordered_m in split_candidates:
                        cells_map = _load_cells(db, [proto_dict["prototype_id"]]).get(proto_dict["prototype_id"], {})
                        plan = _prepare_split_plan(proto_dict, ordered_m, cells_map, engine.retrieval)
                        if plan is not None:
                            _apply_split(db, proto_dict, plan, rows_by_id, row_order, engine.config)
                else:
                    recompute_pids = set()
                    deleted_ids = set(current_state) - set(row_digests)
                    for eid in deleted_ids:
                        affected = [
                            r[0]
                            for r in db.execute(
                                "SELECT prototype_id FROM prototype_members WHERE evidence_id=?", (eid,)
                            ).fetchall()
                        ]
                        recompute_pids.update(affected)
                        db.execute("DELETE FROM prototype_members WHERE evidence_id=?", (eid,))
                        db.execute("DELETE FROM prototype_row_state WHERE evidence_id=?", (eid,))

                    # 1. Amendments to already-tracked rows:
                    for idx, r in enumerate(rows):
                        eid = r["id"]
                        if eid in current_state and current_state[eid] != row_digests[eid]:
                            affected = [
                                row_p[0]
                                for row_p in db.execute(
                                    "SELECT prototype_id FROM prototype_members WHERE evidence_id=?",
                                    (eid,),
                                ).fetchall()
                            ]
                            if not affected and r["outcome"] in OUTCOMES:
                                # Previously interrupted row was amended to a model outcome: attach now!
                                active_for_type = _load_active_prototypes(db, task_type=r["task_type"])
                                rep = synthesize_representative(r["task_type"], r)
                                hits = _search_prototypes(
                                    engine.retrieval, active_for_type, rep["representative_task"]
                                )
                                if hits:
                                    for proto, score, w in hits:
                                        db.execute(
                                            "INSERT OR REPLACE INTO prototype_members VALUES (?, ?, ?)",
                                            (eid, proto["prototype_id"], float(w * _sigmoid(score))),
                                        )
                                        affected.append(proto["prototype_id"])
                                else:
                                    proto = _insert_prototype(db, r["task_type"], r["risk_class"], rep)
                                    db.execute(
                                        "INSERT OR REPLACE INTO prototype_members VALUES (?, ?, ?)",
                                        (eid, proto["prototype_id"], float(_sigmoid(10.0))),
                                    )
                                    affected.append(proto["prototype_id"])
                            recompute_pids.update(affected)
                            db.execute(
                                "INSERT OR REPLACE INTO prototype_row_state VALUES (?, ?, ?)",
                                (eid, row_digests[eid], idx),
                            )

                    for pid in sorted(recompute_pids):
                        _recompute_prototype(db, pid, rows_by_id, row_order, engine.config)

                    # 2. Incremental appends for brand-new rows:
                    touched_for_split = set()
                    for idx, r in enumerate(rows):
                        eid = r["id"]
                        if eid in current_state:
                            continue
                        db.execute(
                            "INSERT OR REPLACE INTO prototype_row_state VALUES (?, ?, ?)",
                            (eid, row_digests[eid], idx),
                        )
                        if r["outcome"] not in OUTCOMES:
                            continue
                        active_for_type = _load_active_prototypes(db, task_type=r["task_type"])
                        rep = synthesize_representative(r["task_type"], r)
                        hits = _search_prototypes(
                            engine.retrieval, active_for_type, rep["representative_task"]
                        )
                        if hits:
                            targets = [(p, float(w * _sigmoid(score))) for p, score, w in hits]
                        else:
                            new_proto = _insert_prototype(db, r["task_type"], r["risk_class"], rep)
                            # Determine actual self-similarity score for custom test doubles like SimilarityDouble.
                            self_hits = _search_prototypes(
                                engine.retrieval, [new_proto], rep["representative_task"]
                            )
                            sim_w = float(_sigmoid(self_hits[0][1])) if self_hits else float(_sigmoid(10.0))
                            targets = [(new_proto, sim_w)]

                        now = datetime.now(timezone.utc).isoformat()
                        pair = (r["model"], r["resolved_effort"])
                        for proto, m_weight in targets:
                            pid = proto["prototype_id"]
                            db.execute(
                                "INSERT OR REPLACE INTO prototype_members VALUES (?, ?, ?)",
                                (eid, pid, m_weight),
                            )
                            existing_cell = db.execute(
                                "SELECT prior_json, stats_json FROM prototype_cells "
                                "WHERE prototype_id=? AND model=? AND effort=?",
                                (pid, pair[0], pair[1]),
                            ).fetchone()
                            if existing_cell is not None:
                                prior_obj = (
                                    json.loads(existing_cell[0])
                                    if existing_cell[0] is not None
                                    else None
                                )
                                st = json.loads(existing_cell[1])
                            else:
                                prior_obj = None
                                st = empty_stats()
                            _apply_member_to_stats(st, r, m_weight, engine.config, prior=prior_obj)
                            db.execute(
                                "INSERT OR REPLACE INTO prototype_cells VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (
                                    pid,
                                    pair[0],
                                    r["family"],
                                    pair[1],
                                    json.dumps(prior_obj, sort_keys=True, separators=(",", ":"))
                                    if prior_obj is not None
                                    else None,
                                    json.dumps(st, sort_keys=True, separators=(",", ":")),
                                    now,
                                ),
                            )
                            new_count = int(
                                db.execute(
                                    "SELECT member_count FROM prototypes WHERE prototype_id=?",
                                    (pid,),
                                ).fetchone()[0]
                            ) + 1
                            db.execute(
                                "UPDATE prototypes SET member_count=? WHERE prototype_id=?",
                                (new_count, pid),
                            )
                            if new_count >= SPLIT_MIN_MEMBERS:
                                touched_for_split.add(pid)

                    for pid in sorted(touched_for_split):
                        proto_row = db.execute(
                            "SELECT prototype_id, task_type, risk_class, representative_task, "
                            "representative_context, active, parent_id, member_count, created_at "
                            "FROM prototypes WHERE prototype_id=? AND active=1",
                            (pid,),
                        ).fetchone()
                        if proto_row is None:
                            continue
                        cols = (
                            "prototype_id",
                            "task_type",
                            "risk_class",
                            "representative_task",
                            "representative_context",
                            "active",
                            "parent_id",
                            "member_count",
                            "created_at",
                        )
                        proto_dict = dict(zip(cols, proto_row))
                        mem_rows_raw = db.execute(
                            "SELECT evidence_id, weight FROM prototype_members WHERE prototype_id=?",
                            (pid,),
                        ).fetchall()
                        mem_pairs = [
                            (rows_by_id[eid], float(w))
                            for eid, w in sorted(mem_rows_raw, key=lambda it: row_order.get(it[0], 0))
                            if eid in rows_by_id and rows_by_id[eid]["outcome"] in OUTCOMES
                        ]
                        cells_map = _load_cells(db, [pid]).get(pid, {})
                        plan = _prepare_split_plan(proto_dict, mem_pairs, cells_map, engine.retrieval)
                        if plan is not None:
                            _apply_split(db, proto_dict, plan, rows_by_id, row_order, engine.config)

                db.execute(
                    "INSERT OR REPLACE INTO prototype_meta VALUES ('journal_digest', ?)",
                    (target_digest,),
                )
        finally:
            db.close()

    engine._prototype_journal_digest = target_digest
    return target_digest


def match_context(engine, task_type, context):
    """Match (task_type, context) against active prototypes with bounded memoization."""
    from .payload_store import _connect

    rep = synthesize_representative(task_type, context)
    canonical_task = rep["representative_task"]
    j_digest = journal_digest(engine.rows)

    db = _connect(engine.config, write=False)
    if db is None:
        return [], rep
    try:
        if not _has_tables(db):
            return [], rep
        active = _load_active_prototypes(db, task_type=task_type)
        proto_sig = digest([(p["prototype_id"], p["member_count"]) for p in active])
    finally:
        db.close()

    if not active:
        return [], rep

    cache_key = (
        str(engine.config.state_root),
        j_digest,
        proto_sig,
        _retrieval_signature(engine.retrieval),
        task_type,
        canonical_task,
    )
    if cache_key in _MATCH_CACHE:
        return deepcopy(_MATCH_CACHE[cache_key]), rep

    hits = _search_prototypes(engine.retrieval, active, canonical_task)
    if len(_MATCH_CACHE) >= 256:
        _MATCH_CACHE.pop(next(iter(_MATCH_CACHE)))
    _MATCH_CACHE[cache_key] = deepcopy(hits)
    return hits, rep


def _synthesize_stage1_filter(cells, covered_info, policy):
    """Synthesize a valid within-model-ordinal filter from materialized prototype Stage 1 priors."""
    from .capability import EFFORT_ORDER, _forecast_band

    by_model = {}
    for c in cells:
        by_model.setdefault(c["model"], []).append(c)

    chains = {}
    raw_clear = {}
    raw_timely = {}
    empirical_only = set()
    for model, m_cells in by_model.items():
        ordered = sorted(
            m_cells,
            key=lambda c: (
                EFFORT_ORDER.get(c["effort"], 99),
                covered_info[c["key"]]["sort_score"],
                c["effort"],
            ),
        )
        chains[model] = [c["key"] for c in ordered]
        for c in ordered:
            info = covered_info[c["key"]]
            raw_clear[c["key"]] = info["prior_clear"]
            raw_timely[c["key"]] = info["prior_timely"]
            if info["empirical_only"]:
                empirical_only.add(c["key"])

    order = []
    while any(chains.values()):
        m = min(
            (name for name in chains if chains[name]),
            key=lambda name: (
                covered_info[chains[name][0]]["sort_score"],
                raw_clear[chains[name][0]],
                name,
            ),
        )
        order.append(chains[m].pop(0))

    # Assign neutral fit-band prior CDF across empirical-only or explicitly in-band cells.
    minimum, reliable = policy["min_clear_probability"], policy["reliability_probability"]
    fit_lo = max(minimum, reliable - 0.2)
    in_band_keys = [k for k in order if covered_info[k].get("in_band") or k in empirical_only]
    if in_band_keys:
        first_idx = order.index(in_band_keys[0])
        last_idx = min(order.index(in_band_keys[-1]), first_idx + policy["max_forecast_cells"] - 1)
        span_len = last_idx - first_idx + 1
        fit_vals = np.linspace(fit_lo, reliable, span_len).tolist() if span_len > 1 else [reliable]
        for offset, val in enumerate(fit_vals):
            k = order[first_idx + offset]
            raw_clear[k] = val
            raw_timely[k] = val
        return {"order": order, "lower": order[first_idx], "upper": order[last_idx]}

    # Project monotone along order via PAV.
    def pav(values_by_key):
        blocks = []
        for k in order:
            blocks.append(([k], float(values_by_key[k])))
            while len(blocks) > 1 and blocks[-2][1] > blocks[-1][1]:
                b, a = blocks.pop(), blocks.pop()
                merged_val = (len(a[0]) * a[1] + len(b[0]) * b[1]) / (len(a[0]) + len(b[0]))
                blocks.append((a[0] + b[0], merged_val))
        return {k: val for grp, val in blocks for k in grp}

    clear = pav(raw_clear)
    timely = pav({k: min(clear[k], raw_timely[k]) for k in order})

    if max(clear.values()) < minimum:
        return {"order": order, "lower": order[-1], "upper": order[-1]}

    band = _forecast_band(order, clear, timely, policy)
    lo_key = band[0]
    hi_key = band[min(len(band), policy["max_forecast_cells"]) - 1]
    if order.index(hi_key) - order.index(lo_key) + 1 > policy["max_forecast_cells"]:
        hi_key = order[order.index(lo_key) + policy["max_forecast_cells"] - 1]
    return {"order": order, "lower": lo_key, "upper": hi_key}


def prepare_stage1(reader, bundle):
    """Attach prototype match metadata to bundle and pre-materialize Stage 1 when all cells are covered."""
    from .payload_store import _connect

    payload = bundle["payload"]
    hits, fallback_rep = match_context(reader, payload["task_type"], payload["context"])
    if not hits:
        proto_ctx = prototype_context_for_baseline(payload["context"], fallback_rep)
        payload["prototype_context"] = proto_ctx
        bundle["prototype_state"] = {
            "mode": "miss",
            "matches": [],
            "primary_id": None,
            "representative": fallback_rep,
            "stage1_covered_priors": {},
        }
        return

    primary = hits[0][0]
    primary_rep = {
        "task_type": primary["task_type"],
        "representative_task": primary["representative_task"],
        "representative_context": json.loads(primary["representative_context"]),
    }
    payload["prototype_context"] = prototype_context_for_baseline(payload["context"], primary_rep)

    pids = [p["prototype_id"] for p, _, _ in hits]
    db = _connect(reader.config, write=False)
    try:
        cells_by_proto = _load_cells(db, pids) if db is not None else {}
    finally:
        if db is not None:
            db.close()

    covered_info = {}
    covered_priors = {}
    for cell in payload["cells"]:
        pair = (cell["model"], cell["effort"])
        cap_w = 0.0
        cap_clean = 0.0
        prior_w = 0.0
        prior_clear_sum = 0.0
        prior_timely_sum = 0.0
        rank_sum = 0.0
        in_band_w = 0.0
        for proto, _, w in hits:
            entry = cells_by_proto.get(proto["prototype_id"], {}).get(pair)
            if entry is None:
                continue
            st = entry["stats"]
            cap_w += w * float(st.get("capability_weight", 0.0))
            cap_clean += w * float(st.get("capability_clean", 0.0))
            pr = entry["prior"]
            if isinstance(pr, dict) and "P_clear" in pr:
                prior_w += w
                prior_clear_sum += w * float(pr["P_clear"])
                prior_timely_sum += w * float(pr.get("P_timely", pr["P_clear"]))
                rank_sum += w * float(pr.get("stage1_rank", pr["P_clear"]))
                if pr.get("stage1_in_band"):
                    in_band_w += w

        if prior_w > 0:
            p_clear = prior_clear_sum / prior_w
            p_timely = min(p_clear, prior_timely_sum / prior_w)
            in_band = (in_band_w / prior_w) >= 0.5
            covered_info[cell["key"]] = {
                "prior_clear": p_clear,
                "prior_timely": p_timely,
                "sort_score": (rank_sum / prior_w, p_clear),
                "empirical_only": False,
                "in_band": in_band,
            }
            covered_priors[cell["key"]] = p_clear
        elif cap_w >= STAGE1_MIN_CAP_WEIGHT:
            emp_rate = (cap_clean + 0.88) / (cap_w + 1.0)
            covered_info[cell["key"]] = {
                "prior_clear": emp_rate,
                "prior_timely": emp_rate,
                "sort_score": (emp_rate, emp_rate),
                "empirical_only": True,
            }

    bundle["prototype_state"] = {
        "mode": "hit",
        "matches": [
            {"prototype_id": p["prototype_id"], "score": float(s), "weight": float(w)}
            for p, s, w in hits
        ],
        "primary_id": primary["prototype_id"],
        "representative": primary_rep,
        "stage1_covered_priors": covered_priors,
    }

    if len(covered_info) == len(payload["cells"]):
        flt = _synthesize_stage1_filter(payload["cells"], covered_info, payload["filter_policy"])
        payload["no_memory_filter"] = flt
        payload["contract"]["filter"] = (
            "Omit this field. The isolated no_memory value is fixed and must not be changed."
        )


def record_stage1_baseline(engine, bundle, baseline_filter):
    """Persist newly evaluated Stage 1 cell priors onto the primary/new prototype."""
    from .capability import prior_cdf
    from .payload_store import _connect

    proto_state = bundle.get("prototype_state")
    if not proto_state:
        return
    payload = bundle["payload"]
    cdf = prior_cdf(baseline_filter, payload["filter_policy"])
    order = baseline_filter["order"]
    lo_idx = order.index(baseline_filter["lower"])
    hi_idx = order.index(baseline_filter["upper"])
    pos_map = {k: idx for idx, k in enumerate(order)}
    rank_map = {k: idx / max(1, len(order) - 1) for idx, k in enumerate(order)}
    already_covered = set(proto_state.get("stage1_covered_priors", {}))

    with locked(engine.config, operation="prototype stage1 prior"):
        db = _connect(engine.config, write=True)
        try:
            with db:
                pid = proto_state.get("primary_id")
                if pid is None or db.execute("SELECT 1 FROM prototypes WHERE prototype_id=?", (pid,)).fetchone() is None:
                    created = _insert_prototype(
                        db,
                        payload["task_type"],
                        payload["risk_class"],
                        proto_state["representative"],
                    )
                    pid = created["prototype_id"]
                    proto_state["primary_id"] = pid
                    proto_state["mode"] = "hit"
                    proto_state["matches"] = [{"prototype_id": pid, "score": 10.0, "weight": 1.0}]
                    _MATCH_CACHE.clear()

                now = datetime.now(timezone.utc).isoformat()
                for cell in payload["cells"]:
                    if cell["key"] in already_covered:
                        continue
                    row = db.execute(
                        "SELECT prior_json, stats_json FROM prototype_cells "
                        "WHERE prototype_id=? AND model=? AND effort=?",
                        (pid, cell["model"], cell["effort"]),
                    ).fetchone()
                    prior_obj = json.loads(row[0]) if row and row[0] is not None else {}
                    stats_obj = json.loads(row[1]) if row else empty_stats()
                    if "P_clear" not in prior_obj:
                        c_pos = pos_map[cell["key"]]
                        prior_obj["P_clear"] = float(cdf[cell["key"]])
                        prior_obj["P_timely"] = float(cdf[cell["key"]])
                        prior_obj["stage1_rank"] = float(rank_map[cell["key"]])
                        prior_obj["stage1_in_band"] = bool(lo_idx <= c_pos <= hi_idx)
                    db.execute(
                        "INSERT OR REPLACE INTO prototype_cells VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            pid,
                            cell["model"],
                            cell["family"],
                            cell["effort"],
                            json.dumps(prior_obj, sort_keys=True, separators=(",", ":")),
                            json.dumps(stats_obj, sort_keys=True, separators=(",", ":")),
                            now,
                        ),
                    )
        finally:
            db.close()


def _containing(interval, value):
    """A stored range, widened only by the rounding of a re-derived value it must still contain."""
    if interval[0] <= value <= interval[1]:
        return deepcopy(interval)
    return [min(float(interval[0]), value), max(float(interval[1]), value)]


def stored_forecasts(config, *, task_type=None):
    """The Stage 2 forecasts kept on active prototypes, each with its stated ranges and host.

    ``host`` is the name of the host whose request produced the forecast; ``None`` is the one
    unnamed host, which is also how forecasts stored before hosts were named read.
    """
    from .payload_store import _connect

    db = _connect(config, write=False)
    if db is None:
        return []
    try:
        if not _has_tables(db):
            return []
        prototypes = _load_active_prototypes(db, task_type=task_type)
        cells = _load_cells(db, [p["prototype_id"] for p in prototypes])
    finally:
        db.close()
    result = []
    for proto in prototypes:
        for (model, effort), entry in sorted(cells.get(proto["prototype_id"], {}).items()):
            prior = entry["prior"]
            if not isinstance(prior, dict) or not all(k in prior for k in OUTCOMES):
                continue
            result.append({"prototype_id": proto["prototype_id"], "task_type": proto["task_type"],
                           "model": model, "family": entry["family"], "effort": effort,
                           "host": prior.get("host"),
                           "forecast": {k: deepcopy(prior[k]) for k in FORECAST_FIELDS if k in prior}})
    return result


def _build_token_usage_from_shares(scalar_tokens, shares):
    counts_list = [max(0.0, float(scalar_tokens) * float(s)) for s in shares]
    diff = float(scalar_tokens) - math.fsum(counts_list)
    best_idx = int(np.argmax(counts_list))
    counts_list[best_idx] = max(0.0, counts_list[best_idx] + diff)
    return dict(zip(CATEGORIES, counts_list))


def prepare_stage2(engine, bundle):
    """Pre-materialize covered Stage 2 band cells and emit baseline_request only for unseen cells."""
    from .payload_store import _connect

    payload = bundle["payload"]
    proto_state = bundle.get("prototype_state")
    if not proto_state:
        return
    payload["prototype_context"] = prototype_context_for_baseline(
        payload["context"], proto_state["representative"]
    )
    matches = proto_state.get("matches", [])
    if not matches:
        return

    pids = [m["prototype_id"] for m in matches]
    db = _connect(engine.config, write=False)
    try:
        cells_by_proto = _load_cells(db, pids) if db is not None else {}
    finally:
        if db is not None:
            db.close()

    cand_by_pair = {(c["model"], c["effort"]): c for c in bundle["native_candidates"]}
    covered_rows = []
    unseen_cells = []

    for cell in payload["cells"]:
        pair = (cell["model"], cell["effort"])
        cand = cand_by_pair.get(pair, {})
        has_card = "rate_card" in cand
        cell_priced = (
            all(v is not None for v in cand["rate_card"]["rates"].values())
            if has_card
            else cand.get("usd_per_mtok") is not None
        )
        needs_category = cell.get("forecast_format") == "category-v1"

        w_emp = 0.0
        dur_w = 0.0
        dur_log_sum = 0.0
        tok_w = 0.0
        tok_log_sum = 0.0
        share_w = 0.0
        share_accum = [0.0] * len(CATEGORIES)

        prior_out_w = 0.0
        prior_probs = [0.0, 0.0, 0.0]
        prior_cis = {}
        prior_seconds_ci = None
        prior_token_ci = None
        prior_sec_w = 0.0
        prior_sec_sum = 0.0
        prior_tok_w = 0.0
        prior_tok_sum = 0.0
        prior_cat_w = 0.0
        prior_cat_accum = [0.0] * len(CATEGORIES)

        for m in matches:
            w = float(m["weight"])
            entry = cells_by_proto.get(m["prototype_id"], {}).get(pair)
            if entry is None:
                continue
            st = entry["stats"]
            w_emp += w * math.fsum(st.get("counts", (0.0, 0.0, 0.0)))
            if st.get("duration_w", 0.0) > 0:
                dw = w * float(st["duration_w"])
                dur_w += dw
                dur_log_sum += dw * float(st["duration_mean_log"])
            if st.get("tokens_w", 0.0) > 0:
                tw = w * float(st["tokens_w"])
                tok_w += tw
                tok_log_sum += tw * float(st["tokens_mean_log"])
            if st.get("token_usage_shares") is not None and st.get("token_shares_w", 0.0) > 0:
                sw = w * float(st["token_shares_w"])
                share_w += sw
                for idx, val in enumerate(st["token_usage_shares"]):
                    share_accum[idx] += sw * float(val)

            pr = entry["prior"]
            if isinstance(pr, dict):
                if all(k in pr for k in OUTCOMES):
                    prior_out_w += w
                    for idx, name in enumerate(OUTCOMES):
                        prior_probs[idx] += w * float(pr[name])
                    if len(matches) == 1:
                        for ci_key in ("loud_ci", "silent_ci"):
                            if ci_key in pr:
                                prior_cis[ci_key] = deepcopy(pr[ci_key])
                if "seconds" in pr:
                    prior_sec_w += w
                    prior_sec_sum += w * float(pr["seconds"])
                    if len(matches) == 1 and "seconds_ci" in pr:
                        prior_seconds_ci = pr["seconds_ci"]
                if "tokens" in pr:
                    prior_tok_w += w
                    prior_tok_sum += w * float(pr["tokens"])
                if "token_usage" in pr and isinstance(pr["token_usage"], dict):
                    prior_cat_w += w
                    for idx, cat in enumerate(CATEGORIES):
                        prior_cat_accum[idx] += w * float(pr["token_usage"][cat])
                    if len(matches) == 1 and "token_usage_ci" in pr:
                        prior_token_ci = pr["token_usage_ci"]

        outcome_ok = prior_out_w > 0 or w_emp >= STAGE2_MIN_EMP_WEIGHT
        sec_ok = prior_sec_w > 0 or dur_w > 0
        if needs_category:
            tok_ok = (prior_tok_w > 0 and prior_cat_w > 0) or (
                not cell_priced or (tok_w > 0 and share_w > 0)
            )
        else:
            tok_ok = prior_tok_w > 0 or (not cell_priced or tok_w > 0)

        if not (outcome_ok and sec_ok and tok_ok):
            unseen_cells.append(deepcopy(cell))
            continue

        if prior_out_w > 0:
            probs = [v / prior_out_w for v in prior_probs]
            p_tot = math.fsum(probs)
            probs = [v / p_tot for v in probs]
        else:
            probs = list(EMPIRICAL_BASE)

        seconds_val = (
            prior_sec_sum / prior_sec_w if prior_sec_w > 0 else math.exp(dur_log_sum / dur_w)
        )
        token_usage_ci_val = None
        if prior_tok_w > 0 and (not needs_category or prior_cat_w > 0):
            tokens_val = prior_tok_sum / prior_tok_w
            if needs_category:
                cat_vals = [v / prior_cat_w for v in prior_cat_accum]
                c_tot = math.fsum(cat_vals)
                shares = [v / c_tot for v in cat_vals] if c_tot > 0 else [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                token_usage_val = _build_token_usage_from_shares(tokens_val, shares)
                # Like the rate ranges, a stated range is returned only with the one forecast it belongs to.
                if prior_token_ci is not None:
                    token_usage_ci_val = {kind: _containing(interval, token_usage_val[kind])
                                          for kind, interval in prior_token_ci.items()}
            else:
                token_usage_val = None
        elif tok_w > 0:
            tokens_val = float(max(1, round(math.exp(tok_log_sum / tok_w))))
            if needs_category:
                shares = [v / share_w for v in share_accum]
                s_tot = math.fsum(shares)
                shares = [v / s_tot for v in shares] if s_tot > 0 else [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                token_usage_val = _build_token_usage_from_shares(tokens_val, shares)
            else:
                token_usage_val = None
        else:
            tokens_val = 0.0
            token_usage_val = _build_token_usage_from_shares(0.0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) if needs_category else None

        row = {
            "family": cell["family"],
            "effort": cell["effort"],
            "clean": probs[0],
            "loud": probs[1],
            "silent": probs[2],
            "tokens": tokens_val,
            "seconds": seconds_val,
            **prior_cis,
        }
        if prior_seconds_ci is not None and prior_sec_w > 0:
            row["seconds_ci"] = _containing(prior_seconds_ci, seconds_val)
        if token_usage_val is not None:
            row["token_usage"] = token_usage_val
        if token_usage_ci_val is not None:
            row["token_usage_ci"] = token_usage_ci_val
        covered_rows.append(row)

    if not unseen_cells and covered_rows:
        payload["no_memory_forecasts"] = covered_rows
        payload["contract"]["forecasts"] = (
            "Omit this field. The isolated no_memory value is fixed and must not be changed."
        )
    elif covered_rows and unseen_cells:
        payload["preset_forecasts"] = covered_rows
        payload["unseen_cells"] = unseen_cells


def record_stage2_baseline(engine, bundle, new_rows):
    """Persist newly evaluated Stage 2 cell priors onto the primary prototype."""
    from .payload_store import _connect

    if not new_rows:
        return
    proto_state = bundle.get("prototype_state")
    if not proto_state:
        return
    payload = bundle["payload"]
    cell_lookup = {(c["family"], c["effort"]): c for c in payload["cells"]}

    with locked(engine.config, operation="prototype stage2 prior"):
        db = _connect(engine.config, write=True)
        try:
            with db:
                pid = proto_state.get("primary_id")
                if pid is None or db.execute("SELECT 1 FROM prototypes WHERE prototype_id=?", (pid,)).fetchone() is None:
                    created = _insert_prototype(
                        db,
                        payload["task_type"],
                        payload["risk_class"],
                        proto_state["representative"],
                    )
                    pid = created["prototype_id"]
                    proto_state["primary_id"] = pid
                    proto_state["mode"] = "hit"
                    proto_state["matches"] = [{"prototype_id": pid, "score": 10.0, "weight": 1.0}]
                    _MATCH_CACHE.clear()

                now = datetime.now(timezone.utc).isoformat()
                for f_row in new_rows:
                    cell = cell_lookup.get((f_row["family"], f_row["effort"]))
                    if cell is None:
                        continue
                    existing = db.execute(
                        "SELECT prior_json, stats_json FROM prototype_cells "
                        "WHERE prototype_id=? AND model=? AND effort=?",
                        (pid, cell["model"], cell["effort"]),
                    ).fetchone()
                    prior_obj = json.loads(existing[0]) if existing and existing[0] is not None else {}
                    stats_obj = json.loads(existing[1]) if existing else empty_stats()
                    # A range on active time or a token kind belongs to the forecast that stated it.
                    for field in ("seconds_ci", "token_usage_ci"):
                        prior_obj.pop(field, None)
                    for field in FORECAST_FIELDS:
                        if field in f_row:
                            prior_obj[field] = deepcopy(f_row[field])
                    prior_obj["host"] = bundle.get("host")
                    db.execute(
                        "INSERT OR REPLACE INTO prototype_cells VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            pid,
                            cell["model"],
                            cell["family"],
                            cell["effort"],
                            json.dumps(prior_obj, sort_keys=True, separators=(",", ":")),
                            json.dumps(stats_obj, sort_keys=True, separators=(",", ":")),
                            now,
                        ),
                    )
        finally:
            db.close()


def matched_members(engine, proto_state):
    """The IDs of the recorded runs in at least one matched prototype: the evidence candidates.

    Prototypes only choose the candidates. Their scores and membership weights are not read.
    """
    from .payload_store import _connect

    if not proto_state or not proto_state.get("matches"):
        return set()
    pids = [m["prototype_id"] for m in proto_state["matches"]]
    db = _connect(engine.config, write=False)
    if db is None:
        return set()
    try:
        if not _has_tables(db):
            return set()
        placeholders = ",".join("?" for _ in pids)
        rows = db.execute(
            f"SELECT DISTINCT evidence_id FROM prototype_members WHERE prototype_id IN ({placeholders})",
            pids,
        ).fetchall()
    finally:
        db.close()
    return {row[0] for row in rows}
