"""Public application API: fit once, inspect summaries and make many seeded picks."""

from pathlib import Path
import secrets
import time
from datetime import datetime, timezone

from . import cache, storage
from .config import Config, load_config
from .outcomes import fit
from .prior import build_prior
from .regression import fit_regression
from .selector import choose, summarize
from .transfer import export_prior, import_prior, read_export
from . import decisions
from .context import FORMAT, canonical, capture_project, validate_text
from .context import digest
from .validation import HowHardError
from .memory import MemoryStore
from .llm import resolve
from . import semantic


class Engine:
    def __init__(self, config=None, *, directory=None, state_directory=None, cache_path=None, journal_path=None, llm=None):
        self.config = config if isinstance(config, Config) else load_config(config, directory=directory, state_directory=state_directory)
        if isinstance(config, Config) and (directory is not None or state_directory is not None):
            from copy import deepcopy
            from .state import roots
            self.config = deepcopy(config)
            project, store, identity_path, identity = roots(directory or config.project_root, explicit=config.explicit,
                                              state_directory=state_directory or config.state_root)
            self.config.project_root, self.config.state_root, self.config.identity_path, self.config.project_identity = project, store, identity_path, identity
        if cache_path is not None:
            self.config.data["cache"] = str(Path(cache_path).resolve())
        if journal_path is not None:
            self.config.data["journal"] = str(Path(journal_path).resolve())
        self.config.validate_paths()
        self.llm, self.backend = resolve(llm)
        self.last_memory_update = {"status": "not-requested"}
        self._summaries = {}
        self.refresh()
        self.memory = MemoryStore(self.config, evidence=lambda: {r["id"] for r in self.raw_rows})

    def refresh(self, *, full=False):
        config = self.config
        with storage.locked(config):
            from . import native, state
            state.check(config)
            native.apply(config, native.read(config))
            raw_rows, blobs = storage.snapshot(config)
            state.claim(config)
        self.raw_rows = raw_rows
        self.rows = storage.effective_rows(config, raw_rows)
        imported, imported_bytes = None, b""
        if "import" in config.data["prior"]:
            path = config.path(config.data["prior"]["import"])
            if path.exists():
                imported = read_export(path, config)
                imported_bytes = path.read_bytes()
        extra_types = {t: config.task_types.get(t, "standard") for t in imported["vocabularies"]["task_types"]} if imported else None
        self.prior = build_prior(config, self.rows, extra_types=extra_types, imported=imported)
        prior_hash, data_hash = cache.hashes(config, blobs, imported_bytes)
        self.fit_key = [prior_hash, data_hash]
        path = config.path(config.data["cache"])
        cached = None if full else cache.load(path, self.prior)
        start = None
        if cached is not None and cached[1] == prior_hash:
            previous = cached[0]
            exact_labels = all(getattr(previous, b).labels == getattr(self.prior, b).labels for b in ("outcome", "tokens", "time"))
            if cached[2] == data_hash and exact_labels:
                previous.outcome.statistics = {"kind": "cached", "iterations": 0, "final_step": 0.0}
                self.posterior = previous
                self._summaries.clear()
                return previous
            start = previous.outcome
        self.posterior = cache.Posterior(fit(self.prior, self.rows, start),
            fit_regression(self.prior, self.rows, "tokens"), fit_regression(self.prior, self.rows, "time"), self.prior)
        cache.save(path, self.posterior, prior_hash, data_hash)
        self._summaries.clear()
        return self.posterior

    @property
    def paths(self):
        return {"project_root": str(self.config.project_root), "state_root": str(self.config.state_root)}

    def evidence(self, evidence_id, *, raw=False):
        from copy import deepcopy
        if raw:
            row = next((r for r in self.raw_rows if r["id"] == evidence_id), None)
            if row is None:
                raise HowHardError(f"evidence: unknown id {evidence_id}")
            return deepcopy(row)
        return semantic.tool(self, {"tool": "evidence.read", "id": evidence_id}, allow_project=False)

    def _register(self, native_candidates):
        from .native import register
        previous = self.config.native_models
        inventory = register(self.config, native_candidates)
        if previous != self.config.native_models:
            self.refresh()
        return inventory

    def _prepared(self, task_type, risk, task_text="", project_context=None, native_candidates=None):
        context = {"encoder": FORMAT, "task_text": validate_text(task_text), "project_context": capture_project(self.config, project_context)}
        key = task_type, risk, canonical(context), canonical(native_candidates)
        if key not in self._summaries:
            self._summaries[key] = summarize(self.posterior, self.config, task_type, risk, context, native_candidates=native_candidates)
        return self._summaries[key]

    def summary(self, task_type, task_text="", *, project_context=None, risk=None, native_candidates=None):
        from copy import deepcopy
        inventory = self._register(native_candidates)
        prepared = self._semantic_prepared(self._prepared(task_type, risk, task_text, project_context, inventory))
        return deepcopy(prepared[0])

    def pick(self, task_type, task_text="", *, project_context=None, risk=None, seed=None, native_candidates=None):
        from .native import preflight
        native_candidates = preflight(self.config, native_candidates)
        started = time.monotonic()
        try:
            result = self._pick(task_type, task_text, project_context=project_context, risk=risk, seed=seed, native_candidates=native_candidates)
        except Exception as exc:
            self._decision_event(None, task_type, started, error=type(exc).__name__)
            raise
        self._decision_event(result, task_type, started)
        return result

    def _decision_event(self, result, task_type, started, error=None):
        event = {"event_id": secrets.token_hex(12), "at_utc": datetime.now(timezone.utc).isoformat(),
                 "task_type": task_type if isinstance(task_type, str) else type(task_type).__name__, "status": "failed" if error else "completed", "error": error,
                 "seconds": time.monotonic() - started, "action_probability": None,
                 "decision_id": result["decision_id"] if result else None,
                 "backend": result.get("backend", self.backend) if result else self.backend}
        if result and "semantic" in result:
            result["routing_overhead"]["end_to_end_seconds"] = event["seconds"]
        with storage.locked(self.config):
            storage.append_bytes(self.config.state_root / "decision-events.jsonl", canonical(event) + b"\n")

    def _pick(self, task_type, task_text="", *, project_context=None, risk=None, seed=None, native_candidates=None):
        if seed is not None and (type(seed) is not int or seed < 0):
            raise HowHardError("seed: expected a non-negative integer")
        inventory = self._register(native_candidates)
        prepared = self._semantic_prepared(self._prepared(task_type, risk, task_text, project_context, inventory))
        parts = None
        if "semantic" in prepared[0]:
            seed = secrets.randbits(64) if seed is None else seed
            if type(seed) is not int or seed < 0:
                raise HowHardError("seed: expected a non-negative integer")
            parts = semantic.sample(prepared, seed, self.posterior.prior.noise)
        result = choose(prepared, self.posterior, self.config, seed, parts)
        result["decision_id"] = decisions.save(self.config, result)
        return result

    def _semantic_prepared(self, prepared, response=None, payload=None, snapshot=None):
        if self.llm is None and response is None:
            if self.backend["mode"] == "auto":
                from copy import deepcopy
                result = deepcopy(prepared[0])
                result["backend"] = {"mode": "statistical-fallback", "reason": self.backend.get("reason", "No LLM adapter is available")}
                return result, prepared[1], prepared[2]
            return prepared
        if payload is None:
            payload, snapshot = semantic.request(self, prepared)
        if response is None:
            try:
                response, payload, usage = semantic.complete(self, payload, snapshot)
            except HowHardError as exc:
                if self.backend["mode"] != "auto":
                    raise
                from copy import deepcopy
                result = deepcopy(prepared[0])
                result["backend"] = {"mode": "statistical-fallback", "reason": str(exc)}
                result["routing_overhead"] = deepcopy(getattr(self, "last_inference_usage", {"cost_usd": None, "seconds": None}))
                return result, prepared[1], prepared[2]
        else:
            response = semantic.validate_response(response, payload, snapshot)
            semantic.external_submission(self, payload)
            usage = {"backend": "calling-agent", "seconds": 0.0, "cost_usd": None, "calls": [], "cached": False, "call_seconds": 0.0, "call_cost_usd": None}
        return semantic.augment(self, prepared, response, payload, snapshot, usage)

    def memory_beliefs(self):
        return semantic.beliefs(self)

    def memory_learn(self, evidence_id):
        self.last_memory_update = semantic.learn(self, evidence_id)
        return self.last_memory_update

    def prepare(self, task_type, task_text="", *, project_context=None, risk=None, native_candidates=None):
        inventory = self._register(native_candidates)
        prepared = self._prepared(task_type, risk, task_text, project_context, inventory)
        payload, snapshot = semantic.request(self, prepared)
        bundle = {"format": "howhard-agent-request-v2", "fit_key": self.fit_key, "payload": payload, "snapshot": snapshot}
        bundle["native_candidates"] = inventory
        bundle["project_root"] = str(self.config.project_root)
        return self._save_request(bundle)

    def _save_request(self, bundle):
        from .transfer import atomic_write
        request_id = digest(bundle)
        atomic_write(self.config.state_root / "requests" / f"{request_id}.json", canonical(bundle) + b"\n")
        return {"request_id": request_id, "request": bundle["payload"], "baseline_request": semantic.baseline_request(bundle["payload"])}

    def _load_request(self, request_id):
        import json
        import re
        if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{64}", request_id):
            raise HowHardError("request_id: expected SHA-256 request ID")
        try:
            bundle = json.loads((self.config.state_root / "requests" / f"{request_id}.json").read_bytes())
        except (OSError, ValueError) as exc:
            raise HowHardError("request_id: unavailable or invalid agent request") from exc
        from .validation import keys
        keys(bundle, ["format", "fit_key", "payload", "snapshot", "native_candidates", "project_root"], where="request")
        if digest(bundle) != request_id or bundle["format"] != "howhard-agent-request-v2":
            raise HowHardError("request_id: request integrity or format mismatch")
        if bundle["project_root"] != str(self.config.project_root):
            raise HowHardError("request_id: belongs to another worktree; prepare in the current project")
        if bundle["fit_key"] != self.fit_key or bundle["snapshot"]["revision"] != self.memory.snapshot()["revision"]:
            raise HowHardError("request_id: stale evidence or memory; prepare a new decision")
        return bundle

    def retrieve(self, request_id, queries):
        if not isinstance(queries, list) or not 1 <= len(queries) <= 3:
            raise HowHardError("retrieval: provide between one and three tool requests")
        bundle = self._load_request(request_id)
        if len(bundle["payload"]["retrieval"]) + len(queries) > 3 * semantic.SETTINGS["retrieval_rounds"]:
            raise HowHardError("retrieval: request budget exceeded")
        for query in queries:
            result = semantic.tool(self, query, project_snapshot=bundle["payload"]["context"]["project_context"])
            bundle["payload"]["retrieval"].append({"request": query, "result": result})
        return self._save_request(bundle)

    def decide(self, request_id, response, *, seed=None):
        started = time.monotonic()
        try:
            result = self._decide(request_id, response, seed=seed)
        except Exception as exc:
            self._decision_event(None, "calling-agent", started, error=type(exc).__name__)
            raise
        self._decision_event(result, result["task_type"], started)
        return result

    def _decide(self, request_id, response, *, seed=None):
        if seed is not None and (type(seed) is not int or seed < 0):
            raise HowHardError("seed: expected non-negative integer")
        bundle = self._load_request(request_id)
        payload = bundle["payload"]
        prepared = self._prepared(payload["task_type"], payload["risk_class"], payload["context"]["task_text"], payload["context"]["project_context"], bundle["native_candidates"])
        prepared = self._semantic_prepared(prepared, response, payload, bundle["snapshot"])
        seed = secrets.randbits(64) if seed is None else seed
        if type(seed) is not int or seed < 0:
            raise HowHardError("seed: expected non-negative integer")
        parts = semantic.sample(prepared, seed, self.posterior.prior.noise)
        result = choose(prepared, self.posterior, self.config, seed, parts)
        result["decision_id"] = decisions.save(self.config, result)
        return result

    def memory_learning_context(self, evidence_id):
        request, event_key = semantic.learning_request(self, evidence_id)
        return {"request": request, "event_key": event_key, "memory_revision": request["memory_revision"]}

    def apply_memory_learning(self, evidence_id, response, *, expected_revision, event_key):
        self.last_memory_update = semantic.apply_learning(self, evidence_id, response, expected_revision=expected_revision, event_key=event_key)
        return self.last_memory_update

    def observe(self, task_type, model, effort, outcome, *, decision=None, task_text=None, project_context=None, native=False, **options):
        automatic_context = []
        if decision is not None:
            if native:
                raise HowHardError("native: actual execution admission cannot attach an unexecuted decision")
            receipt = decisions.load(self.config, decision)
            if task_text is not None or project_context is not None or options.get("amends") is not None:
                raise HowHardError("decision: cannot override receipt context or combine with an amendment")
            if (task_type, model, effort) != (receipt["task_type"], receipt["chosen"]["model"], receipt["chosen"]["effort"]):
                raise HowHardError("decision: task type, model and effort must match the receipt")
            options.update(task_text=receipt["context"]["task_text"], project_context=receipt["context"]["project_context"],
                           risk_class=receipt["risk_class"], decision_id=decisions.digest(receipt))
        elif not options.get("amends"):
            if task_text is None:
                automatic_context.append("task_text")
            if project_context is None:
                automatic_context.extend(("project_context", "context_encoder"))
            options.update(task_text="" if task_text is None else task_text, project_context=capture_project(self.config, project_context))
        elif task_text is not None or project_context is not None:
            raise HowHardError("amends: historical context belongs to the original run")
        row = storage.observe(self.config, task_type, model, effort, outcome, _automatic_context=automatic_context, native=native, **options)
        self.refresh()
        try:
            self.last_memory_update = semantic.learn(self, row["id"])
        except HowHardError as exc:
            # The observation is durable. Retrying the writer must not duplicate it.
            self.last_memory_update = {"status": "failed", "error": str(exc), "evidence_id": row["id"]}
        return row

    def observe_decision(self, decision, outcome, **options):
        receipt = decisions.load(self.config, decision)
        return self.observe(receipt["task_type"], receipt["chosen"]["model"], receipt["chosen"]["effort"], outcome, decision=receipt, **options)

    def save_decision(self, decision, path):
        receipt = decisions.load(self.config, decision)
        from .transfer import atomic_write
        atomic_write(path, canonical(receipt) + b"\n")

    def flush(self):
        ids = storage.flush(self.config)
        self.refresh()
        return ids

    def export_prior(self, path):
        return export_prior(self.posterior, self.config, path)

    def import_prior(self, path):
        destination = import_prior(self.config, path)
        self.refresh()
        return destination
