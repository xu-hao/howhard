"""Public application API: prepare LLM requests, retain evidence and select native cells."""

from pathlib import Path
import secrets
import time
from datetime import datetime, timezone

from . import storage
from .config import Config, load_config
from .selector import choose, prepare
from . import decisions
from .context import FORMAT, canonical, capture_project, validate_text
from .context import digest
from .validation import HowHardError
from .memory import MemoryStore
from .llm import LLMError, resolve
from . import semantic


class Engine:
    def __init__(self, config=None, *, directory=None, state_directory=None, journal_path=None, llm=None, retrieval=None,
                 write_timeout=30.0, write_cancel=None):
        if llm is not None and not callable(getattr(llm, "complete", None)):
            raise LLMError("llm: expected an adapter with complete(request)")
        self.config = config if isinstance(config, Config) else load_config(config, directory=directory, state_directory=state_directory)
        from .writes import WritePolicy
        from copy import copy, deepcopy
        self.config = copy(self.config)
        self.config.data = deepcopy(self.config.data)
        self.config.native_models = deepcopy(self.config.native_models)
        self.config.write_policy = WritePolicy(write_timeout, write_cancel)
        if isinstance(config, Config) and (directory is not None or state_directory is not None):
            from .state import roots
            project, store, identity_path, identity = roots(directory or config.project_root, explicit=config.explicit,
                                              state_directory=state_directory or config.state_root)
            self.config.project_root, self.config.state_root, self.config.identity_path, self.config.project_identity = project, store, identity_path, identity
        if journal_path is not None:
            self.config.data["journal"] = str(Path(journal_path).resolve())
        self.config.validate_paths()
        self.llm = llm
        self.backend = {"mode": "explicit", "backend": str(getattr(llm, "identity", type(llm).__name__))} if llm is not None else {"mode": "calling-agent"}
        self.last_memory_update = {"status": "not-requested"}
        self._summaries = {}
        self.refresh()
        from .retrieval import RetrievalPipeline
        self.retrieval = retrieval if retrieval is not None else RetrievalPipeline.from_config(self.config)
        self.memory = MemoryStore(self.config, evidence=lambda: {r["id"] for r in self.raw_rows}, retrieval=self.retrieval)

    def refresh(self):
        config = self.config
        claimed = (config.state_root / "store.json").exists()
        boundary = storage.reading(config) if claimed else storage.locked(config, operation="store initialize")
        with boundary:
            from . import native, state
            state.check(config)
            native.apply(config, native.read(config))
            raw_rows, _ = storage.snapshot(config)
            if not claimed:
                state.claim(config)
        self.raw_rows = raw_rows
        self.rows = storage.effective_rows(config, raw_rows)
        # Historical identity admission does not change forecasts for a disjoint
        # inventory. Its actual cells are bound separately in every request.
        configured = {k: v for k, v in config.data["models"].items() if not v.get("native")}
        evidence_key = digest({"config": {**config.data, "models": configured}, "rows": self.raw_rows})
        if evidence_key != getattr(self, "evidence_key", None):
            self._summaries.clear()
        self.evidence_key = evidence_key

    def _inference(self):
        if self.llm is None:
            self.llm, self.backend = resolve()
        return self.llm

    @property
    def paths(self):
        return {"project_root": str(self.config.project_root), "state_root": str(self.config.state_root)}

    def write_status(self):
        from .writes import status
        return status(self.config.state_root)

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
        inventory = register(self.config, native_candidates)
        self.refresh()
        return inventory

    def _prepared(self, task_type, risk, task_text="", project_context=None, native_candidates=None):
        context = {"encoder": FORMAT, "task_text": validate_text(task_text), "project_context": capture_project(self.config, project_context)}
        key = task_type, risk, canonical(context), canonical(native_candidates)
        if key not in self._summaries:
            self._summaries[key] = prepare(self.config, task_type, risk, context, native_candidates=native_candidates)
        return self._summaries[key]

    def summary(self, task_type, task_text="", *, project_context=None, risk=None, native_candidates=None, inventory=None):
        from .native import resolve
        candidates = resolve(self.config, native_candidates, inventory)
        try:
            return self._drive(task_type, task_text, project_context=project_context, risk=risk,
                               native_candidates=candidates, select=False)
        except HowHardError as exc:
            self._refresh_stale(exc)
            raise

    def pick(self, task_type, task_text="", *, project_context=None, risk=None, seed=None, native_candidates=None, inventory=None):
        from .native import resolve
        native_candidates = resolve(self.config, native_candidates, inventory)
        started = time.monotonic()
        try:
            result = self._pick(task_type, task_text, project_context=project_context, risk=risk, seed=seed, native_candidates=native_candidates)
        except Exception as exc:
            self._refresh_stale(exc)
            self._failed_decision_event(exc, task_type, started)
            raise
        self._decision_event(result, task_type, started)
        return result

    def _failed_decision_event(self, error, task_type, started):
        from .writes import WriteError
        from .request_state import StaleRequest
        if isinstance(error, WriteError):
            return
        try:
            self._decision_event(None, task_type, started, error=type(error).__name__)
        except WriteError as audit:
            if not isinstance(error, StaleRequest):
                raise
            error.audit_error = audit.as_dict()

    def _decision_event(self, result, task_type, started, error=None):
        event = {"event_id": secrets.token_hex(12), "at_utc": datetime.now(timezone.utc).isoformat(),
                 "task_type": task_type if isinstance(task_type, str) else type(task_type).__name__, "status": "failed" if error else "completed", "error": error,
                 "seconds": time.monotonic() - started, "action_probability": None,
                 "decision_id": result["decision_id"] if result else None,
                 "backend": result.get("backend", self.backend) if result else self.backend}
        if result and "semantic" in result:
            result["routing_overhead"]["end_to_end_seconds"] = event["seconds"]
        with storage.locked(self.config, operation="decision event"):
            storage.append_bytes(self.config.state_root / "decision-events.jsonl", canonical(event) + b"\n")

    def _pick(self, task_type, task_text="", *, project_context=None, risk=None, seed=None, native_candidates=None):
        if seed is not None and (type(seed) is not int or seed < 0):
            raise HowHardError("seed: expected a non-negative integer")
        return self._drive(task_type, task_text, project_context=project_context, risk=risk,
                           seed=seed, native_candidates=native_candidates)

    def _drive(self, task_type, task_text, *, project_context=None, risk=None, seed=None,
               native_candidates=None, select=True):
        pending = self.prepare(task_type, task_text, project_context=project_context, risk=risk,
                               native_candidates=native_candidates)
        self._inference()
        while "request_id" in pending:
            from .request_state import view, StaleRequest
            try:
                bundle = self._load_request(pending["request_id"])
                reader = view(self, bundle["read_state"]) if "read_state" in bundle else self
                response, payload, usage = semantic.complete(reader, bundle["payload"], bundle["snapshot"])
                bundle["payload"] = payload
                pending = self._advance(bundle, response, seed=seed, usage=usage, select=select)
            except StaleRequest as exc:
                exc.request_id = pending["request_id"]
                raise
        return pending

    def _validate_state(self, evidence_key, revision):
        self.refresh()
        if evidence_key != self.evidence_key or revision != self.memory.snapshot()["revision"]:
            raise HowHardError("request_id: stale evidence or memory; prepare a new decision")

    def memory_beliefs(self):
        return semantic.beliefs(self)

    def memory_learn(self, evidence_id):
        self.refresh()
        self._inference()
        self.last_memory_update = semantic.learn(self, evidence_id)
        return self.last_memory_update

    def prepare(self, task_type, task_text="", *, project_context=None, risk=None, native_candidates=None, inventory=None):
        from .native import resolve
        from .request_state import capture, configuration
        candidates = self._register(resolve(self.config, native_candidates, inventory))
        reader, read_state = capture(self)
        prepared = reader._prepared(task_type, risk, task_text, project_context, candidates)
        payload, snapshot = semantic.request(reader, prepared)
        payload["weights_policy"] = "snapshot"
        self.last_retrieval_usage = reader.last_retrieval_usage
        from .capability import request, observations
        bundle = {"format": "howhard-agent-request-v5", "evidence_key": reader.evidence_key,
                  "payload": request(payload, self.config.data["capability"]), "snapshot": snapshot,
                  "stage": "filter", "capability": None, "refilter_round": 0,
                  "forecast_template": payload, "forecast_state": None, "coarse_usage": None}
        bundle["read_state"] = read_state
        bundle["dependencies"] = {"configuration": configuration(reader.config), "memories": {}, "evidence": {}}
        bundle["learned_state"] = {"weights_policy": "snapshot", "capability_observations": observations(reader, bundle["payload"])}
        bundle["inventory_source"] = inventory
        bundle["native_candidates"] = candidates
        bundle["project_root"] = str(self.config.project_root)
        return self._save_request(bundle, validate=False)

    def _save_request(self, bundle, *, validate=True):
        from .files import atomic_write
        from .request_state import extend
        extend(self, bundle)
        request_id = digest(bundle)
        with storage.locked(self.config, operation="request commit"):
            if validate:
                self._check_request(bundle)
            atomic_write(self.config.state_root / "requests" / f"{request_id}.json", canonical(bundle) + b"\n")
        return {"request_id": request_id, "stage": bundle["stage"], "refilter_round": bundle["refilter_round"],
                "capability": bundle["capability"], "request": semantic.visible_request(bundle["payload"]),
                "baseline_request": semantic.baseline_request(bundle["payload"])}

    def _load_request(self, request_id):
        self.refresh()
        import json
        import re
        if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{64}", request_id):
            raise HowHardError("request_id: expected SHA-256 request ID")
        try:
            bundle = json.loads((self.config.state_root / "requests" / f"{request_id}.json").read_bytes())
        except (OSError, ValueError) as exc:
            raise HowHardError("request_id: unavailable or invalid agent request") from exc
        from .validation import keys
        keys(bundle, ["format", "evidence_key", "payload", "snapshot", "native_candidates", "project_root", "stage", "capability", "refilter_round", "forecast_template", "forecast_state", "coarse_usage"],
             ["read_state", "dependencies", "learned_state", "inventory_source"], where="request")
        if digest(bundle) != request_id or bundle["format"] != "howhard-agent-request-v5":
            raise HowHardError("request_id: request integrity or format mismatch")
        if bundle["project_root"] != str(self.config.project_root):
            raise HowHardError("request_id: belongs to another worktree; prepare in the current project")
        with storage.reading(self.config):
            self._check_request(bundle)
        return bundle

    def _check_request(self, bundle):
        from .request_state import changes, StaleRequest
        self.refresh()
        if "dependencies" not in bundle:
            return self._validate_state(bundle["evidence_key"], bundle["snapshot"]["revision"])
        changed = changes(self, bundle)
        if changed:
            raise StaleRequest(changed, bundle)

    def _refresh_stale(self, error, request_id=None):
        from .request_state import StaleRequest
        if not isinstance(error, StaleRequest):
            return
        if request_id is not None:
            error.request_id = request_id
        bundle, payload = error.bundle, error.bundle["payload"]
        try:
            options = ({"inventory": bundle["inventory_source"]} if bundle.get("inventory_source")
                       else {"native_candidates": bundle["native_candidates"]})
            error.refreshed_request = self.prepare(payload["task_type"], payload["context"]["task_text"],
                project_context=payload["context"]["project_context"], risk=payload["risk_class"], **options)
            error.baseline_unchanged = semantic.baseline_request(payload) == error.refreshed_request["baseline_request"]
        except HowHardError as exc:
            error.refresh_error = str(exc)

    def retrieve(self, request_id, queries):
        try:
            return self._retrieve(request_id, queries)
        except HowHardError as exc:
            self._refresh_stale(exc, request_id)
            raise

    def _retrieve(self, request_id, queries):
        if not isinstance(queries, list) or not 1 <= len(queries) <= 3:
            raise HowHardError("retrieval: provide between one and three tool requests")
        bundle = self._load_request(request_id)
        from .request_state import view
        reader = view(self, bundle["read_state"]) if "read_state" in bundle else self
        if len(bundle["payload"]["retrieval"]) + len(queries) > 3 * semantic.SETTINGS["retrieval_rounds"]:
            raise HowHardError("retrieval: request budget exceeded")
        for query in queries:
            result = semantic.tool(reader, query, project_snapshot=bundle["payload"]["context"]["project_context"])
            bundle["payload"]["retrieval"].append({"request": query, "result": result})
        return self._save_request(bundle)

    def decide(self, request_id, response, *, seed=None):
        started = time.monotonic()
        try:
            result = self._decide(request_id, response, seed=seed)
        except Exception as exc:
            self._refresh_stale(exc, request_id)
            self._failed_decision_event(exc, "calling-agent", started)
            raise
        if "decision_id" in result:
            self._decision_event(result, result["task_type"], started)
        return result

    def _decide(self, request_id, response, *, seed=None):
        if seed is not None and (type(seed) is not int or seed < 0):
            raise HowHardError("seed: expected non-negative integer")
        return self._advance(self._load_request(request_id), response, seed=seed)

    def _forecast_request(self, bundle, requested):
        from copy import deepcopy
        from .capability import cell_key
        bundle["stage"] = "forecast"
        bundle["payload"] = deepcopy(bundle["forecast_template"])
        bundle["payload"]["cells"] = [c for c in bundle["payload"]["cells"] if cell_key(c) in requested]
        all_keys = requested | (set(cell_key(c) for c in bundle["forecast_state"]["cells"]) if bundle["forecast_state"] else set())
        inventory = [c for c in bundle["native_candidates"] if cell_key(c) in all_keys]
        p = bundle["payload"]
        band = self._prepared(p["task_type"], p["risk_class"], p["context"]["task_text"], p["context"]["project_context"], inventory)
        bundle["payload"]["cost_policy"] = band[0]["cost_policy"]
        return self._save_request(bundle)

    def _advance(self, bundle, response, *, seed=None, usage=None, select=True):
        from copy import deepcopy
        from . import capability
        payload, snapshot = bundle["payload"], bundle["snapshot"]
        from .request_state import extend
        extend(self, bundle)
        response = semantic.validate_response(response, payload, snapshot)
        if usage is None:
            semantic.external_submission(self, payload)
            usage = {"backend": "calling-agent", "seconds": 0., "cost_usd": None, "calls": [],
                     "cached": False, "call_seconds": 0., "call_cost_usd": None}
        with storage.reading(self.config):
            self._check_request(bundle)
        if bundle["stage"] == "filter":
            state = capability.aggregate(self, payload, response,
                observed=bundle.get("learned_state", {}).get("capability_observations"))
            bundle["coarse_usage"] = semantic.combine_usage(bundle["coarse_usage"], usage)
            state["reader_usage"] = {k: deepcopy(bundle["coarse_usage"][k]) for k in ("backend", "seconds", "cost_usd", "calls")}
            state["refilter_rounds"] = bundle["refilter_round"]
            if len(state["band"]) > state["policy"]["max_forecast_cells"]:
                if bundle["refilter_round"]:
                    raise HowHardError("capability.max_forecast_cells exceeded after re-filter; provide a tighter capability filter")
                bundle["refilter_round"] = 1
                bundle["payload"]["refilter"] = {"reason": "max_forecast_cells exceeded; provide a tighter filter", "previous": response["filter"],
                                                   "band_cells": len(state["band"]), "max_forecast_cells": state["policy"]["max_forecast_cells"]}
                return self._save_request(bundle)
            bundle["capability"] = state
            return self._forecast_request(bundle, set(state["band"]))
        # Validate the new panel before combining it with earlier widening rounds.
        previous = bundle["forecast_state"]
        combined = deepcopy(response)
        combined_payload = deepcopy(payload)
        if previous is not None:
            combined["forecasts"] = previous["response"]["forecasts"] + combined["forecasts"]
            items = {(m["id"], m["version"]): deepcopy(m) for m in previous["response"]["memory_forecasts"]}
            for item in response["memory_forecasts"]:
                pair = item["id"], item["version"]
                if pair in items:
                    items[pair]["forecasts"].extend(item["forecasts"])
                else:
                    items[pair] = deepcopy(item)
            combined["memory_forecasts"] = list(items.values())
            combined_payload["cells"] = previous["cells"] + combined_payload["cells"]
            usage = semantic.combine_usage(previous["usage"], usage)
        forecast_pairs = {(c["family"], c["effort"]) for c in combined["forecasts"]}
        inventory = [c for c in bundle["native_candidates"] if (self.config.data["models"][c["model"]]["family"], c["effort"]) in forecast_pairs]
        full = self._prepared(payload["task_type"], payload["risk_class"], payload["context"]["task_text"],
                              payload["context"]["project_context"], inventory)
        prepared = deepcopy(full[0]), [deepcopy(c) for c in full[1] if (c["family"], c["effort"]) in forecast_pairs]
        prepared[0]["native_candidates"] = deepcopy(bundle["native_candidates"])
        prepared = semantic.augment(self, prepared, combined, combined_payload, snapshot, usage)
        state = deepcopy(bundle["capability"])
        state["disagreements"] = capability.disagreements(state, prepared)
        admissible = any(c["admissible"] for c in prepared[0]["cells"])
        forecasted = {capability.cell_key(c) for c in prepared[0]["cells"]}
        stronger = [c["key"] for c in state["cells"] if c["classification"] == "overpowered" and c["key"] not in forecasted]
        if (not admissible and stronger and len(state["widening"]) < state["policy"]["widening_rounds"]
                and len(forecasted) < state["policy"]["max_forecast_cells"]):
            next_cell = stronger[0]
            state["widening"].append({"round": len(state["widening"])+1, "cells": [next_cell], "trigger": "silent-cap-unmet"})
            bundle["capability"] = state
            bundle["forecast_state"] = {"response": combined, "cells": combined_payload["cells"], "usage": usage}
            return self._forecast_request(bundle, {next_cell})
        state["widening_stop"] = "cap-met" if admissible else "forecast-budget" if len(forecasted) >= state["policy"]["max_forecast_cells"] else "round-limit" if stronger else "exhausted"
        prepared[0]["capability"] = state
        if "dependencies" in bundle:
            prepared[0]["request_snapshot"] = {"weights_policy": "snapshot", "dependencies": deepcopy(bundle["dependencies"]),
                "evidence_key": bundle["evidence_key"], "memory_revision": snapshot["revision"]}
        # Include both reader calls in routing overhead; no invented provider costs.
        coarse = bundle["coarse_usage"]
        prepared[0]["backend"]["inference_cached"] = prepared[0]["backend"]["inference_cached"] and coarse["cached"]
        prepared[0]["routing_overhead"]["seconds"] += coarse["call_seconds"]
        if coarse["call_cost_usd"] is None or prepared[0]["routing_overhead"]["cost_usd"] is None:
            prepared[0]["routing_overhead"]["cost_usd"] = None
        else:
            prepared[0]["routing_overhead"]["cost_usd"] += coarse["call_cost_usd"]
        if not select:
            return prepared[0]
        seed = secrets.randbits(64) if seed is None else seed
        result = choose(prepared, self.config, seed, semantic.sample(prepared, seed))
        chosen = next(c for c in state["cells"] if (c["model"], c["effort"]) == (result["chosen"]["model"], result["chosen"]["effort"]))
        result["capability"]["executed"] = {k: chosen[k] for k in ("key", "classification", "position", "P_clear")}
        with storage.locked(self.config, operation="decision commit"):
            self._check_request(bundle)
            result["decision_id"] = decisions.save(self.config, result)
        return result

    def memory_learning_context(self, evidence_id):
        self.refresh()
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

    def attach_usage(self, execution_id, manifest, *, id):
        """Collect independently of outcome durability, then append one correction."""
        from .accounting import collect
        from copy import deepcopy
        usage = collect(manifest)
        if usage["execution_id"] != execution_id:
            raise HowHardError("usage manifest.execution_id: must match target execution")
        self.refresh()
        current = self.evidence(execution_id)
        if current["id"] != execution_id:
            raise HowHardError("usage attach: target must be the original execution ID")
        # Same ID retries compare the originally recorded outcome as well as usage.
        # Later outcome amendments must not turn a successful measurement retry into a new event.
        existing = next((r for r in self.raw_rows if r["id"] == id), None)
        if existing is not None:
            if existing["amends"] != execution_id or existing.get("measurement_corrections") != {"execution_usage": usage}:
                raise HowHardError("usage attach.id: conflicting attachment")
            return deepcopy({k: v for k, v in existing.items() if k != "project"})
        if current.get("execution_usage") == usage:
            return deepcopy(current)
        return self.observe(current["task_type"], current["model"], current["effort"],
                            current["outcome"], cause=current["cause"], amends=execution_id,
                            id=id, _preserve_outcome=True, measurement_corrections={"execution_usage": usage})

    def save_decision(self, decision, path):
        receipt = decisions.load(self.config, decision)
        from .files import atomic_write
        atomic_write(path, canonical(receipt) + b"\n")

    def flush(self):
        ids = storage.flush(self.config)
        self.refresh()
        return ids
