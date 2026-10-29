"""Public application API: fit once, inspect summaries and make many seeded picks."""

from pathlib import Path

from . import cache, storage
from .config import Config, load_config
from .outcomes import fit
from .prior import build_prior
from .regression import fit_regression
from .selector import choose, summarize
from .transfer import export_prior, import_prior, read_export
from . import decisions
from .context import FORMAT, canonical, capture_project, validate_text
from .validation import HowHardError


class Engine:
    def __init__(self, config=None, *, directory=None, cache_path=None, journal_path=None):
        self.config = config if isinstance(config, Config) else load_config(config, directory=directory)
        if cache_path is not None:
            self.config.data["cache"] = str(Path(cache_path).resolve())
        if journal_path is not None:
            self.config.data["journal"] = str(Path(journal_path).resolve())
        self.config.validate_paths()
        self._summaries = {}
        self.refresh()

    def refresh(self, *, full=False):
        config = self.config
        with storage.locked(config):
            raw_rows, blobs = storage.snapshot(config)
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

    def _prepared(self, task_type, risk, task_text="", project_context=None):
        context = {"encoder": FORMAT, "task_text": validate_text(task_text), "project_context": capture_project(self.config, project_context)}
        key = task_type, risk, canonical(context)
        if key not in self._summaries:
            self._summaries[key] = summarize(self.posterior, self.config, task_type, risk, context)
        return self._summaries[key]

    def summary(self, task_type, task_text="", *, project_context=None, risk=None):
        from copy import deepcopy
        return deepcopy(self._prepared(task_type, risk, task_text, project_context)[0])

    def pick(self, task_type, task_text="", *, project_context=None, risk=None, seed=None, candidates=None):
        result = choose(self._prepared(task_type, risk, task_text, project_context), self.posterior, self.config, seed, candidates)
        result["decision_id"] = decisions.save(self.config, result)
        return result

    def observe(self, task_type, model, effort, outcome, *, decision=None, task_text=None, project_context=None, **options):
        if decision is not None:
            receipt = decisions.load(self.config, decision)
            if task_text is not None or project_context is not None or options.get("amends") is not None:
                raise HowHardError("decision: cannot override receipt context or combine with an amendment")
            if (task_type, model, effort) != (receipt["task_type"], receipt["chosen"]["model"], receipt["chosen"]["effort"]):
                raise HowHardError("decision: task type, model and effort must match the receipt")
            options.update(task_text=receipt["context"]["task_text"], project_context=receipt["context"]["project_context"],
                           risk_class=receipt["risk_class"], decision_id=decisions.digest(receipt))
        elif not options.get("amends"):
            options.update(task_text="" if task_text is None else task_text, project_context=capture_project(self.config, project_context))
        elif task_text is not None or project_context is not None:
            raise HowHardError("amends: historical context belongs to the original run")
        row = storage.observe(self.config, task_type, model, effort, outcome, **options)
        self.refresh()
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
