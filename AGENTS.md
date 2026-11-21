# howhard contributor instructions

Keep this file limited to repository-wide guardrails. Follow the canonical contracts and
procedures listed in [docs/index.md](docs/index.md). These guardrails adopt AGCoord's
contributor workflow for this repository.

- Use the installed `agrein` skills: `ticket-delivery` before changing code, docs,
  config, infrastructure, or live state; `epic-queue` when decomposing multi-ticket
  work or coordinating subagent workers; `bug-investigation` before fixing defects or
  failing checks; and `owner-escalation` for unsettled product/policy rulings, live
  operation approvals, and decision handoffs.
- Coordinate ticket work by delegating each authorized ticket to a worker selected with
  the installed `agroute` and `howhard` skills; an assigned leaf worker implements its
  ticket directly without recursively delegating it (report a capability gap first if
  `agroute` or worker delegation is unavailable on the host):
  1. Inspect or refresh live agent quota state with `agr usage`.
  2. Emit eligible provider-qualified candidates for this host with
     `agr candidates --host <codex|claude|agy> --output /tmp/TASK-candidates.json --json`.
     If `price_request` is non-null, look up official first-party provider pricing (or
     classify internal harness models as `"alias"` or `"unlisted"`), run
     `agr prices resolve --file /tmp/TASK-prices.json --json`, and re-emit candidates once.
  3. Select a route with `howhard` (`context`/`retrieve`/`decide` with an independent
     stateless baseline forecast, or `pick`), then resolve the execution plan with
     `agr plan --host <codex|claude|agy> --decision /tmp/TASK-decision.json --json`.
  4. Honor `needs_review`. Execute native plans (`target_agent == host_agent`) with the
     host's native subagent tool and cross-agent plans (`target_agent != host_agent`) with
     the returned non-interactive CLI command.
  5. Record the actual outcome and measured usage in `howhard` following the plan's
     `feedback` block (`matched_receipt` or `raw_native`), then update `howhard` memory.
- Track every code, test, documentation, or packaging change with a descriptive issue.
  Pair every issue number with a sentence explaining the work it tracks. Do not file an
  issue solely for machine-local installation, deployment, configuration, service
  lifecycle, or other operator state; create one only when that work also requires a
  repository change or the user explicitly requests an issue.
- Develop and validate only in an issue-specific Git worktree branched from `main`.
  Preserve unrelated work and do not develop in the primary checkout.
- Verify behavior at public boundaries. Start bug fixes with a failing behavioral test;
  accompany runtime changes with focused tests, affected canonical documentation, and a
  `CHANGELOG.md` entry when the change is user-facing.
- A documentation-only change needs no tests or AGCoord gate. It qualifies only when every
  changed path is `README.md`, `CHANGELOG.md`, `AGENTS.md`, Markdown below `docs/`, or a
  `.png` or `.svg` image below `docs/assets/` whose provenance is recorded in
  `docs/assets/README.md`. Verify the exact pull-request file list and diff, then merge it
  through the forge; the issue, worktree, branch, pull-request, and no-direct-target-push
  rules still apply.
- Tests own and stop every process, repository, and temporary state they create.
  Never inspect or clean another agent's state.
- Keep the package independent of a parent application or sibling checkout.
  Put project-specific assumptions in optional configuration; the ordinary workflow
  must work without configuration or statistical parameter tuning.
- For every other change after bootstrap, route checks and publication through the local
  coordinator, declare every scarce resource, and never submit coordinated work from an
  admitted job. Gate and publish through one `agc land` request; never bypass or separate
  its verdict from publication.
- Never upload a howhard distribution to PyPI unless the user explicitly requests it.
  Permission to implement, test, commit, push, merge, tag, or create a GitHub release
  does not authorize an upload.
- End changed work with the four-part [session handoff](docs/session_handoff_format.md);
  answer read-only questions directly without an empty handoff shell.
- Follow the [contributor workflow](docs/contributing.md) for commands, bootstrap,
  recovery, and cleanup. The [model and storage contracts](docs/reference.md) and
  [release guide](docs/releasing.md) define the corresponding component contracts.
