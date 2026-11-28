## howhard routing and attribution

- Before delegating an authorized task to a worker, use the installed `howhard` skill to
  select the worker's native model and reasoning effort from the host's supported
  controls. Respect explicit user model choices and required review.
- Evaluate any independent baseline forecast in a fresh, stateless context. It is part
  of routing, so never route it through `howhard` again.
- Start the worker with the selected controls. If routing is unavailable or the controls
  cannot be enforced, report the limitation instead of bypassing routing.
- After execution, record the truthful outcome and measured usage under a stable
  execution ID, leaving unmeasured usage unknown. Then update memory with
  `howhard memory learn`.
- During every bug-fix task or requested audit of completed work, invoke the installed
  `attribute-failure` skill to trace whether a defect escaped an earlier accepted agent
  task. Amend only supported attribution in `howhard`; leave uncertain attribution
  unscored.
