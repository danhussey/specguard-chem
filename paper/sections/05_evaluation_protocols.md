# Evaluation Protocols

SpecGuard-Chem evaluates three protocol levels.

L1 is one-shot and does not expose verifier tools. L2 may expose public feedback according to the task protocol, but not hidden oracle evidence. L3 permits verifier-tool interaction under explicit budgets. Tool-forced tasks require verifier availability, and interrupt/resume tasks require state tracking across at least two steps.

Normal model adapters consume PublicTaskView, not raw task objects. This prevents answer leakage from hidden fields and keeps audit_accept/audit_reject internal labels out of model-visible prompts. Visible task names are label-neutral, such as `candidate_audit`, `construct`, `repair`, and `feasibility_check`.

Protocol diagnostics are reported with denominators. Boundary precision has 22 test tasks, representation invariance has 20 test tasks, and interrupt/resume has 11 test tasks in sgchem_v1.0. All are treated as diagnostic slices unless future releases increase denominators.
