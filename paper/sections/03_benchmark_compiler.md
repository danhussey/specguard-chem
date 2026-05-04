# Benchmark Compiler

SpecGuard-Chem compiles tasks from bundles. A bundle is one underlying specification scenario with multiple controlled task views. This makes related tasks explicit and lets the split policy operate on bundles rather than individual prompts.

The compiler is oracle-first. It creates tasks only when there is machine-checkable evidence:

- feasible witnesses for construct and accept cases
- violation certificates for reject and repair cases
- explicit contradiction certificates for abstention
- boundary certificates for paired pass/fail threshold cases
- equivalence certificates for representation-invariance cases
- interrupt certificates for stateful protocol tasks

The public/private boundary is part of the compiler contract. Hidden fields such as expected_action, evidence, oracle_type, witnesses, certificates, task_id, bundle_id, and internal task_type labels are not provided to normal adapters. PublicTaskView exposes only rendered_agent_input, public input molecules, public specification fields, protocol, budgets, allowed actions, allowed tools, round index, and permitted feedback.

The split policy assigns bundles to train/dev/test, then keeps all tasks from each bundle in the same split. Leakage audits check bundle overlap, agent-visible hash overlap, boundary group leakage, invariance group leakage, canonical input/spec leakage, and related witness/spec leakage.
