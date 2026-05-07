# SpecGuard-Chem — Venue-Tuned Pitch Variants

Use this to quickly retune title/abstract for different A* reviewer priors without changing the underlying artifact/repro story.

## Variant A (Recommended): LLM/Agent Evaluation + Tool Use
**Tagline:** Budgeted constraint-following with measurable tool gating and abstention.

**Title (option):** SpecGuard-Chem: A Frozen Benchmark for Budgeted Constraint Following, Tool Use, and Abstention

**Abstract (draft):**
Agentic LLM systems are increasingly asked to follow explicit constraints, use tools, and decide when to abstain. We introduce SpecGuard-Chem, a solver-agnostic benchmark harness for evaluating structured constraint compliance under strict interaction budgets with deterministic, offline RDKit-backed verifiers. SpecGuard-Chem supports a protocol ladder (L1 one-shot, L2 repair with coarse feedback, L3 tool-in-loop with explicit verify calls) that gates detailed feedback behind measurable tool usage. The benchmark reports pass@budget, hard-constraint violation rates, verifier economy, abstention utility under configurable cost models, calibration (ECE/Brier), and robustness suites (boundary precision, adversarial SMILES invariance, interrupt/resume). We release a frozen benchmark artifact (sgchem_v0.3) with train/dev/test splits, manifest checksums, machine-checkable labels (witnesses for ACCEPT and contradiction proofs for ABSTAIN), and bootstrap confidence intervals. Track-separated results highlight why retrieval-enabled solvers must be reported as an upper-bound rather than combined with closed-book rankings.

**Contributions (3):**
- Protocol ladder with explicit tool gating and budgets.
- Frozen benchmark artifact + auditability (manifest, checksums, trace logs).
- Safety-relevant metrics (utility, calibration, robustness, interrupts) beyond pass rate.

## Variant B: Reliability/Safety of Agentic Systems
**Tagline:** When to abstain, how to handle interrupts, and how to resist gaming under bounded interaction.

**Title (option):** SpecGuard-Chem: Auditable Evaluation of Abstention, Interrupt Safety, and Gaming Resistance Under Budgeted Constraints

**Abstract (draft):**
Systems that act under explicit constraints must also decide when to abstain, handle interruptions safely, and avoid brittle behavior that can be gamed. We present SpecGuard-Chem, an offline-deterministic evaluation harness that measures these behaviors in a chemistry-flavored domain with strict, machine-checkable constraints and RDKit-backed verifiers. The benchmark enforces interaction budgets across a protocol ladder (L1/L2/L3) and reports decision-level outcomes (accept/reject/abstain), cost-sensitive utility, calibration, interrupt/resume success with step costs, and robustness under adversarial invariance and boundary-precision tasks. We release a frozen benchmark artifact (sgchem_v0.3) with checksummed manifests, split files, and machine-checkable task labels and evidence. Initial track-separated results show non-trivial closed-book performance and measurable failure modes on robustness and interrupt suites, motivating evaluation beyond aggregate pass rates.

**Contributions (3):**
- Auditable decision semantics and cost-sensitive utility for abstention behavior.
- Robustness suites (boundary + adversarial invariance) and interrupt/resume protocols.
- Frozen artifact and offline determinism for reproducible safety evaluation.

## Variant C: Cheminformatics / Computational Methods (Non-Discovery)
**Tagline:** A reproducible benchmark for constraint satisfaction and repair in molecular representations.

**Title (option):** SpecGuard-Chem: Deterministic, RDKit-Verifiable Benchmarking of Constraint Satisfaction and Repair in Molecular Editing

**Abstract (draft):**
We present SpecGuard-Chem, a deterministic benchmark harness for evaluating molecular editing agents against explicit, machine-checkable constraints using RDKit-backed verifiers. The benchmark formalizes specs as typed constraint objects (properties, alert families, substructures, SA proxy) and tasks as budgeted episodes with expected actions and evidence. A protocol ladder (L1 one-shot, L2 assisted repair, L3 tool-in-loop verification) enables controlled study of feedback and tool usage. The evaluation produces trace logs and a metrics report covering constraint compliance, repair efficiency under budgets, edit economy, calibration of predicted hard-pass probability, and robustness to representational and boundary effects (SMILES invariance and near-boundary precision). We release a frozen benchmark artifact (sgchem_v0.3) with manifest checksums, split files, and machine-checkable labels and evidence for reproducible benchmarking, while explicitly making no claims about biological activity, toxicity, or synthesis planning.

**Contributions (3):**
- Strict schemas and deterministic verifiers for rule-based evaluation.
- Budgeted protocols for proposal/repair and explicit verification.
- Frozen release packaging (manifest/checksums) and paper-ready reporting.
