# MD Research Interpretation Notes

These notes translate the MD result tables into paper-facing claims. They are intended as a writing log, not a replacement for the canonical tables in `md_research/results/tables/`.

## Main Empirical Reading

The results support the MD paper framing: SpecGuard-Chem is best interpreted as a benchmark-validity case study, not as evidence of medicinal chemistry or drug-discovery capability.

The central result is that molecule-level success and task-level scientific action correctness can diverge. Several systems can produce molecules that pass visible hard constraints while failing the required action label. This is the reason the paper should distinguish molecule acceptance from action accuracy.

## Hypotheses And Results

### H1: Valid molecules are not the same thing as valid decisions

Supported strongly, especially by deterministic molecule-producing baselines.

Examples from `tables/molecule_acceptance_vs_action_accuracy.md`:

- `corpus_search`: molecule acceptance `0.900`, action accuracy `0.787`, balanced action accuracy `0.333`, reject recall `0.000`, abstain recall `0.000`.
- `local_mutation`: molecule acceptance `0.850`, task-inconsistent accept rate `0.529`.
- `verify_first`: molecule acceptance `0.850`, task-inconsistent accept rate `0.529`.
- `always_accept`: molecule acceptance `0.738`, task-inconsistent accept rate `0.471`.

This supports the paper claim that molecule-level metrics can overstate task-level success.

### H2: Reject and abstain semantics expose hidden failure modes

Supported, but the wording should be precise.

The deterministic molecule-producing baselines mostly collapse on reject and abstain:

- `corpus_search`: reject recall `0.000`, abstain recall `0.000`.
- `local_mutation`: reject recall `0.000`, abstain recall `0.000`.
- `verify_first`: reject recall `0.000`, abstain recall `0.000`.

The external LLM rows are more varied. Strong OpenAI and strong Anthropic rows identify reject and abstain tasks well in the current canonical table, while weaker or differently configured rows may over-abstain or underperform on accept tasks. Therefore the safe paper claim is not "LLMs fail reject/abstain"; it is: reject and abstain recall are necessary diagnostics because they reveal action collapse that molecule acceptance alone hides.

### H3: Verifier/tool context helps

Mixed. Do not overclaim.

DeepSeek fast improves under verify-L3 in the canonical table:

- `deepseek_fast_closed`: action accuracy `0.675`, molecule acceptance `0.475`.
- `deepseek_fast_verify_l3`: action accuracy `0.738`, molecule acceptance `0.537`.

But strong OpenAI and strong Anthropic do not uniformly improve under verify-L3 in this snapshot. This may reflect interaction between forced verifier-first behavior, feedback loops, budgets, and native task groupings. The protocol table already gives the necessary caveat: protocol slices are descriptive groupings rather than a forced same-task causal intervention across all protocol levels.

### H4: Wrapper saturation limits benchmark claims

Supported strongly.

The `well_engineered_wrapper` reaches:

- action accuracy `1.000`
- balanced action accuracy `1.000`
- accept recall `1.000`
- reject recall `1.000`
- abstain recall `1.000`
- task-inconsistent accept rate `0.000`

This means SpecGuard-Chem can be solved as a formal contract-compliance and verifier/search engineering problem. That limits claims about medicinal chemistry expertise. It does not make the artifact useless; it clarifies what the artifact measures.

## Metric Meanings

### Action

The benchmark has three primary scientific action labels:

- `ACCEPT`: provide or accept a molecule as satisfying the task.
- `REJECT`: reject a candidate or attempted molecule because it fails the required criteria.
- `ABSTAIN`: decline because the visible hard constraints are contradictory or the task cannot be completed under the stated rules.

Action accuracy is the fraction of tasks where the final decision exactly matches the hidden expected action.

### Recall

Recall is action-specific sensitivity.

- Accept recall: among tasks whose correct action is `ACCEPT`, how many ended as `ACCEPT`.
- Reject recall: among tasks whose correct action is `REJECT`, how many ended as `REJECT`.
- Abstain recall: among tasks whose correct action is `ABSTAIN`, how many ended as `ABSTAIN`.

The MD subset has 80 tasks: 63 `ACCEPT`, 9 `REJECT`, and 8 `ABSTAIN`. Therefore `reject_recall = 0.000` means the system got zero of the 9 reject-required tasks correct.

### Balanced Action Accuracy

Balanced action accuracy is the mean of accept recall, reject recall, and abstain recall. It prevents the result from being dominated by the majority class.

In this subset, most tasks are `ACCEPT`. A system can get a decent raw action accuracy by often accepting, even if it never handles reject or abstain. Balanced action accuracy makes that visible. For example, `corpus_search` accepts all 63 accept-required tasks, but it gets no reject-required or abstain-required tasks correct, so its balanced action accuracy is only `0.333`.

### Molecule Acceptance Rate

Molecule acceptance rate is the fraction of tasks ending in final `ACCEPT`. It is not task success. It tells us how often the runner accepted a molecule, not whether accepting was the right scientific action.

This is why a system can have high molecule acceptance and still fail the benchmark's decision semantics.

### Task-Inconsistent Accept Rate

Task-inconsistent accept rate is:

`ACCEPT decisions on tasks whose expected action is REJECT or ABSTAIN / all REJECT-or-ABSTAIN tasks`

It captures unsafe or scientifically wrong acceptance. In the MD subset, the denominator is 17 tasks: 9 reject-required plus 8 abstain-required tasks.

Example from `corpus_search`: task `sgchem_v1.0__bundle__00010__audit_reject__03` is an `audit_reject` task with expected action `REJECT`. `corpus_search` returned `CC(=O)NC(C)C(C)N`, which passed visible hard constraints, so the runner accepted it as a molecule. But the correct action for the task was to reject the given candidate, not to substitute a passing molecule. That is a valid molecule but the wrong scientific action.

## Corpus Search Baseline

`corpus_search` is deterministic retrieval, not random guessing.

It builds a deterministic generated corpus, evaluates candidates against the visible specification, and returns the first hard-passing canonical candidate. For repair tasks, it preferentially chooses the hard-passing candidate most similar to the input by Morgan/Tanimoto similarity.

The point of this baseline is to test whether a system can do well by finding any molecule that satisfies visible constraints, without understanding the task action. It is intentionally action-naive. That is why it gets high molecule acceptance but poor reject/abstain recall.

## Why The Runner Can Allow A Wrong Scientific Action

This is not a bug in the benchmark; it is the point being measured.

The runner can verify whether a proposed molecule passes visible chemical constraints. But task correctness is larger than molecule validity. Some tasks require rejecting a supplied candidate or abstaining on contradictions. If a system ignores that and supplies a different molecule that passes constraints, the molecule can be chemically valid while the decision is scientifically wrong.

The scorer therefore records both:

- molecule-level success: did the final accepted molecule pass hard constraints?
- action-level success: was `ACCEPT`, `REJECT`, or `ABSTAIN` the correct action for this task?

## External LLM Notes

The external LLM snapshot is diagnostic evidence, not the primary offline leaderboard.

A key caveat discovered after the first all-provider run: Anthropic Haiku often returned markdown-fenced JSON. The original strict adapter treated fenced JSON as invalid and normalized it to abstention. That means the canonical `anthropic_fast_*` abstention collapse should be treated cautiously as an adapter/protocol artifact, not a clean model-capability finding.

The adapter has since been updated to parse fenced JSON. A corrected Anthropic rerun was started, but the Anthropic account ran out of credit before the corrected Sonnet rows completed. Until the corrected run is finished and replayed, the safest paper wording is:

- OpenAI and DeepSeek rows are replayed from complete cached live responses.
- Anthropic rows are present in the canonical table, but the Haiku rows should be interpreted with a parser-format caveat.
- Do not make a strong claim that Anthropic Haiku "collapses to abstention" as a model behavior from the current canonical table alone.

OpenAI fast also produced many empty responses in the cached snapshot. That looks like an API/model-output configuration issue rather than a safety refusal. It should be described as an external-adapter diagnostic failure mode, not as evidence that the model was blocked by safety policy.

## Paper-Framing Paragraph

The SpecGuard-Chem results show that chemically typed benchmark tasks can separate molecule-level validity from action-level scientific correctness. Retrieval and mutation baselines can produce many hard-passing molecules while failing reject and abstain semantics, demonstrating that molecule acceptance alone is not a valid measure of task success. The well-engineered wrapper reaches ceiling performance, showing that the benchmark can be solved as a deterministic contract-compliance problem and therefore should not be interpreted as measuring medicinal chemistry expertise. At the same time, live external LLM rows remain below that deterministic ceiling, indicating that the benchmark is not saturated from the perspective of current general-purpose model adapters. The appropriate interpretation is that SpecGuard-Chem is useful as a controlled benchmark-validity artifact for rule-following, action selection, and public/private scoring separation in drug-discovery-adjacent settings, but not as a direct test of therapeutic value, biological activity, or real-world medicinal chemistry decision-making.
