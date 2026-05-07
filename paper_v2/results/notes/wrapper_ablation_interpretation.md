# Wrapper Ablation Interpretation

The full wrapper reached action_accuracy=1.000 on the held-out test split.
Disabling public candidate search changed action_accuracy to 0.977, which estimates the contribution of retrieval/search over public candidates.
Disabling explicit L3 verify tool calls changed action_accuracy to 1.000; this variant still uses deterministic local evaluation of public specification fields, so it should be interpreted as a tool-call ablation rather than a complete removal of verifier semantics.
The smallest measured explicit verify budget with at least 0.95 action accuracy was wrapper_verify_budget_1; the measured budget curve should be cited instead of assuming saturation.
Scrambling/ignoring public task names gave action_accuracy=1.000, testing whether the wrapper depends on visible family-name artifacts.
These rows support the evaluation-contract interpretation: wrapper performance is an access-model ceiling under public verifier/search assumptions, not a closed-book chemistry capability result.
