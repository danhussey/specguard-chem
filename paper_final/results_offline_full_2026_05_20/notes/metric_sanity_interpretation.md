# Metric Sanity Interpretation

Ranking by molecule_acceptance_rate selects corpus_search, which is not necessarily the best action-contract system.
Ranking by action_accuracy selects well_engineered_wrapper, directly measuring whether the system chose Accept, Reject, or Abstain correctly.
Reject recall is led by always_reject, while abstain recall is led by always_abstain; separating these metrics makes action collapse visible.
Task-inconsistent acceptance is a critical counter-metric because it counts Accept decisions on tasks that require Reject or Abstain.
Hard-violation and schema-error rates should remain safety and validity diagnostics rather than substitutes for action accuracy.
Cost-adjusted action scores distinguish systems that spend verifier calls from systems that achieve similar action accuracy with fewer calls.
The paper should present molecule acceptance as a misleading baseline diagnostic, not as specification success.
The most defensible headline is that access model and metric choice jointly determine the apparent winner.
