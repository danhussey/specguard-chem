# Metric Sanity Interpretation

Molecule acceptance is not a stable proxy for task success in SpecGuard-Chem.
Accept-biased and retrieval-heavy systems can score well on molecule acceptance while failing REJECT and ABSTAIN semantics.
Action accuracy and balanced action accuracy expose those failures directly.
Reject recall isolates audit/rejection collapse, while abstain recall isolates contradiction-handling behavior.
The wrapper row is a public verifier/search ceiling and must remain separate from closed-book systems.
Cost-adjusted metrics are useful only after making the access model and verifier-call accounting explicit.
The paper should report metric ranking sensitivity as evidence for action-aware evaluation rather than as a single universal leaderboard.
