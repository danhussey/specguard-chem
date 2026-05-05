# Related Work

## Dataset And Evaluation Documentation

SpecGuard-Chem follows the documentation and artifact-accountability tradition of Datasheets for Datasets (Gebru et al., 2021), Model Cards (Mitchell et al., 2019), Data Cards (Pushkarna et al., 2022), and Croissant/MLCommons metadata. The relevant connection is not simply that the dataset is documented, but that the documentation constrains the evaluative claims: intended use, out-of-scope use, provenance, validation, limitations, and Responsible AI metadata are treated as part of the benchmark contract.

## Agent And Tool-Use Evaluation

Agent and tool-use benchmarks such as AgentBench, ToolLLM/ToolBench, and the Berkeley Function Calling Leaderboard evaluate broad autonomy, function calling, tool selection, and multi-step execution. SpecGuard-Chem is narrower: it tests whether an agent interface preserves a public action contract under hidden oracle certificates and bounded verifier access. The key distinction is that tool use is not only a capability signal; it is also a threat model that can collapse a formalizable benchmark if an engineered wrapper implements the public verifier semantics.

## Chemistry Language-Model Benchmarks

Chemistry language-model benchmarks such as ChemBench test chemical knowledge, question answering, or expert-style reasoning. SpecGuard-Chem does not claim broad chemistry expertise. It uses medicinal-chemistry-inspired rule cards as a typed substrate for machine-checkable specification compliance, and does not evaluate biological activity, toxicity, synthesizability, therapeutic efficacy, clinical utility, target binding, dosing, disease relevance, or drug-discovery success.

## Evaluation Science And Benchmark Saturation

The NeurIPS Evaluations & Datasets framing treats evaluation itself as a scientific object: protocols, audits, stress tests, saturation analysis, negative results, and methods for interpreting evaluative claims are in scope. SpecGuard-Chem fits this line of work by releasing not only prompts, but an oracle-compiled evaluation contract with public/private task isolation, negative controls, leakage audits, denominator reports, and a verifier-wrapper saturation baseline.
