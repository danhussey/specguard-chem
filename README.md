# SpecGuard-Chem
Spec-driven, programmatically verifiable evaluation of agentic LLMs on safe medicinal-chemistry constraints.

**What it is:** a small, model-agnostic test rig. Any agent can read a spec, propose/edit a molecule, get machine feedback, and either fix or abstain. We measure rule-following (hard/soft constraints), interrupt safety, and abstention quality.

**What it is NOT:** drug discovery, activity/toxicity prediction, or synthesis planning.

## Quickstart
```bash
pip install -e .
specguard-chem run --suite basic --protocol L1 --model heuristic
specguard-chem run --suite basic --protocol L3 --model open_source_example
specguard-chem report --run-path runs/2025-01-01_basic_L3/
```

Repro in <10 minutes on a laptop. No proprietary data; all tasks are synthetic and safe.
