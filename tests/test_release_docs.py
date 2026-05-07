from __future__ import annotations

from pathlib import Path


def test_required_docs_exist_and_reference_v1(v1_release) -> None:
    docs = [
        Path("docs/GENERATOR_DESIGN_v1.md"),
        Path("BENCHMARK_CARD.md"),
        Path("SAFETY.md"),
        Path("METHODS.md"),
        Path("METRICS.md"),
        Path("SPEC.md"),
        v1_release / "RELEASE_NOTES.md",
        v1_release / "BENCHMARK_CARD.md",
    ]
    for path in docs:
        assert path.exists(), path
        text = path.read_text(encoding="utf-8")
        assert "sgchem_v1.0" in text
        assert "primary release is `sgchem_v0.3`" not in text
        assert "validate-dataset benchmarks/releases/sgchem_v1.0 --strict" in text
        assert "compile-benchmark" in text


def test_docs_include_non_claims() -> None:
    safety = Path("SAFETY.md").read_text(encoding="utf-8")
    assert "does not evaluate biological activity" in safety
    assert "synthesis feasibility" in safety
    assert "therapeutic efficacy" in safety
