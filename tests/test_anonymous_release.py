from __future__ import annotations

import re


def test_anonymous_release_has_no_local_or_author_identifiers(v1_release) -> None:
    forbidden = [
        "".join(("Da", "niel")),
        "".join(("Hus", "sey")),
        "".join(("/Us", "ers/")),
        "".join(("github.com/", "dan", "hus", "sey")),
    ]
    email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    for path in v1_release.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in {".parquet"}:
            continue
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text
        assert email_re.search(text) is None
