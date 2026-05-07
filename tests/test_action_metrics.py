from __future__ import annotations

from specguard_chem.scoring.reports import summarise


def test_action_confusion_tracks_accept_reject_abstain_and_invalid() -> None:
    records = [
        {"task_id": "a", "expected_action": "ACCEPT", "final_decision": "ACCEPT", "hard_pass": True, "rounds": []},
        {"task_id": "b", "expected_action": "REJECT", "final_decision": "REJECT", "hard_pass": False, "rounds": []},
        {"task_id": "c", "expected_action": "ABSTAIN", "final_decision": "ABSTAIN", "hard_pass": False, "rounds": []},
        {"task_id": "d", "expected_action": "REJECT", "final_decision": "ACCEPT", "hard_pass": True, "rounds": []},
        {"task_id": "e", "expected_action": "ACCEPT", "final_decision": "REJECT", "hard_pass": False, "rounds": []},
        {"task_id": "f", "expected_action": "ACCEPT", "final_decision": "ABSTAIN", "hard_pass": False, "rounds": []},
        {"task_id": "g", "expected_action": "ABSTAIN", "final_decision": "ABSTAIN", "schema_error": True, "rounds": []},
    ]
    summary = summarise(records)
    assert summary["confusion"]["ACCEPT"]["ACCEPT"] == 1
    assert summary["confusion"]["REJECT"]["REJECT"] == 1
    assert summary["confusion"]["ABSTAIN"]["ABSTAIN"] == 1
    assert summary["confusion"]["REJECT"]["ACCEPT"] == 1
    assert summary["confusion"]["ACCEPT"]["REJECT"] == 1
    assert summary["confusion"]["ACCEPT"]["ABSTAIN"] == 1
    assert summary["confusion"]["ABSTAIN"]["INVALID"] == 1
    assert summary["correct_reject_rate"] == 0.5
    assert summary["invalid_output_rate"] > 0
