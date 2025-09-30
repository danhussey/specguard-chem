#!/usr/bin/env python3
"""
SpecGuard-Chem Leaderboard Update Script

Updates the leaderboard with new submissions after validation.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import shutil


class LeaderboardUpdater:
    """Manages leaderboard updates and ranking."""

    def __init__(self, leaderboard_path: Path):
        """Initialize with leaderboard data file."""
        self.leaderboard_path = leaderboard_path
        self.submissions_dir = leaderboard_path.parent / "submissions"
        self.submissions_dir.mkdir(exist_ok=True)

    def load_leaderboard(self) -> List[Dict[Any, Any]]:
        """Load current leaderboard data."""
        if not self.leaderboard_path.exists():
            return []

        with open(self.leaderboard_path) as f:
            return json.load(f)

    def add_submission(self, submission_path: Path) -> bool:
        """
        Add a new submission to the leaderboard.

        Returns:
            True if successfully added, False if already exists
        """
        with open(submission_path) as f:
            submission = json.load(f)

        leaderboard = self.load_leaderboard()
        model_id = submission["model_info"]["id"]

        # Check if model already exists
        existing_model = next((m for m in leaderboard if m["id"] == model_id), None)
        if existing_model:
            print(f"Model {model_id} already exists in leaderboard")
            return False

        # Convert submission format to leaderboard format
        leaderboard_entry = self._convert_submission_to_entry(submission)

        # Add to leaderboard
        leaderboard.append(leaderboard_entry)

        # Sort by safety score (descending)
        leaderboard.sort(key=lambda x: x["safety_score"], reverse=True)

        # Save updated leaderboard
        self._save_leaderboard(leaderboard)

        # Archive the submission
        submission_archive_path = self.submissions_dir / f"{model_id}.json"
        shutil.copy2(submission_path, submission_archive_path)

        print(f"✅ Added {model_id} to leaderboard with safety score {leaderboard_entry['safety_score']:.1f}")
        return True

    def _convert_submission_to_entry(self, submission: Dict[Any, Any]) -> Dict[Any, Any]:
        """Convert submission format to leaderboard entry format."""
        model_info = submission["model_info"]
        results = submission["results"]
        evaluation_info = submission["evaluation_info"]

        return {
            "id": model_info["id"],
            "name": model_info["name"],
            "organization": model_info["organization"],
            "type": model_info["type"],
            "safety_score": results["summary"]["safety_score"],
            "spec_compliance": results["summary"]["spec_compliance"],
            "abstention_quality": results["summary"]["abstention_quality"],
            "tasks_completed": results["summary"]["tasks_completed"],
            "suite_scores": {
                suite: data["suite_score"]
                for suite, data in results["suite_results"].items()
            },
            "submission_date": evaluation_info["submission_date"],
            "paper_url": model_info.get("paper_url"),
            "code_url": model_info.get("code_url"),
            "description": model_info["description"],
            "model_details": model_info.get("model_details", {}),
            "detailed_results_url": results.get("detailed_results_url")
        }

    def _save_leaderboard(self, leaderboard: List[Dict[Any, Any]]) -> None:
        """Save leaderboard data with pretty formatting."""
        with open(self.leaderboard_path, 'w') as f:
            json.dump(leaderboard, f, indent=2, sort_keys=True)

    def remove_submission(self, model_id: str) -> bool:
        """Remove a submission from the leaderboard."""
        leaderboard = self.load_leaderboard()
        original_count = len(leaderboard)

        leaderboard = [m for m in leaderboard if m["id"] != model_id]

        if len(leaderboard) == original_count:
            print(f"Model {model_id} not found in leaderboard")
            return False

        self._save_leaderboard(leaderboard)

        # Remove archived submission
        archive_path = self.submissions_dir / f"{model_id}.json"
        if archive_path.exists():
            archive_path.unlink()

        print(f"✅ Removed {model_id} from leaderboard")
        return True

    def update_submission(self, submission_path: Path) -> bool:
        """Update an existing submission."""
        with open(submission_path) as f:
            submission = json.load(f)

        model_id = submission["model_info"]["id"]

        # Remove existing entry
        if not self.remove_submission(model_id):
            print(f"Warning: {model_id} not found for update, adding as new entry")

        # Add updated entry
        return self.add_submission(submission_path)

    def generate_statistics(self) -> Dict[str, Any]:
        """Generate leaderboard statistics."""
        leaderboard = self.load_leaderboard()

        if not leaderboard:
            return {
                "total_models": 0,
                "model_types": {},
                "score_statistics": {},
                "suite_statistics": {}
            }

        # Model type distribution
        type_counts = {}
        for model in leaderboard:
            model_type = model["type"]
            type_counts[model_type] = type_counts.get(model_type, 0) + 1

        # Score statistics
        safety_scores = [m["safety_score"] for m in leaderboard]
        spec_compliance = [m["spec_compliance"] for m in leaderboard]
        abstention_quality = [m["abstention_quality"] for m in leaderboard]

        score_stats = {
            "safety_score": {
                "mean": sum(safety_scores) / len(safety_scores),
                "min": min(safety_scores),
                "max": max(safety_scores),
                "median": sorted(safety_scores)[len(safety_scores) // 2]
            },
            "spec_compliance": {
                "mean": sum(spec_compliance) / len(spec_compliance),
                "min": min(spec_compliance),
                "max": max(spec_compliance)
            },
            "abstention_quality": {
                "mean": sum(abstention_quality) / len(abstention_quality),
                "min": min(abstention_quality),
                "max": max(abstention_quality)
            }
        }

        # Suite-wise statistics
        suite_names = ["basic", "adversarial_safety", "drug_interactions",
                      "organ_toxicity", "admet_optimization",
                      "regulatory_compliance", "therapeutic_window"]

        suite_stats = {}
        for suite in suite_names:
            scores = [m["suite_scores"].get(suite, 0) for m in leaderboard if suite in m["suite_scores"]]
            if scores:
                suite_stats[suite] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "models_evaluated": len(scores)
                }

        return {
            "total_models": len(leaderboard),
            "model_types": type_counts,
            "score_statistics": score_stats,
            "suite_statistics": suite_stats,
            "last_updated": datetime.now().isoformat()
        }

    def export_rankings_csv(self, output_path: Path) -> None:
        """Export current rankings to CSV format."""
        import csv

        leaderboard = self.load_leaderboard()

        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Header
            writer.writerow([
                'Rank', 'Model', 'Organization', 'Type', 'Safety Score',
                'Spec Compliance', 'Abstention Quality', 'Tasks Completed',
                'Basic', 'Adversarial Safety', 'Drug Interactions',
                'Organ Toxicity', 'ADMET Optimization',
                'Regulatory Compliance', 'Therapeutic Window',
                'Submission Date'
            ])

            # Data rows
            for rank, model in enumerate(leaderboard, 1):
                suite_scores = model["suite_scores"]
                writer.writerow([
                    rank,
                    model["name"],
                    model["organization"],
                    model["type"],
                    f"{model['safety_score']:.1f}",
                    f"{model['spec_compliance']:.3f}",
                    f"{model['abstention_quality']:.3f}",
                    model["tasks_completed"],
                    f"{suite_scores.get('basic', 0):.3f}",
                    f"{suite_scores.get('adversarial_safety', 0):.3f}",
                    f"{suite_scores.get('drug_interactions', 0):.3f}",
                    f"{suite_scores.get('organ_toxicity', 0):.3f}",
                    f"{suite_scores.get('admet_optimization', 0):.3f}",
                    f"{suite_scores.get('regulatory_compliance', 0):.3f}",
                    f"{suite_scores.get('therapeutic_window', 0):.3f}",
                    model["submission_date"]
                ])

        print(f"✅ Exported rankings to {output_path}")


def main():
    """Main leaderboard update function."""
    parser = argparse.ArgumentParser(description="Update SpecGuard-Chem leaderboard")
    parser.add_argument("--leaderboard", type=Path,
                       default=Path(__file__).parent.parent / "leaderboard/data/leaderboard.json",
                       help="Path to leaderboard JSON file")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add submission
    add_parser = subparsers.add_parser("add", help="Add new submission")
    add_parser.add_argument("submission", type=Path, help="Path to submission file")

    # Update submission
    update_parser = subparsers.add_parser("update", help="Update existing submission")
    update_parser.add_argument("submission", type=Path, help="Path to updated submission file")

    # Remove submission
    remove_parser = subparsers.add_parser("remove", help="Remove submission")
    remove_parser.add_argument("model_id", help="Model ID to remove")

    # Generate statistics
    stats_parser = subparsers.add_parser("stats", help="Generate statistics")
    stats_parser.add_argument("--output", type=Path, help="Output file for statistics")

    # Export CSV
    csv_parser = subparsers.add_parser("export-csv", help="Export rankings to CSV")
    csv_parser.add_argument("output", type=Path, help="Output CSV file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    updater = LeaderboardUpdater(args.leaderboard)

    if args.command == "add":
        if not args.submission.exists():
            print(f"Error: Submission file {args.submission} does not exist")
            sys.exit(1)
        updater.add_submission(args.submission)

    elif args.command == "update":
        if not args.submission.exists():
            print(f"Error: Submission file {args.submission} does not exist")
            sys.exit(1)
        updater.update_submission(args.submission)

    elif args.command == "remove":
        updater.remove_submission(args.model_id)

    elif args.command == "stats":
        stats = updater.generate_statistics()
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"✅ Statistics saved to {args.output}")
        else:
            print(json.dumps(stats, indent=2))

    elif args.command == "export-csv":
        updater.export_rankings_csv(args.output)


if __name__ == "__main__":
    main()