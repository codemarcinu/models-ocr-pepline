#!/usr/bin/env python3
"""
Test Data Setup
Prepares directories and validates ground truth annotations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDataValidator:
    """Validate and set up test data structure."""

    REQUIRED_FIELDS = {
        "merchant_name": str,
        "date": str,
        "time": str,
        "total_amount": float,
        "tax_amount": float,
        "items": list,
        "payment_method": str,
    }

    ITEM_FIELDS = {
        "description": str,
        "quantity": (int, float),
        "unit_price": float,
        "total": float,
    }

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.test_receipts_dir = self.base_dir / "test_receipts"
        self.ground_truth_dir = self.base_dir / "ground_truth"
        self.results_dir = self.base_dir / "results"

    def setup_directories(self):
        """Create required directory structure."""
        dirs = [
            self.test_receipts_dir,
            self.ground_truth_dir,
            self.results_dir,
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")

    def validate_ground_truth_file(self, file_path: Path) -> bool:
        """Validate ground truth JSON file."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Check required fields
            for field, field_type in self.REQUIRED_FIELDS.items():
                if field not in data:
                    logger.error(f"Missing required field: {field}")
                    return False

                if field_type == str and not isinstance(data[field], str):
                    logger.error(f"Field {field} must be string")
                    return False

                if field_type == float:
                    try:
                        float(data[field])
                    except (ValueError, TypeError):
                        logger.error(f"Field {field} must be numeric")
                        return False

                if field_type == list:
                    if not isinstance(data[field], list):
                        logger.error(f"Field {field} must be list")
                        return False

                    # Validate items
                    for i, item in enumerate(data[field]):
                        for item_field, item_type in self.ITEM_FIELDS.items():
                            if item_field not in item:
                                logger.warning(
                                    f"Item {i} missing field: {item_field}"
                                )

            # Validate date format
            try:
                datetime.strptime(data["date"], "%Y-%m-%d")
            except ValueError:
                logger.error(f"Date must be YYYY-MM-DD format, got: {data['date']}")
                return False

            # Validate time format
            try:
                datetime.strptime(data["time"], "%H:%M")
            except ValueError:
                logger.warning(f"Time should be HH:MM format, got: {data['time']}")

            # Validate amounts
            if data["total_amount"] < 0:
                logger.error("total_amount cannot be negative")
                return False

            if data["tax_amount"] < 0:
                logger.error("tax_amount cannot be negative")
                return False

            # Validate items sum
            items_total = sum(
                float(item.get("total", item.get("price", 0)))
                for item in data.get("items", [])
            )
            receipt_total = float(data["total_amount"])

            # Allow 1% tolerance for rounding
            if receipt_total > 0:
                difference = abs(items_total - receipt_total)
                tolerance = receipt_total * 0.01
                if difference > tolerance:
                    logger.warning(
                        f"Items total ({items_total}) differs from receipt total "
                        f"({receipt_total}) by {difference} (tolerance: {tolerance})"
                    )

            logger.info(f"Validated: {file_path}")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Validation error for {file_path}: {e}")
            return False

    def validate_all_ground_truth(self) -> bool:
        """Validate all ground truth files."""
        json_files = list(self.ground_truth_dir.glob("*.json"))

        if not json_files:
            logger.warning(f"No JSON files found in {self.ground_truth_dir}")
            return False

        valid_count = 0
        for json_file in json_files:
            if self.validate_ground_truth_file(json_file):
                valid_count += 1

        logger.info(f"Valid files: {valid_count}/{len(json_files)}")
        return valid_count == len(json_files)

    def generate_sample_ground_truth(self, num_samples: int = 5):
        """Generate sample ground truth files for testing."""
        merchants = [
            "Tesco Express",
            "Sainsbury's Local",
            "Costa Coffee",
            "McDonald's",
            "Boots Pharmacy",
            "Pret A Manger",
            "Starbucks",
            "Waitrose",
        ]

        items_pool = [
            ("Milk 1L", 1, 1.20),
            ("Bread", 1, 1.50),
            ("Eggs x6", 2, 2.00),
            ("Coffee", 1, 3.50),
            ("Sandwich", 1, 3.99),
            ("Banana bunch", 1, 0.50),
            ("Cheese", 1, 2.50),
            ("Yogurt", 2, 0.80),
            ("Tea bags", 1, 2.00),
            ("Olive oil", 1, 4.50),
        ]

        for i in range(num_samples):
            receipt_id = f"receipt_{i+1:03d}"
            merchant = random.choice(merchants)

            # Generate items
            num_items = random.randint(2, 6)
            selected_items = random.sample(items_pool, num_items)

            items = []
            items_total = 0
            for description, quantity, unit_price in selected_items:
                item_total = quantity * unit_price
                items_total += item_total
                items.append(
                    {
                        "description": description,
                        "quantity": quantity,
                        "unit_price": unit_price,
                        "total": round(item_total, 2),
                    }
                )

            # Calculate tax (20% UK VAT)
            tax = round(items_total * 0.20, 2)
            total = round(items_total + tax, 2)

            # Generate date/time
            random_days_ago = random.randint(0, 30)
            receipt_date = (datetime.now() - timedelta(days=random_days_ago)).date()
            receipt_time = f"{random.randint(8, 20):02d}:{random.randint(0, 59):02d}"

            ground_truth = {
                "merchant_name": merchant,
                "date": str(receipt_date),
                "time": receipt_time,
                "total_amount": total,
                "tax_amount": tax,
                "items": items,
                "payment_method": random.choice(["card", "cash", "phone"]),
                "raw_text": f"[OCR text for {receipt_id}]",  # Placeholder
            }

            # Save ground truth
            gt_file = self.ground_truth_dir / f"{receipt_id}.json"
            with open(gt_file, "w") as f:
                json.dump(ground_truth, f, indent=2)

            logger.info(f"Generated ground truth: {receipt_id}")

            # Validate
            self.validate_ground_truth_file(gt_file)

        logger.info(f"Generated {num_samples} sample ground truth files")

    def generate_report(self) -> str:
        """Generate setup report."""
        report = []
        report.append("\n" + "="*60)
        report.append("TEST DATA SETUP REPORT")
        report.append("="*60)
        report.append("")

        # Count files
        test_receipts = list(self.test_receipts_dir.glob("*.*"))
        ground_truths = list(self.ground_truth_dir.glob("*.json"))

        report.append(f"Test Receipts: {len(test_receipts)} files")
        for f in test_receipts[:5]:
            report.append(f"  - {f.name}")
        if len(test_receipts) > 5:
            report.append(f"  ... and {len(test_receipts) - 5} more")

        report.append("")
        report.append(f"Ground Truth Annotations: {len(ground_truths)} files")
        for f in ground_truths[:5]:
            report.append(f"  - {f.name}")
        if len(ground_truths) > 5:
            report.append(f"  ... and {len(ground_truths) - 5} more")

        report.append("")

        # Matching
        receipt_names = {f.stem for f in test_receipts}
        gt_names = {f.stem for f in ground_truths}
        matched = receipt_names & gt_names
        missing_gt = receipt_names - gt_names
        missing_images = gt_names - receipt_names

        report.append(f"Matched Pairs: {len(matched)}/{max(len(test_receipts), len(ground_truths))}")
        if missing_gt:
            report.append(f"\nMissing Ground Truth:")
            for name in list(missing_gt)[:5]:
                report.append(f"  - {name}")
            if len(missing_gt) > 5:
                report.append(f"  ... and {len(missing_gt) - 5} more")

        if missing_images:
            report.append(f"\nMissing Images:")
            for name in list(missing_images)[:5]:
                report.append(f"  - {name}")
            if len(missing_images) > 5:
                report.append(f"  ... and {len(missing_images) - 5} more")

        report.append("")
        report.append("RECOMMENDATIONS:")
        if not test_receipts:
            report.append("  1. Add receipt images to test_receipts/")
        if len(matched) < len(test_receipts):
            report.append(f"  2. Create ground truth annotations for {len(missing_gt)} receipts")
        if len(matched) == len(test_receipts):
            report.append("  ✓ Ready to run benchmarks")

        report.append("")
        report.append("="*60)
        report.append("")

        return "\n".join(report)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Setup test data for benchmarking")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory"
    )
    parser.add_argument(
        "--generate-samples",
        type=int,
        help="Generate N sample ground truth files"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate all ground truth files"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate setup report"
    )

    args = parser.parse_args()

    validator = TestDataValidator(args.base_dir)

    # Setup directories
    validator.setup_directories()

    # Generate samples if requested
    if args.generate_samples:
        validator.generate_sample_ground_truth(args.generate_samples)

    # Validate if requested
    if args.validate:
        validator.validate_all_ground_truth()

    # Generate report
    if args.report or True:  # Always show report
        print(validator.generate_report())


if __name__ == "__main__":
    main()
