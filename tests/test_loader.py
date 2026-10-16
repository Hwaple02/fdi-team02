import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from load_training_data import PERIODS, VARIABLES, load_training_data, make_training_targets


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for period, (start, end) in PERIODS.items():
            for variable in VARIABLES:
                path = self.root / f"{period}__{variable}.csv"
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["month", "000020", "000040"])
                    for month in pd.period_range(start, end, freq="M").astype(str):
                        values = ["0.1", "0.2"]
                        if variable == "company_name":
                            values = ["회사\nA", "NA"]
                        elif variable == "isin":
                            values = ["KR7000020008", "KR7000040006"]
                        elif variable == "adjusted_monthly_return" and month == "2001-02":
                            values = ["", "0"]
                        writer.writerow([month, *values])

    def test_allowlist_identifiers_and_missing(self):
        # A test file exists but any access to it must fail the test.
        (self.root / "test_2022_2025__adjusted_monthly_return.csv").write_text("do not read")
        original = Path.read_bytes
        opened = []

        def guarded(path):
            self.assertFalse(path.name.startswith("test_"))
            opened.append(path.name)
            return original(path)

        with patch.object(Path, "read_bytes", guarded):
            result = load_training_data(self.root, 2)
        self.assertEqual(len(opened), 12)
        self.assertEqual(len(result.panel), 264 * 2)
        self.assertEqual(result.panel.stock_code.unique().tolist(), ["000020", "000040"])
        self.assertIn("NA", result.panel.company_name.tolist())
        self.assertIn("회사\nA", result.panel.company_name.tolist())
        month = result.panel[result.panel.month == "2001-02"].set_index("stock_code")
        self.assertTrue(pd.isna(month.loc["000020", "adjusted_monthly_return"]))
        self.assertEqual(month.loc["000040", "adjusted_monthly_return"], 0)

    def test_calendar_labels_and_train_boundary(self):
        panel = load_training_data(self.root, 2).panel
        # Remove a complete calendar month to ensure no next-observation shift.
        panel = panel[panel.month != "2001-02"]
        target = make_training_targets(panel)
        self.assertFalse(target[target.month == "2001-01"].has_target.any())
        self.assertEqual(target.target_month.min(), "2001-01")
        self.assertEqual(target.target_month.max(), "2021-12")
        self.assertFalse((target.month == "2021-12").any())
        first = target[target.target_month == "2001-01"].sort_values("stock_code")
        self.assertEqual(first.target_rank.tolist(), [0.5, 1.0])

    def test_ties_and_missing_target(self):
        panel = load_training_data(self.root, 2).panel
        panel.loc[panel.month == "2001-03", "adjusted_monthly_return"] = 0.2
        target = make_training_targets(panel)
        self.assertEqual(target[target.target_month == "2001-03"].target_rank.tolist(), [0.75, 0.75])
        missing = target[(target.target_month == "2001-02") & (target.stock_code == "000020")]
        self.assertTrue(missing.target_rank.isna().all())

    def test_duplicate_code_rejected(self):
        path = self.root / "warmup_2000__adjusted_monthly_return.csv"
        path.write_text(path.read_text().replace("month,000020,000040", "month,000020,000020"))
        with self.assertRaisesRegex(ValueError, "중복"):
            load_training_data(self.root, 2)

    def test_missing_month_rejected(self):
        path = self.root / "warmup_2000__adjusted_monthly_return.csv"
        path.write_text(path.read_text().replace("2000-02,0.1,0.2\n", ""))
        with self.assertRaisesRegex(ValueError, "월 누락"):
            load_training_data(self.root, 2)

    def test_misaligned_header_rejected(self):
        path = self.root / "train_2001_2021__isin.csv"
        path.write_text(path.read_text().replace("month,000020,000040", "month,000040,000020"))
        with self.assertRaisesRegex(ValueError, "순서 불일치"):
            load_training_data(self.root, 2)

    def test_invalid_number_rejected(self):
        path = self.root / "warmup_2000__adjusted_monthly_return.csv"
        path.write_text(path.read_text().replace("2000-01,0.1", "2000-01,unknown"))
        with self.assertRaisesRegex(ValueError, "비수치"):
            load_training_data(self.root, 2)

    def test_test_period_panel_rejected(self):
        panel = load_training_data(self.root, 2).panel
        panel.loc[0, "month"] = "2022-01"
        with self.assertRaisesRegex(ValueError, "기간 밖"):
            make_training_targets(panel)


if __name__ == "__main__":
    unittest.main()
