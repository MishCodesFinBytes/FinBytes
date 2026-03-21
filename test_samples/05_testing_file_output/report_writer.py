"""
report_writer.py — Production module used in:
"Testing File Output in Python Without Touching the Filesystem"
"""

import pandas as pd


class ReportWriter:
    def save_report(self, df: pd.DataFrame, file_path: str):
        df.to_excel(file_path, index=False)

    def _write_excel(self, df: pd.DataFrame, path: str):
        """Thin wrapper — makes patching cleaner in tests."""
        df.to_excel(path, index=False)

    def save_report_via_wrapper(self, df: pd.DataFrame, path: str):
        self._write_excel(df, path)
