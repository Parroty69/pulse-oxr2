from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "src" / "ui" / "app.py"


def test_ui_imports_from_outside_repository(tmp_path: Path) -> None:
    test_script = textwrap.dedent(
        f"""
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file({str(APP_PATH)!r}).run(timeout=10)
        if app.exception:
            raise RuntimeError("; ".join(exception.message for exception in app.exception))
        """
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", test_script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
