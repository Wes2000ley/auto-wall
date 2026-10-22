import os
from pathlib import Path
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def qt_app():
    from PyQt6.QtGui import QFont, QFontDatabase
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    # Windows' offscreen plugin has no system font discovery. Real windows do.
    font = Path("C:/Windows/Fonts/segoeui.ttf")
    if not QFontDatabase.families() and font.exists():
        QFontDatabase.addApplicationFont(str(font))
        app.setFont(QFont("Segoe UI", 9))
    return app
