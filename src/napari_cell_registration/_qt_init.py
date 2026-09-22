"""Qt startup tweaks for the napari plugin."""

from __future__ import annotations

import sys


_INSTALLED = False
_PREVIOUS_QT_HANDLER = None

_SUPPRESSED_QT_MESSAGES = (
    "QWindowsWindow::setGeometry: Unable to set geometry",
    "DirectWrite: CreateFontFaceFromHDC() failed",
)


def configure_qt() -> None:
    """Apply small Windows/Qt fixes without requiring Qt in headless imports."""
    global _INSTALLED, _PREVIOUS_QT_HANDLER
    if _INSTALLED:
        return

    try:
        from qtpy.QtCore import qInstallMessageHandler
    except Exception:
        return

    _PREVIOUS_QT_HANDLER = qInstallMessageHandler(_qt_message_handler)
    _INSTALLED = True
    apply_default_font()


def apply_default_font() -> None:
    """Set a known-good Windows font to avoid DirectWrite fallback warnings."""
    try:
        from qtpy.QtGui import QFont, QFontDatabase
        from qtpy.QtWidgets import QApplication
    except Exception:
        return

    app = QApplication.instance()
    if app is None:
        return

    try:
        families = set(QFontDatabase().families())
    except TypeError:
        families = set(QFontDatabase.families())
    except Exception:
        return
    for family in ("Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", "Arial"):
        if family in families:
            app.setFont(QFont(family, app.font().pointSize()))
            return


def _qt_message_handler(mode, context, message: str) -> None:
    if any(pattern in message for pattern in _SUPPRESSED_QT_MESSAGES):
        return

    if _PREVIOUS_QT_HANDLER is not None:
        _PREVIOUS_QT_HANDLER(mode, context, message)
        return

    print(message, file=sys.stderr)
