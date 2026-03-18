"""
Entry point for the Security Intelligence System.

Run:
    python main.py
"""

from app.ui.main_window import MainWindow

if __name__ == "__main__":
    app = MainWindow()
    app.run()
