from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_demo_dashboard_discloses_synthetic_mode(monkeypatch):
    monkeypatch.setenv("ARGUS_MODE", "demo")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/streamlit_app.py")).run(
        timeout=15
    )
    assert not app.exception
    assert any("synthetic" in warning.value for warning in app.warning)
