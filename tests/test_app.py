"""Headless dashboard test (Streamlit AppTest) on real nuScenes data, all three modes, CPU."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import requires_dataset

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _app(mode_index: int, explain: bool = False):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    at.sidebar.selectbox[1].set_value("cpu").run()          # device
    at.sidebar.radio[0].set_value(at.sidebar.radio[0].options[mode_index]).run()
    if explain:
        next(b for b in at.button if "Explain" in b.label).click().run()
    return at


@requires_dataset
@pytest.mark.parametrize("mode_index", [0, 1, 2])
def test_dashboard_modes_render_without_exceptions(mode_index: int) -> None:
    at = _app(mode_index, explain=mode_index == 2)
    assert not at.exception, [e.value for e in at.exception]
    assert any("MULTI-SENSOR AUTONOMOUS DRIVING SYSTEM" in m.value for m in at.markdown)


@requires_dataset
def test_dashboard_next_frame() -> None:
    at = _app(1)
    next(b for b in at.button if "Next" in b.label).click().run()
    assert not at.exception
    assert at.slider[0].value == 1
