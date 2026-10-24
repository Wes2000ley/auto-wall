"""Bundled model selection stays local and does not initialize ML at startup."""

import builtins
from dataclasses import replace

import pytest

from src.core.fast_pipeline import FastSettings
from src.gui import fast_setup


@pytest.fixture
def model_directory(monkeypatch, tmp_path):
    for filename, _, _ in fast_setup.BUNDLED_MODELS:
        # Discovery must not parse or load even the selected checkpoint.
        (tmp_path / filename).write_bytes(b"not loaded during discovery")
    monkeypatch.setattr(fast_setup, "BUNDLED_MODEL_DIRECTORY", tmp_path)
    return tmp_path


def test_model_list_needs_no_ml_imports(qt_app, model_directory, monkeypatch):
    original_import = builtins.__import__

    def without_ml(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "safetensors"}:
            pytest.fail("Building the model list must not import ML libraries")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_ml)
    dialog = fast_setup.FastSetupDialog()
    try:
        assert dialog.model_choices.count() == 5
        assert dialog.model_choices.currentData() == ""
        assert dialog.settings().model_path == ""
        assert [dialog.model_choices.itemData(index) for index in range(1, 5)] == [
            str(model_directory / filename) for filename, _, _ in fast_setup.BUNDLED_MODELS
        ]
    finally:
        dialog.close()


def test_unavailable_bundled_models_are_not_offered(qt_app, model_directory):
    missing = model_directory / fast_setup.BUNDLED_MODELS[0][0]
    missing.unlink()
    dialog = fast_setup.FastSetupDialog()
    try:
        assert dialog.model_choices.count() == 4
        assert dialog.model_choices.findData(str(missing)) == -1
    finally:
        dialog.close()


@pytest.mark.parametrize("model_index", range(4))
def test_selecting_bundled_model_sets_compatible_classes(qt_app, model_directory, model_index):
    dialog = fast_setup.FastSetupDialog()
    settings = replace(FastSettings(), model_path="custom.safetensors", ml_profile="windows",
                       ml_wall_confidence=.96, ml_recover_jambs=True, max_dimension=1536)
    dialog.apply_settings(settings)
    selected = str(model_directory / fast_setup.BUNDLED_MODELS[model_index][0])
    try:
        dialog.model_choices.setCurrentIndex(dialog.model_choices.findData(selected))
        assert dialog.settings() == replace(settings, model_path=selected, ml_profile="legacy")
        assert dialog.model_path.text() == selected
        assert dialog.model_description.text() == fast_setup.BUNDLED_MODELS[model_index][2]
    finally:
        dialog.close()


def test_relative_launcher_path_and_custom_imports_roundtrip(qt_app, model_directory, monkeypatch):
    monkeypatch.chdir(model_directory)
    dialog = fast_setup.FastSetupDialog()
    relative_path = fast_setup.BUNDLED_MODELS[0][0]
    settings = replace(FastSettings(), model_path=relative_path, ml_recover_jambs=True)
    try:
        dialog.apply_settings(settings)
        assert dialog.settings() == settings
        assert dialog.model_choices.currentData() == str(model_directory / relative_path)
        for path in ("first.safetensors", "different.雪.safetensors"):
            custom = replace(settings, model_path=path, ml_profile="windows")
            dialog.apply_settings(custom)
            assert dialog.settings() == custom
            assert dialog.model_choices.currentData() == path
            assert dialog.model_choices.count() == 6
        dialog.apply_settings(settings)
        assert dialog.model_choices.count() == 5
        assert dialog.settings() == settings
    finally:
        dialog.close()


def test_browsing_and_clearing_models_updates_selection(qt_app, model_directory, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    dialog = fast_setup.FastSetupDialog()
    settings = replace(FastSettings(), model_path="custom.safetensors", ml_profile="windows")
    dialog.apply_settings(settings)
    selected = str(model_directory / fast_setup.BUNDLED_MODELS[1][0])
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (selected, ""))
    try:
        dialog.choose_model()
        assert dialog.model_choices.currentData() == selected
        assert dialog.settings() == replace(settings, model_path=selected, ml_profile="legacy")
        dialog.apply_settings(settings)
        dialog.clear_model()
        assert dialog.model_choices.currentIndex() == 0
        assert dialog.settings() == replace(settings, model_path="", ml_profile="legacy")
        assert dialog.model_choices.count() == 5
    finally:
        dialog.close()


def test_model_details_are_readable_in_the_ml_tab(qt_app, model_directory):
    dialog = fast_setup.FastSetupDialog()
    dialog.model_choices.setCurrentIndex(1)
    dialog.tabs.setCurrentIndex(next(index for index in range(dialog.tabs.count())
                                     if dialog.tabs.tabText(index) == "ML"))
    dialog.show()
    qt_app.processEvents()
    try:
        assert dialog.model_description.height() >= dialog.model_description.heightForWidth(
            dialog.model_description.width())
        scroll = dialog.tabs.currentWidget()
        assert scroll.horizontalScrollBar().maximum() == 0
        scroll.ensureWidgetVisible(dialog.ml_recover_jambs)
        qt_app.processEvents()
        bounds = scroll.viewport().rect()
        assert bounds.contains(dialog.ml_recover_jambs.mapTo(scroll.viewport(),
                                                            dialog.ml_recover_jambs.rect().center()))
    finally:
        dialog.close()
