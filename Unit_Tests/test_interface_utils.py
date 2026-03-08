"""
Unit tests for Interface_Modules/utils.py and Interface_Modules/constants.py

Covers:
  - discover_scripts: finds .py files, excludes dunders, handles missing dirs
  - discover_csvs:    finds .csv files, handles missing dirs
  - discover_models:  finds .pt files, handles missing dirs
  - PIPELINE_STAGES:  correct count, 4-tuple structure, semantic constraints
  - ROOT_DIR:         resolves to the real project root on disk
  - FREQ_OPTIONS:     contains expected frequency strings in the right order
"""

import os

import pytest

from constants import FREQ_OPTIONS, PIPELINE_STAGES, ROOT_DIR
from utils import discover_csvs, discover_models, discover_scripts


# ── discover_scripts ──────────────────────────────────────────────────────────

class TestDiscoverScripts:

    @pytest.fixture
    def tmp_script_dir(self, tmp_path):
        """Temp directory containing regular .py scripts, a dunder file, and non-.py files."""
        (tmp_path / "train.py").touch()
        (tmp_path / "evaluate.py").touch()
        (tmp_path / "__init__.py").touch()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "readme.md").touch()
        (tmp_path / "data.csv").touch()
        return tmp_path

    def test_finds_regular_py_files(self, tmp_script_dir):
        result = discover_scripts(str(tmp_script_dir))
        assert "train.py" in result
        assert "evaluate.py" in result

    def test_excludes_dunder_files(self, tmp_script_dir):
        """__init__.py and other dunder files must be silently excluded."""
        result = discover_scripts(str(tmp_script_dir))
        assert "__init__.py" not in result

    def test_excludes_non_py_files(self, tmp_script_dir):
        result = discover_scripts(str(tmp_script_dir))
        assert "readme.md" not in result
        assert "data.csv" not in result

    def test_nonexistent_directory_returns_empty_list(self):
        assert discover_scripts("/this/path/does/not/exist/at/all") == []

    def test_result_is_sorted(self, tmp_script_dir):
        result = discover_scripts(str(tmp_script_dir))
        assert result == sorted(result)


# ── discover_csvs ─────────────────────────────────────────────────────────────

class TestDiscoverCsvs:

    @pytest.fixture
    def tmp_csv_dir(self, tmp_path):
        (tmp_path / "dataset_a.csv").touch()
        (tmp_path / "dataset_b.csv").touch()
        (tmp_path / "notes.txt").touch()
        (tmp_path / "checkpoint.pt").touch()
        return tmp_path

    def test_finds_csv_files(self, tmp_csv_dir):
        result = discover_csvs(str(tmp_csv_dir))
        assert "dataset_a.csv" in result
        assert "dataset_b.csv" in result

    def test_excludes_non_csv_files(self, tmp_csv_dir):
        result = discover_csvs(str(tmp_csv_dir))
        assert "notes.txt" not in result
        assert "checkpoint.pt" not in result

    def test_nonexistent_directory_returns_empty_list(self):
        assert discover_csvs("/no/such/directory") == []


# ── discover_models ───────────────────────────────────────────────────────────

class TestDiscoverModels:

    @pytest.fixture
    def tmp_model_dir(self, tmp_path):
        (tmp_path / "xlstm_run1.pt").touch()
        (tmp_path / "memd_run1.pt").touch()
        (tmp_path / "notes.txt").touch()
        (tmp_path / "data.csv").touch()
        return tmp_path

    def test_finds_pt_files(self, tmp_model_dir):
        result = discover_models(str(tmp_model_dir))
        assert "xlstm_run1.pt" in result
        assert "memd_run1.pt" in result

    def test_excludes_non_pt_files(self, tmp_model_dir):
        result = discover_models(str(tmp_model_dir))
        assert "notes.txt" not in result
        assert "data.csv" not in result

    def test_nonexistent_directory_returns_empty_list(self):
        assert discover_models("/no/such/directory") == []


# ── PIPELINE_STAGES ───────────────────────────────────────────────────────────

class TestPipelineStages:

    def test_exactly_five_stages(self):
        """The pipeline must define exactly 5 stages matching the UI design."""
        assert len(PIPELINE_STAGES) == 5

    def test_each_stage_is_a_four_tuple(self):
        for stage in PIPELINE_STAGES:
            assert len(stage) == 4, f"Stage is not a 4-tuple: {stage}"

    def test_stage_labels_are_non_empty_strings(self):
        for label, _path, _multi, _diag in PIPELINE_STAGES:
            assert isinstance(label, str) and label.strip() != ""

    def test_multi_select_field_is_bool(self):
        for _label, _path, multi, _diag in PIPELINE_STAGES:
            assert isinstance(multi, bool), (
                f"multi_select for stage '{_label}' is {type(multi)}, expected bool"
            )

    def test_ai_model_designs_is_single_select(self):
        """The 'AI Model Designs' stage must be single-select; running multiple
        model scripts simultaneously is not a supported workflow."""
        stage = next(s for s in PIPELINE_STAGES if s[0] == "AI Model Designs")
        assert stage[2] is False

    def test_all_stage_paths_are_strings(self):
        for _label, path, _multi, _diag in PIPELINE_STAGES:
            assert isinstance(path, str) and path != ""


# ── ROOT_DIR and FREQ_OPTIONS ─────────────────────────────────────────────────

class TestConstants:

    def test_root_dir_is_an_existing_directory(self):
        assert os.path.isdir(ROOT_DIR), (
            f"ROOT_DIR '{ROOT_DIR}' does not point to a real directory"
        )

    def test_root_dir_contains_model_designer(self):
        """The project entry-point must be present at the resolved root."""
        assert os.path.isfile(os.path.join(ROOT_DIR, "Model_Designer.py"))

    def test_freq_options_contains_standard_intervals(self):
        for freq in ("1m", "1h", "1d", "1wk"):
            assert freq in FREQ_OPTIONS, f"Expected '{freq}' in FREQ_OPTIONS"

    def test_freq_options_intraday_before_daily(self):
        """Intraday frequencies must precede daily in the ordered list."""
        assert FREQ_OPTIONS.index("1h") < FREQ_OPTIONS.index("1d")
