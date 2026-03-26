# AI declaration:
# Github copilot was used for portions of the planning, research, feedback and editing of the software artefact. Mostly utilised for syntax, logic and error checking with ChatGPT and Claude Sonnet 4.6 used as the models.

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

# discover_scripts
class TestDiscoverScripts: # (Anthropic, 2026)
    """Tests for the discover_scripts() function from utils.py."""

    @pytest.fixture
    def tmp_script_dir(self, tmp_path): # (Anthropic, 2026)
        """A temporary directory populated with a representative mix of file types.

        Contains two regular .py files ('train.py', 'evaluate.py'), one dunder
        file ('__init__.py'), one directory ('__pycache__'), and two non-.py files
        ('readme.md', 'data.csv') to exercise all filtering branches.

        Returns:
          A pathlib.Path pointing to the populated temporary directory.
        """
        (tmp_path / "train.py").touch()
        (tmp_path / "evaluate.py").touch()
        (tmp_path / "__init__.py").touch()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "readme.md").touch()
        (tmp_path / "data.csv").touch()
        return tmp_path

    def test_finds_regular_py_files(self, tmp_script_dir): # (Anthropic, 2026)
        """Asserts that discover_scripts returns all non-dunder .py filenames in the directory."""
        result = discover_scripts(str(tmp_script_dir))
        assert "train.py" in result
        assert "evaluate.py" in result

    def test_excludes_dunder_files(self, tmp_script_dir): # (Anthropic, 2026)
        """Asserts that filenames beginning and ending with double underscores are excluded."""
        result = discover_scripts(str(tmp_script_dir))
        assert "__init__.py" not in result

    def test_excludes_non_py_files(self, tmp_script_dir): # (Anthropic, 2026)
        """Asserts that non-.py files are not included in the result."""
        result = discover_scripts(str(tmp_script_dir))
        assert "readme.md" not in result
        assert "data.csv" not in result

    def test_nonexistent_directory_returns_empty_list(self): # (Anthropic, 2026)
        """Asserts that discover_scripts returns an empty list when the directory does not exist."""
        assert discover_scripts("/this/path/does/not/exist/at/all") == []

    def test_result_is_sorted(self, tmp_script_dir): # (Anthropic, 2026)
        """Asserts that discover_scripts returns filenames in ascending lexicographic order."""
        result = discover_scripts(str(tmp_script_dir))
        assert result == sorted(result)

# discover_csvs
class TestDiscoverCsvs: # (Anthropic, 2026)
    """Tests for the discover_csvs() function from utils.py."""

    @pytest.fixture
    def tmp_csv_dir(self, tmp_path): # (Anthropic, 2026)
        """A temporary directory containing two .csv files and two non-.csv files.

        Returns:
          A pathlib.Path pointing to the populated temporary directory.
        """
        (tmp_path / "dataset_a.csv").touch()
        (tmp_path / "dataset_b.csv").touch()
        (tmp_path / "notes.txt").touch()
        (tmp_path / "checkpoint.pt").touch()
        return tmp_path

    def test_finds_csv_files(self, tmp_csv_dir): # (Anthropic, 2026)
        """Asserts that discover_csvs returns all .csv filenames present in the directory."""
        result = discover_csvs(str(tmp_csv_dir))
        assert "dataset_a.csv" in result
        assert "dataset_b.csv" in result

    def test_excludes_non_csv_files(self, tmp_csv_dir): # (Anthropic, 2026)
        """Asserts that non-.csv files are not included in the result."""
        result = discover_csvs(str(tmp_csv_dir))
        assert "notes.txt" not in result
        assert "checkpoint.pt" not in result

    def test_nonexistent_directory_returns_empty_list(self): # (Anthropic, 2026)
        """Asserts that discover_csvs returns an empty list when the directory does not exist."""
        assert discover_csvs("/no/such/directory") == []

# Discover models
class TestDiscoverModels: # (Anthropic, 2026)
    """Tests for the discover_models() function from utils.py."""

    @pytest.fixture
    def tmp_model_dir(self, tmp_path): # (Anthropic, 2026)
        """A temporary directory containing two .pt model files and two non-.pt files.

        Returns:
          A pathlib.Path pointing to the populated temporary directory.
        """
        (tmp_path / "xlstm_run1.pt").touch()
        (tmp_path / "memd_run1.pt").touch()
        (tmp_path / "notes.txt").touch()
        (tmp_path / "data.csv").touch()
        return tmp_path

    def test_finds_pt_files(self, tmp_model_dir): # (Anthropic, 2026)
        """Asserts that discover_models returns all .pt filenames present in the directory."""
        result = discover_models(str(tmp_model_dir))
        assert "xlstm_run1.pt" in result
        assert "memd_run1.pt" in result

    def test_excludes_non_pt_files(self, tmp_model_dir): # (Anthropic, 2026)
        """Asserts that non-.pt files are not included in the result."""
        result = discover_models(str(tmp_model_dir))
        assert "notes.txt" not in result
        assert "data.csv" not in result

    def test_nonexistent_directory_returns_empty_list(self): # (Anthropic, 2026)
        """Asserts that discover_models returns an empty list when the directory does not exist."""
        assert discover_models("/no/such/directory") == []

# Pipeline stages 
class TestPipelineStages: # (Anthropic, 2026)
    """Tests for the PIPELINE_STAGES constant defined in constants.py."""

    def test_exactly_five_stages(self): # (Anthropic, 2026)
        """Asserts that PIPELINE_STAGES defines exactly 5 stages, matching the UI design."""
        assert len(PIPELINE_STAGES) == 5

    def test_each_stage_is_a_four_tuple(self): # (Anthropic, 2026)
        """Asserts that every entry in PIPELINE_STAGES has exactly 4 elements."""
        for stage in PIPELINE_STAGES:
            assert len(stage) == 4, f"Stage is not a 4-tuple: {stage}"

    def test_stage_labels_are_non_empty_strings(self): # (Anthropic, 2026)
        """Asserts that the label field of every stage is a non-empty string."""
        for label, _path, _multi, _diag in PIPELINE_STAGES:
            assert isinstance(label, str) and label.strip() != ""

    def test_multi_select_field_is_bool(self): # (Anthropic, 2026)
        """Asserts that the multi-select field of every stage is a bool, not a truthy int."""
        for _label, _path, multi, _diag in PIPELINE_STAGES:
            assert isinstance(multi, bool), (
                f"multi_select for stage '{_label}' is {type(multi)}, expected bool"
            )

    def test_ai_model_designs_is_single_select(self): # (Anthropic, 2026)
        """Asserts that the 'AI Model Designs' stage uses single-select (multi=False).

        Running multiple model-design scripts simultaneously is not a supported
        workflow, so the multi-select flag for this stage must be False.
        """
        stage = next(s for s in PIPELINE_STAGES if s[0] == "AI Model Designs")
        assert stage[2] is False

    def test_all_stage_paths_are_strings(self): # (Anthropic, 2026)
        """Asserts that the path field of every stage is a non-empty string."""
        for _label, path, _multi, _diag in PIPELINE_STAGES:
            assert isinstance(path, str) and path != ""

# Root_dir and Freq_Options
class TestConstants: # (Anthropic, 2026)
    """Tests for ROOT_DIR and FREQ_OPTIONS constants defined in constants.py."""

    def test_root_dir_is_an_existing_directory(self): # (Anthropic, 2026)
        """Asserts that ROOT_DIR resolves to a real, existing directory on disk."""
        assert os.path.isdir(ROOT_DIR), (
            f"ROOT_DIR '{ROOT_DIR}' does not point to a real directory"
        )

    def test_root_dir_contains_model_designer(self): # (Anthropic, 2026)
        """Asserts that Model_Designer.py exists at the path given by ROOT_DIR."""
        assert os.path.isfile(os.path.join(ROOT_DIR, "Model_Designer.py"))

    def test_freq_options_contains_standard_intervals(self): # (Anthropic, 2026)
        """Asserts that FREQ_OPTIONS includes all standard trading frequency strings."""
        for freq in ("1m", "1h", "1d", "1wk"):
            assert freq in FREQ_OPTIONS, f"Expected '{freq}' in FREQ_OPTIONS"

    def test_freq_options_intraday_before_daily(self): # (Anthropic, 2026)
        """Asserts that intraday frequencies appear before daily frequency in FREQ_OPTIONS."""
        assert FREQ_OPTIONS.index("1h") < FREQ_OPTIONS.index("1d")
