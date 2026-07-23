from __future__ import annotations

import sys

import pytest

from simulator import cli
from simulator.config import Config
from simulator.dataset_state import DatasetConfigMismatchError
from simulator.empirical_maps import EmpiricalMapStore


def test_cli_reports_completed_dataset_mismatch_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cfg = Config(out_dir=str(tmp_path), recomb_map="", mut_map="")
    monkeypatch.setattr(cli, "parse_args", lambda: cfg)
    monkeypatch.setattr(cli, "load_empirical_maps", lambda _cfg: EmpiricalMapStore())

    def reject_reuse(_cfg, _empirical, **_kwargs):
        raise DatasetConfigMismatchError("configuration changed; rerun with --force")

    monkeypatch.setattr(cli, "generate_dataset", reject_reuse)

    with pytest.raises(SystemExit, match=r"^\[error\].*--force$"):
        cli.main()


def test_cli_rejects_empirical_only_source_when_maps_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cfg = Config(
        out_dir=str(tmp_path),
        recomb_map="",
        mut_map="",
        p_clean_constant=0.0,
        p_random_global=0.0,
        p_custom_hetero=0.0,
        p_empirical_slice=1.0,
        p_stdpopsim_anchor=0.0,
    )
    monkeypatch.setattr(cli, "parse_args", lambda: cfg)
    monkeypatch.setattr(cli, "load_empirical_maps", lambda _cfg: EmpiricalMapStore())

    with pytest.raises(SystemExit, match="only enabled source"):
        cli.main()


def test_unknown_preset_is_an_argparse_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["dlcoalsim", "--preset", "unknown"])

    with pytest.raises(SystemExit) as exc_info:
        cli.parse_args()

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "Unknown preset" in stderr
    assert "Traceback" not in stderr


def test_unusable_output_path_has_a_clean_cli_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("occupied", encoding="utf-8")
    cfg = Config(
        out_dir=str(parent_file / "dataset"),
        recomb_map="",
        mut_map="",
    )
    monkeypatch.setattr(cli, "parse_args", lambda: cfg)

    with pytest.raises(SystemExit, match=r"^\[error\] filesystem operation failed"):
        cli.main()
