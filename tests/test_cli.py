from __future__ import annotations

import sys

import pytest

from necast.simulator import cli
from necast.simulator.config import Config
from necast.simulator.dataset_state import DatasetConfigMismatchError
from necast.simulator.empirical_maps import EmpiricalMapStore
from necast.simulator.shards import (
    DatasetGenerationIncompleteError,
    DatasetGenerationLockedError,
    dataset_output_lock,
)


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


def test_cli_rejects_enabled_empirical_source_when_maps_are_unavailable(
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

    with pytest.raises(
        SystemExit,
        match="refusing to disable the requested empirical source",
    ):
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


def test_parse_args_records_distributed_generation_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dlcoalsim",
            "--generate-only",
            "--rank",
            "2",
            "--world-size",
            "5",
            "--out-dir",
            str(tmp_path),
        ],
    )

    cfg = cli.parse_args()

    assert cfg._generation_mode == "generate"
    assert cfg._distributed_rank == 2
    assert cfg._distributed_world_size == 5


def test_auto_mode_uses_slurm_rank_partition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cfg = Config(out_dir=str(tmp_path), recomb_map="", mut_map="")
    monkeypatch.setattr(cli, "parse_args", lambda: cfg)
    monkeypatch.setattr(cli, "load_empirical_maps", lambda _cfg: EmpiricalMapStore())
    monkeypatch.setenv("SLURM_PROCID", "2")
    monkeypatch.setenv("SLURM_NTASKS", "4")
    calls: list[tuple[int, int]] = []

    def generate_partition(_cfg, _empirical, *, rank, world_size):
        calls.append((rank, world_size))
        return {"generated_chunks": 3, "reused_chunks": 1}

    monkeypatch.setattr(cli, "generate_dataset_partition", generate_partition)
    monkeypatch.setattr(
        cli,
        "generate_dataset",
        lambda *_args, **_kwargs: pytest.fail("single-node generator was called"),
    )

    cli.main()

    assert calls == [(2, 4)]


def test_prepare_only_loads_maps_and_allows_force(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cfg = Config(
        out_dir=str(tmp_path),
        recomb_map="",
        mut_map="",
        force=True,
    )
    cfg._generation_mode = "prepare"
    cfg._distributed_rank = None
    cfg._distributed_world_size = None
    empirical = EmpiricalMapStore()
    calls: list[tuple[Config, EmpiricalMapStore]] = []
    monkeypatch.setattr(cli, "parse_args", lambda: cfg)
    monkeypatch.setattr(cli, "load_empirical_maps", lambda _cfg: empirical)
    monkeypatch.setattr(
        cli,
        "prepare_dataset",
        lambda called_cfg, called_empirical: calls.append(
            (called_cfg, called_empirical)
        ),
    )

    cli.main()

    assert calls == [(cfg, empirical)]


def test_finalize_only_skips_map_loading_and_writes_dataset_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cfg = Config(out_dir=str(tmp_path), recomb_map="", mut_map="")
    cfg._generation_mode = "finalize"
    cfg._distributed_rank = None
    cfg._distributed_world_size = None
    manifest = {"n_samples": cfg.n_samples}
    writes: list[tuple[Config, dict]] = []
    monkeypatch.setattr(cli, "parse_args", lambda: cfg)
    monkeypatch.setattr(
        cli,
        "load_empirical_maps",
        lambda _cfg: pytest.fail("finalization loaded empirical maps"),
    )
    monkeypatch.setattr(
        cli,
        "finalize_dataset",
        lambda _cfg, **_kwargs: manifest,
    )
    def record_write(called_cfg, called_manifest):
        with pytest.raises(DatasetGenerationLockedError):
            with dataset_output_lock(tmp_path):
                pass
        writes.append((called_cfg, called_manifest))

    monkeypatch.setattr(cli, "write_dataset_files", record_write)

    cli.main()

    assert writes == [(cfg, manifest)]


def test_finalize_only_reports_incomplete_generation_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cfg = Config(out_dir=str(tmp_path), recomb_map="", mut_map="")
    cfg._generation_mode = "finalize"
    cfg._distributed_rank = None
    cfg._distributed_world_size = None
    monkeypatch.setattr(cli, "parse_args", lambda: cfg)

    def reject_finalize(_cfg, **_kwargs):
        raise DatasetGenerationIncompleteError("2 chunks are still missing")

    monkeypatch.setattr(cli, "finalize_dataset", reject_finalize)

    with pytest.raises(SystemExit, match=r"^\[error\] 2 chunks are still missing$"):
        cli.main()
