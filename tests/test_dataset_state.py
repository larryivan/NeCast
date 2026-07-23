from __future__ import annotations

import dataclasses
import unittest

from simulator.config import Config
from simulator.dataset_state import (
    DONE_CONFIG_STATE_KEY,
    RUNTIME_CONFIG_FIELDS,
    DatasetConfigMismatchError,
    sample_generation_config,
    sample_generation_fingerprint,
    stamp_done_manifest,
    validate_done_for_reuse,
)


class DatasetStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config()

    def test_snapshot_includes_every_non_runtime_config_field(self) -> None:
        raw = dataclasses.asdict(self.cfg)
        snapshot = sample_generation_config(self.cfg)
        self.assertEqual(set(snapshot), set(raw) - RUNTIME_CONFIG_FIELDS)

    def test_fingerprint_is_stable_and_ignores_runtime_fields(self) -> None:
        original = sample_generation_fingerprint(self.cfg)
        reordered = dict(reversed(list(dataclasses.asdict(self.cfg).items())))
        self.assertEqual(original, sample_generation_fingerprint(reordered))

        changed = dataclasses.replace(
            self.cfg,
            dataset_name="renamed",
            compute_chunk_size=1,
            force=True,
            out_dir="/a/different/output",
            workers=99,
        )
        self.assertEqual(original, sample_generation_fingerprint(changed))

    def test_fingerprint_changes_for_data_and_format_fields(self) -> None:
        original = sample_generation_fingerprint(self.cfg)
        for field, value in {
            "seed": self.cfg.seed + 1,
            "n_samples": self.cfg.n_samples + 1,
            "shard_size": self.cfg.shard_size + 1,
            "seq_len": self.cfg.seq_len + 1,
            "compression": not self.cfg.compression,
            "compression_level": self.cfg.compression_level + 1,
            "include_truth": not self.cfg.include_truth,
            "max_retries": self.cfg.max_retries + 1,
        }.items():
            with self.subTest(field=field):
                changed = dataclasses.replace(self.cfg, **{field: value})
                self.assertNotEqual(original, sample_generation_fingerprint(changed))

    def test_stamped_done_validates_without_legacy_config(self) -> None:
        manifest = stamp_done_manifest({"n_samples": 4}, self.cfg)
        state = validate_done_for_reuse(manifest, self.cfg)
        self.assertEqual(state, manifest[DONE_CONFIG_STATE_KEY])

    def test_stamped_done_rejects_changed_generation_config(self) -> None:
        manifest = stamp_done_manifest({"n_samples": 4}, self.cfg)
        changed = dataclasses.replace(self.cfg, seq_len=self.cfg.seq_len * 2)
        with self.assertRaisesRegex(DatasetConfigMismatchError, r"seq_len.*--force"):
            validate_done_for_reuse(manifest, changed)

    def test_stamped_done_rejects_a_different_generator_schema(self) -> None:
        manifest = stamp_done_manifest({"n_samples": 4}, self.cfg)
        manifest[DONE_CONFIG_STATE_KEY]["generator_schema_version"] += 1
        with self.assertRaisesRegex(
            DatasetConfigMismatchError,
            r"sample-generator schema.*--force",
        ):
            validate_done_for_reuse(manifest, self.cfg)

    def test_legacy_done_requires_force_instead_of_automatic_upgrade(self) -> None:
        with self.assertRaisesRegex(DatasetConfigMismatchError, r"cannot prove.*--force"):
            validate_done_for_reuse({"n_samples": 4}, self.cfg)

    def test_stamp_does_not_mutate_input_manifest(self) -> None:
        manifest = {"n_samples": 4}
        stamped = stamp_done_manifest(manifest, self.cfg)
        self.assertNotIn(DONE_CONFIG_STATE_KEY, manifest)
        self.assertIn(DONE_CONFIG_STATE_KEY, stamped)


if __name__ == "__main__":
    unittest.main()
