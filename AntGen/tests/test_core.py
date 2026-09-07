from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat


ANTGEN_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ANTGEN_ROOT.parent
for path in (str(ANTGEN_ROOT), str(WORKSPACE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from antgen.layout import run_layout
from antgen.metrics import analyze_fullwave_results, rank_surrogate_candidates, response_metrics
from antgen.pipeline import (
    build_cnn_surrogate_inputs,
    convert_topology_to_matlab,
    denormalize_generated_current,
    keep_feed_connected_component,
)
from config import CONFIG
from datasets.h5_dataset import normalize_x
from run import _validate_args, parse_args


class MetricsTests(unittest.TestCase):
    def test_metrics_are_per_condition_and_candidate(self):
        target_s11 = np.array([[1.0, 2.0], [3.0, 4.0]])
        s11_sim = np.array(
            [
                [[1.0, 2.0], [2.0, 3.0]],
                [[2.0, 3.0], [3.0, 4.0]],
            ]
        )
        target_pattern = np.zeros((2, 1, 1, 2))
        pattern_sim = np.zeros((2, 2, 1, 1, 2))
        pattern_sim[:, 1] = 2.0
        result = response_metrics(s11_sim, pattern_sim, target_s11, target_pattern)
        np.testing.assert_allclose(result["s11_mae"], [[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(result["pattern_mae"], [[0.0, 2.0], [0.0, 2.0]])
        np.testing.assert_allclose(result["combined_mae"], [[0.0, 1.5], [0.5, 1.0]])

    def test_metric_weights_must_be_valid(self):
        shape_s11 = np.zeros((1, 1, 1))
        shape_pattern = np.zeros((1, 1, 1, 1, 1))
        with self.assertRaises(ValueError):
            response_metrics(shape_s11, shape_pattern, np.zeros((1, 1)), np.zeros((1, 1, 1, 1)), 0, 0)


class TopologyTests(unittest.TestCase):
    def test_only_feed_component_is_retained(self):
        metal = np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 0, 1],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
            ],
            dtype=bool,
        )
        retained = keep_feed_connected_component(metal, 0, 0)
        expected = np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(retained, expected)

    def test_feed_is_forced_to_metal(self):
        retained = keep_feed_connected_component(np.zeros((3, 3), dtype=bool), 1, 2)
        self.assertEqual(int(retained.sum()), 1)
        self.assertTrue(retained[1, 2])

    def test_python_to_matlab_coordinate_conversion(self):
        metal = np.array([[[0, 1, 0], [0, 0, 1]]], dtype=np.uint8)
        feed = np.array([[0, 2]], dtype=np.int64)
        matlab_metal, matlab_feed = convert_topology_to_matlab(metal, feed)
        np.testing.assert_array_equal(matlab_metal, metal)
        np.testing.assert_array_equal(matlab_feed, [[1, 3]])
        self.assertEqual(matlab_metal.shape, (1, 2, 3))

    def test_h5_preprocess_then_matlab_export_preserves_physical_axes(self):
        physical = np.array([[1, 0, 0], [0, 1, 2]], dtype=np.float32)
        raw_h5_sample = physical.T
        normalized = normalize_x(raw_h5_sample, sigma=1.0)
        feed = np.array([np.unravel_index(np.argmax(normalized[1]), normalized[1].shape)])
        matlab_metal, matlab_feed = convert_topology_to_matlab(
            normalized[None, 0] > 0.5, feed
        )
        np.testing.assert_array_equal(matlab_metal[0], physical > 0)
        np.testing.assert_array_equal(matlab_feed, [[2, 3]])


class CurrentTransformTests(unittest.TestCase):
    def test_signed_log_current_inverse(self):
        class Standardizer:
            input_stats = {
                "current": {
                    "transform": "signed_log_clip_zscore",
                    "alpha": np.array([[2.0]], dtype=np.float32),
                    "mean": np.array([[0.5]], dtype=np.float32),
                    "std": np.array([[0.25]], dtype=np.float32),
                    "clip": 4.0,
                }
            }

        standardized = np.zeros((2, 3, 1, 2, 2), dtype=np.float32)
        physical = denormalize_generated_current(Standardizer(), standardized, (1, 1, 2, 2))
        self.assertEqual(physical.shape, (2, 3, 1, 1, 2, 2))
        np.testing.assert_allclose(physical, 2.0 * np.expm1(0.5))

    def test_cnn_input_has_metal_and_standardized_gaussian_feed(self):
        class Standardizer:
            @staticmethod
            def normalize_input(name, value):
                self_name = name
                if self_name != "X":
                    raise AssertionError(name)
                return value * 2.0 - 1.0

        metal = np.zeros((1, 3, 4), dtype=np.uint8)
        metal[0, 1, 2] = 1
        encoded = build_cnn_surrogate_inputs(
            metal, np.array([[1, 2]]), feed_sigma=2.0, standardizer=Standardizer()
        ).numpy()
        self.assertEqual(encoded.shape, (1, 2, 3, 4))
        self.assertEqual(float(encoded[0, 0, 1, 2]), 1.0)
        self.assertEqual(float(encoded[0, 0, 0, 0]), -1.0)
        self.assertAlmostEqual(float(encoded[0, 1, 1, 2]), 1.0, places=6)


class SurrogateRankingTests(unittest.TestCase):
    @staticmethod
    def _write_candidates(layout, m=2, n=3):
        layout.ensure()
        np.savez_compressed(
            layout.candidates,
            target_s11=np.zeros((m, 2), dtype=np.float32),
            target_pattern=np.zeros((m, 1, 1, 2), dtype=np.float32),
            dataset_indices=np.arange(10, 10 + m),
            test_offsets=np.arange(m),
            feed_rc_python_0based=np.zeros((m, n, 2), dtype=np.int64),
            feed_rc_matlab_1based=np.ones((m, n, 2), dtype=np.int64),
            metal_binary_python=np.ones((m, n, 2, 2), dtype=np.uint8),
        )

    def test_surrogate_ranking_is_stable_and_classified(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = run_layout(Path(temp_dir), create=True)
            self._write_candidates(layout)
            s11 = np.array(
                [
                    [[1, 1], [1, 1], [2, 2]],
                    [[3, 3], [2, 2], [1, 1]],
                ],
                dtype=np.float32,
            )
            pattern = np.zeros((2, 3, 1, 1, 2), dtype=np.float32)
            summary, _ = rank_surrogate_candidates(temp_dir, s11, pattern)
            self.assertEqual(summary["selected_candidate_indices"], [0, 2])
            self.assertTrue(layout.surrogate_ranking.is_file())
            self.assertTrue(layout.surrogate_best.is_file())
            self.assertTrue(layout.surrogate_summary.is_file())

    def test_selected_only_fullwave_keeps_original_candidate_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = run_layout(Path(temp_dir), create=True)
            selected_indices = np.array([2, 1], dtype=np.int64)
            target_s11 = np.zeros((2, 2), dtype=np.float32)
            target_pattern = np.zeros((2, 1, 1, 2), dtype=np.float32)
            np.savez_compressed(
                layout.selected_candidates,
                selected_candidate_indices=selected_indices,
                target_s11=target_s11,
                target_pattern=target_pattern,
                dataset_indices=np.array([20, 21]),
                test_offsets=np.array([0, 1]),
                freq_hz=np.array([8e9, 9e9]),
                pattern_freq_hz=np.array([10e9]),
                pattern_theta_deg=np.array([1.0, 4.0]),
            )
            with layout.surrogate_summary.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "ranking": {"s11_weight": 1.0, "pattern_weight": 1.0},
                        "selected_candidate_indices": selected_indices.tolist(),
                    },
                    handle,
                )
            savemat(
                layout.matlab_output,
                {
                    "s11_sim": np.ones((2, 1, 2), dtype=np.float32),
                    "pattern_sim": np.ones((2, 1, 1, 1, 2), dtype=np.float32),
                    "success": np.ones((2, 1), dtype=np.uint8),
                    "error_messages": np.full((2, 1), "", dtype=object),
                    "selected_candidate_indices": selected_indices.reshape(-1, 1),
                    "freq_hz": np.array([[8e9, 9e9]]),
                    "pattern_freq_hz": np.array([[10e9]]),
                    "pattern_theta_deg": np.array([[1.0, 4.0]]),
                    "spatial_coordinate_convention": "physical_row_col_v1",
                    "pattern_channel_order": "XOZ_Gtheta,XOZ_Gphi,YOZ_Gtheta,YOZ_Gphi",
                    "pattern_value_type": "linear_gain",
                    "matlab_version": "test",
                    "matlab_release": "test",
                },
            )
            summary = analyze_fullwave_results(temp_dir)
            self.assertEqual(
                summary["fullwave_validation"]["selected_candidate_indices"], [2, 1]
            )
            self.assertEqual(
                summary["fullwave_validation"]["num_fullwave_simulations_requested"], 2
            )
            self.assertTrue(layout.final_summary.is_file())

    def test_legacy_matlab_output_without_coordinate_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = run_layout(Path(temp_dir), create=True)
            np.savez_compressed(
                layout.selected_candidates,
                selected_candidate_indices=np.array([0]),
                target_s11=np.zeros((1, 2), dtype=np.float32),
                target_pattern=np.zeros((1, 1, 1, 2), dtype=np.float32),
                dataset_indices=np.array([20]),
                test_offsets=np.array([0]),
                freq_hz=np.array([8e9, 9e9]),
                pattern_freq_hz=np.array([10e9]),
                pattern_theta_deg=np.array([1.0, 4.0]),
            )
            with layout.surrogate_summary.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "ranking": {"s11_weight": 1.0, "pattern_weight": 1.0},
                        "selected_candidate_indices": [0],
                    },
                    handle,
                )
            savemat(
                layout.matlab_output,
                {
                    "s11_sim": np.zeros((1, 1, 2), dtype=np.float32),
                    "pattern_sim": np.zeros((1, 1, 1, 1, 2), dtype=np.float32),
                    "success": np.ones((1, 1), dtype=np.uint8),
                },
            )
            with self.assertRaisesRegex(ValueError, "physical row/col"):
                analyze_fullwave_results(temp_dir)


class ConfigTests(unittest.TestCase):
    def test_run_defaults_come_from_config(self):
        args = parse_args([])
        self.assertEqual(args.num_conditions, CONFIG["sampling"]["num_conditions"])
        self.assertEqual(args.num_candidates, CONFIG["sampling"]["num_candidates"])
        self.assertEqual(args.matlab_workers, CONFIG["matlab"]["workers"])
        self.assertEqual(args.generation_batch_size, CONFIG["inference"]["generation_batch_size"])
        self.assertEqual(args.surrogate_batch_size, CONFIG["surrogate"]["batch_size"])
        _validate_args(args)

    def test_invalid_candidate_count_is_rejected(self):
        args = parse_args([])
        args.num_candidates = 0
        with self.assertRaises(ValueError):
            _validate_args(args)


if __name__ == "__main__":
    unittest.main()
