"""Lightweight tests for the RWKV-state decoder pilot.

They exercise the REAL code path on the GPU but with synthetic features and a
byte tokenizer (`--dry-run`), so they finish in seconds and never load the 1.5B
backbone. Coverage targets the things that must not rot silently:

  * state feature extraction shape/normalisation and refusal of bad states
  * decoder forward, conditioning bottleneck and deterministic generation
  * run identity: same seed -> same losses; identity changes with the ablation
  * resume: continuing from a checkpoint equals an uninterrupted run
  * checkpoints carry model/optimizer/RNG/step/identity; wrong identity refused
  * fail-closed power behaviour: no lease -> save + exit 75, valid lease -> no-op
  * the dataset gates that keep the labels honest

Run: D:\\AI\\DBA\\.venv-rwkv\\Scripts\\python.exe t_train_slots.py
"""
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import torch

import rwkv_slots_data as DATA
import train_rwkv_slots as T


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rwkv-slots-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def args(self, out=None, **changes):
        out = Path(out or self.tmp)
        out.mkdir(parents=True, exist_ok=True)
        raw = ["--dry-run", "--skip-power-check", "--out-dir", str(out),
               "--resume", str(out / "checkpoint-last.pt"), "--log-every", "1",
               "--checkpoint-every", "2", "--heldout-every", "0", "--eval-samples", "1",
               "--eval-max-tokens", "24", "--seed", "1234", "--d-model", "64",
               "--n-heads", "4", "--steps", "4"]
        args = T.parse_args(raw)
        for key, value in changes.items():
            setattr(args, key.replace("-", "_"), value)
        return args

    def run_once(self, out=None, **changes):
        return T.run(self.args(out=out, **changes), hooks={"torch": torch})


class FeatureTests(unittest.TestCase):
    def test_state_features_shape_and_normalisation(self):
        n_layer, width = 3, 16
        state = [torch.randn(4, 4, 4) if index % 3 == 1 else torch.randn(width)
                 for index in range(n_layer * 3)]
        features = T.state_features(state, n_layer, torch)
        self.assertEqual(tuple(features.shape), (n_layer * 3, width))
        for row in features:
            centred = row - row.mean()
            self.assertLess(abs(float(row.mean())), 1e-4)
            self.assertAlmostEqual(float(centred.pow(2).mean().sqrt()), 1.0, places=3)

    def test_axis_means_keep_the_row_count_fixed(self):
        n_layer = 2
        state = [torch.randn(4, 4, 4) if index % 3 == 1 else torch.randn(16)
                 for index in range(n_layer * 3)]
        self.assertEqual(tuple(T.state_features(state, n_layer, torch).shape), (6, 16))

    def test_state_features_rejects_short_or_ragged_state(self):
        with self.assertRaisesRegex(T.TrainingError, "fewer tensors"):
            T.state_features([torch.randn(8)] * 5, 3, torch)
        ragged = [torch.randn(8) if index % 3 != 1 else torch.randn(2, 2, 2)
                  for index in range(6)]
        with self.assertRaisesRegex(T.TrainingError, "inconsistent widths"):
            T.state_features(ragged, 2, torch)

    def test_special_ids_sit_above_every_real_token(self):
        tokenizer = T.ByteTokenizer()
        ids = T.special_ids(tokenizer)
        widest = T.widest_real_id(tokenizer, [DATA.memory_text(case)
                                              for case in DATA.all_cases()])
        self.assertGreater(ids["first"], widest)
        self.assertEqual(ids["bos"] + 1, ids["eos"])
        self.assertEqual(ids["bos"] + 2, ids["pad"])

    def test_every_token_in_a_sequence_fits_the_embedding_table(self):
        # A byte tokenizer reserves ids far above its text range; if the table is
        # too small, nn.Embedding gathers out of bounds (a device-side assert).
        tokenizer = T.ByteTokenizer()
        ids = T.special_ids(tokenizer)
        widest = T.widest_real_id(tokenizer, [DATA.memory_text(case)
                                              for case in DATA.all_cases()])
        sequence = T.encode_sequence(tokenizer, DATA.memory_text(DATA.TRAIN_CASES[0]), ids)
        self.assertLessEqual(widest, tokenizer.real_ids[1])
        self.assertLess(max(sequence), tokenizer.vocab_size)
        table = torch.nn.Embedding(tokenizer.vocab_size, 4)
        table(torch.tensor([sequence]))

    def test_narrow_synthetic_table_still_keeps_specials_in_range(self):
        # Guards the id layout itself: specials must never reach the table size.
        tokenizer = T.ByteTokenizer(span=8)
        ids = T.special_ids(tokenizer)
        self.assertLess(ids["first"], tokenizer.vocab_size)
        self.assertEqual(ids["pad"], tokenizer.vocab_size - 1)
        self.assertEqual(tokenizer.real_ids, (T.EOT_ID + 1, T.EOT_ID + 8))

    def test_byte_tokenizer_round_trip(self):
        tokenizer = T.ByteTokenizer()
        text = DATA.TRAIN_CASES[0]["summary"]
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.config = T.DecoderConfig(6, 16, 256, d_model=64, n_heads=4, n_self_layers=1,
                                      n_cross_layers=1, conditioning_tokens=4)
        self.module = T.build_decoder(self.config, torch)
        self.tokens = torch.randint(1, 250, (2, 5))
        self.features = torch.randn(2, 6, 16)

    def test_forward_contract(self):
        logits = self.module(self.tokens, self.features)
        self.assertEqual(tuple(logits.shape), (2, 5, 256))
        self.assertTrue(torch.isfinite(logits).all())

    def test_padding_mask_preserves_shape(self):
        mask = torch.zeros(2, 5, dtype=torch.bool)
        self.assertEqual(tuple(self.module(self.tokens, self.features, mask).shape), (2, 5, 256))

    def test_conditioning_reaches_the_output(self):
        other = self.features.clone()
        other[:, 0, :] += 5.0
        self.assertFalse(torch.allclose(self.module(self.tokens, self.features),
                                        self.module(self.tokens, other)))
        zeros = torch.zeros_like(self.features)
        self.assertFalse(torch.allclose(self.module(self.tokens, self.features),
                                        self.module(self.tokens, zeros)))

    def test_generation_is_deterministic_and_stays_in_text_vocabulary(self):
        tokenizer = T.ByteTokenizer()
        ids = T.special_ids(tokenizer)
        first = T.generate_text(self.module, self.features[:1], tokenizer, ids, 16, torch,
                                torch.device("cpu"))
        second = T.generate_text(self.module, self.features[:1], tokenizer, ids, 16, torch,
                                 torch.device("cpu"))
        self.assertEqual(first, second)
        self.assertIsInstance(first, str)
        self.assertNotIn(T.ByteTokenizer.name, first)

    def test_conditioning_shape_is_what_the_config_promises(self):
        encoder = T.build_decoder(self.config, torch).state_encoder
        codes = encoder(self.features)
        self.assertEqual(tuple(codes.shape),
                         (2, self.config.conditioning_tokens, self.config.d_model))
        self.assertTrue(torch.isfinite(codes).all())

    def test_config_round_trip_and_validation(self):
        payload = self.config.to_dict()
        self.assertEqual(T.DecoderConfig.from_dict(payload).to_dict(), payload)
        with self.assertRaisesRegex(T.TrainingError, "n_cross_layers"):
            T.DecoderConfig(2, 2, 64, d_model=32, n_heads=4, n_cross_layers=0)
        with self.assertRaisesRegex(T.TrainingError, "divisible"):
            T.DecoderConfig(2, 2, 64, d_model=32, n_heads=5)
        with self.assertRaisesRegex(T.TrainingError, "conditioning_tokens"):
            T.DecoderConfig(2, 2, 64, d_model=32, n_heads=4, conditioning_tokens=0)


class RunTests(TempCase):
    def test_run_writes_status_log_pointer_and_checkpoint(self):
        result = self.run_once()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["step"], 4)
        self.assertEqual(result["label"], DATA.LABEL)
        self.assertFalse(result["backbone_training"])
        status = json.loads((self.tmp / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["phase"], "complete")
        self.assertEqual(status["state_mode"], "true")
        self.assertFalse(status["backbone_training"])
        self.assertEqual(status["run_identity"]["backbone_mode"], "synthetic")
        self.assertEqual(status["run_identity"]["data_hash"], DATA.data_version(DATA.all_cases()))
        self.assertEqual(status["feature_extraction"]["feature_version"], T.FEATURE_VERSION)
        self.assertTrue((self.tmp / "checkpoint-last.pt").is_file())
        self.assertIn("decoder-only pilot", (
            self.tmp / "train.log").read_text(encoding="utf-8"))
        self.assertEqual(json.loads((self.tmp / "latest.json").read_text(
            encoding="utf-8"))["step"], 4)

    def test_same_seed_reproduces_same_losses(self):
        first = self.run_once(out=self.tmp / "a")
        second = self.run_once(out=self.tmp / "b")
        self.assertEqual([round(h["loss"], 6) for h in first["history"]],
                         [round(h["loss"], 6) for h in second["history"]])
        self.assertEqual(first["identity_hash"], second["identity_hash"])

    def test_checkpoint_carries_everything_a_resume_needs(self):
        self.run_once(steps=2)
        payload = torch.load(self.tmp / "checkpoint-last.pt", map_location="cpu",
                             weights_only=False)
        for key in ("model_state", "optimizer_state", "rng", "step", "identity_hash",
                    "run_identity", "history", "loss", "phase"):
            self.assertIn(key, payload)
        self.assertEqual(payload["step"], 2)
        self.assertIn("torch", payload["rng"])
        self.assertIn("torch_cuda", payload["rng"])
        self.assertIn("python", payload["rng"])
        self.assertTrue(any("state_encoder" in name for name in payload["model_state"]))
        rebuilt = T.build_decoder(
            T.DecoderConfig.from_dict(payload["run_identity"]["decoder"]), torch)
        self.assertEqual(sorted(payload["model_state"]), sorted(rebuilt.state_dict()))
        # Optimizer moments must be on CPU so the file is device-portable.
        moments = payload["optimizer_state"]["state"]
        self.assertTrue(moments)
        for entry in moments.values():
            for value in entry.values():
                self.assertFalse(getattr(value, "is_cuda", False))

    def test_resume_is_bit_identical_to_an_uninterrupted_run(self):
        uninterrupted = self.run_once(out=self.tmp / "full", steps=6)
        self.run_once(out=self.tmp / "split", steps=3, checkpoint_every=3)
        resumed = self.run_once(out=self.tmp / "split", steps=6, checkpoint_every=3)
        self.assertEqual(resumed["step"], 6)
        self.assertEqual([round(h["loss"], 6) for h in uninterrupted["history"]],
                         [round(h["loss"], 6) for h in resumed["history"]])

    def test_resume_refuses_a_changed_identity(self):
        self.run_once(steps=2)
        payload = torch.load(self.tmp / "checkpoint-last.pt", map_location="cpu",
                             weights_only=False)
        payload["identity_hash"] = "0" * 64
        torch.save(payload, self.tmp / "checkpoint-last.pt")
        with self.assertRaisesRegex(T.TrainingError, "identity mismatch"):
            self.run_once(steps=4)

    def test_resume_refuses_a_corrupt_checkpoint(self):
        self.run_once(steps=2)
        (self.tmp / "checkpoint-last.pt").write_text("not a checkpoint", encoding="utf-8")
        with self.assertRaises(Exception):
            self.run_once(steps=4)

    def test_checkpoint_past_steps_is_a_no_op(self):
        self.run_once(steps=4)
        again = self.run_once(steps=3)
        self.assertTrue(again["ok"])
        self.assertEqual(again["phase"], "already-complete")
        self.assertEqual(again["step"], 4)

    def test_ablation_modes_are_distinct_runs_with_distinct_identity(self):
        runs = {mode: self.run_once(out=self.tmp / mode, state_mode=mode, steps=3)
                for mode in ("true", "none", "shuffled")}
        for mode, run in runs.items():
            self.assertTrue(run["ok"], (mode, run))
            self.assertEqual(run["step"], 3)
        hashes = {mode: run["identity_hash"] for mode, run in runs.items()}
        self.assertEqual(len(set(hashes.values())), 3, hashes)
        self.assertNotEqual([h["loss"] for h in runs["true"]["history"]],
                            [h["loss"] for h in runs["none"]["history"]])
        self.assertNotEqual(runs["true"]["final_loss"], runs["shuffled"]["final_loss"])

    def test_shuffled_ablation_keeps_everything_else_identical(self):
        reference = self.run_once(out=self.tmp / "reference", state_mode="true", steps=2)
        shuffled = self.run_once(out=self.tmp / "shuffled", state_mode="shuffled", steps=2)
        self.assertTrue(reference["ok"], reference)
        self.assertTrue(shuffled["ok"], shuffled)
        true_status = json.loads((self.tmp / "reference" / "status.json").read_text(
            encoding="utf-8"))
        shuffled_status = json.loads((self.tmp / "shuffled" / "status.json").read_text(
            encoding="utf-8"))
        self.assertEqual(true_status["run_identity"]["feature_hash"],
                         shuffled_status["run_identity"]["feature_hash"])
        self.assertEqual(true_status["run_identity"]["data_hash"],
                         shuffled_status["run_identity"]["data_hash"])
        self.assertEqual(true_status["decoder_parameters"],
                         shuffled_status["decoder_parameters"])
        self.assertEqual(true_status["run_identity"]["backbone_training"], False)
        self.assertNotEqual(true_status["identity_hash"], shuffled_status["identity_hash"])

    def test_none_mode_conditions_on_a_constant_vector(self):
        args = self.args(state_mode="none", steps=2)
        captured = {}
        original = T.build_decoder

        def spy(config, torch_module):
            module = original(config, torch_module)
            original_forward = module.forward

            def forward(tokens, features, pad_mask=None):
                captured["features"] = features.detach().clone()
                return original_forward(tokens, features, pad_mask)

            module.forward = forward
            return module

        T.build_decoder = spy
        try:
            result = T.run(args, hooks={"torch": torch})
        finally:
            T.build_decoder = original
        self.assertTrue(result["ok"])
        features = captured["features"]
        self.assertTrue(torch.all(features == 0))
        self.assertEqual(features.shape[0], 1)

    def test_flag_validation_edges(self):
        with self.assertRaisesRegex(T.TrainingError, "eval-samples"):
            T.parse_args(["--steps", "1", "--dry-run", "--eval-samples", "-1"])
        with self.assertRaisesRegex(T.TrainingError, "checkpoint-every"):
            T.parse_args(["--steps", "1", "--dry-run", "--checkpoint-every", "0"])

    def test_missing_cuda_is_refused(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        class FakeTorch:
            cuda = FakeCuda()

        with self.assertRaisesRegex(T.TrainingError, "CUDA is required"):
            T.run(self.args(), hooks={"torch": FakeTorch()})


class AcFailClosedTests(TempCase):
    def setUp(self):
        super().setUp()
        self.control = self.tmp / "control.json"
        self.saved = []
        self.original = T.pause_if_requested
        os.environ.pop("DBA_TRAIN_CONTROL", None)
        self.reload_control()

    def tearDown(self):
        T.pause_if_requested = self.original
        T.training_allowed = self.original_allowed
        os.environ.pop("DBA_TRAIN_CONTROL", None)
        super().tearDown()

    def reload_control(self):
        import training_power_control
        importlib.reload(training_power_control)
        if not hasattr(self, "original_allowed"):
            self.original_allowed = T.training_allowed
        T.pause_if_requested = training_power_control.pause_if_requested
        T.training_allowed = training_power_control.training_allowed

    def test_no_lease_pauses_after_saving(self):
        with self.assertRaises(SystemExit) as caught:
            T.pause_if_requested(lambda: self.saved.append("checkpoint"))
        self.assertEqual(caught.exception.code, T.EXIT_PAUSED)
        self.assertEqual(self.saved, ["checkpoint"])
        self.assertFalse(T.training_allowed())

    def test_valid_lease_does_not_pause(self):
        self.control.write_text(json.dumps(
            {"allow_training": True, "valid_until": time.time() + 600}), encoding="utf-8")
        os.environ["DBA_TRAIN_CONTROL"] = str(self.control)
        self.reload_control()
        self.assertTrue(T.training_allowed())
        T.pause_if_requested(lambda: self.saved.append("checkpoint"))
        self.assertEqual(self.saved, [])

    def test_expired_lease_pauses_and_unreadable_control_fails_closed(self):
        self.control.write_text(json.dumps({"allow_training": True, "valid_until": 1.0}),
                                encoding="utf-8")
        os.environ["DBA_TRAIN_CONTROL"] = str(self.control)
        self.reload_control()
        with self.assertRaises(SystemExit) as caught:
            T.pause_if_requested(lambda: self.saved.append("checkpoint"))
        self.assertEqual(caught.exception.code, T.EXIT_PAUSED)
        self.control.write_text("{{{ not json", encoding="utf-8")
        self.assertFalse(T.training_allowed())

    def test_power_report_refuses_without_lease_and_reports_ac(self):
        with self.assertRaisesRegex(T.PowerLoss, "guard_lease=False"):
            T.power_report("unit-test", require_lease=True)
        report = T.power_report("unit-test", require_lease=False)
        self.assertIsInstance(report["ac_connected"], bool)
        self.assertIsInstance(T.main_connected(), bool)

    def test_main_returns_75_when_the_guard_denies_the_lease(self):
        # The whole run must stop before any CUDA allocation is attempted.
        out = self.tmp / "denied"
        code = T.main(["--steps", "2", "--out-dir", str(out),
                       "--resume", str(out / "checkpoint-last.pt"),
                       "--dry-run", "--eval-samples", "0"])
        self.assertEqual(code, T.EXIT_PAUSED)
        status = json.loads((out / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["phase"], "paused")
        self.assertEqual(status["paused_before"], "before-cuda")
        self.assertFalse((out / "checkpoint-last.pt").exists())


class DatasetGateTests(unittest.TestCase):
    def test_curated_cases_are_disjoint_unique_and_valid(self):
        items = DATA.all_cases()
        counts = DATA.split_counts(items)
        self.assertEqual(counts["train"], len(DATA.TRAIN_CASES))
        self.assertEqual(counts["heldout"], len(DATA.HELDOUT_CASES))
        self.assertEqual(len({case["id"] for case in items}), len(items))
        self.assertEqual(len({case["summary"] for case in items}), len(items))
        self.assertEqual(DATA.validate(items), items)

    def test_validate_rejects_a_lying_unverified_label(self):
        case = dict(DATA.TRAIN_CASES[0])
        case["summary"] = case["summary"].replace("尚未确认闭环调试结果", "已完成闭环调试")
        with self.assertRaisesRegex(ValueError, "reads as done"):
            DATA.validate([case])

    def test_validate_rejects_a_done_action_marked_unverified(self):
        case = dict(DATA.HELDOUT_CASES[1])
        case["unverified"] = True
        with self.assertRaisesRegex(ValueError, "reads as done"):
            DATA.validate([case])

    def test_validate_rejects_verbatim_result(self):
        case = dict(DATA.TRAIN_CASES[0])
        # An unverified label whose result lifts a sentence from the assistant turn.
        case["summary"] = ("主题：电机控制器闭环调试\n约束：完成前不得汇报调试通过\n"
                           "未完成：核对编码器极性并跑低速空载测试\n"
                           "结果：尚未确认，我先核对编码器极性并准备低速空载测试")
        self.assertIn("我先核对编码器极性并准备低速空载测试", DATA.source_text(case))
        with self.assertRaisesRegex(ValueError, "copies its source"):
            DATA.validate([case])

    def test_validate_rejects_bad_slot_order_duplicates_and_missing_slots(self):
        broken = dict(DATA.TRAIN_CASES[1])
        broken["id"] = "ORDER1"
        broken["summary"] = "约束：只允许GPU\n主题：推理硬件约束\n未完成：无\n结果：已确认GPU约束"
        with self.assertRaisesRegex(ValueError, "four slots in order"):
            DATA.validate([broken])
        duplicate_text = dict(DATA.TRAIN_CASES[0])
        duplicate_text["id"] = "COPY1"
        with self.assertRaisesRegex(ValueError, "duplicate memory text"):
            DATA.validate([DATA.TRAIN_CASES[0], duplicate_text])
        with self.assertRaisesRegex(ValueError, "duplicate case id"):
            DATA.validate([DATA.TRAIN_CASES[0], dict(DATA.TRAIN_CASES[0])])
        missing = dict(DATA.TRAIN_CASES[0])
        missing["id"] = "X"
        missing["summary"] = "主题：电机控制器闭环调试\n约束：完成前不得汇报调试通过\n结果：尚未确认"
        with self.assertRaisesRegex(ValueError, "four slots in order"):
            DATA.validate([missing])

    def test_validate_rejects_a_case_missing_fields(self):
        with self.assertRaisesRegex(ValueError, "missing fields"):
            DATA._normalized({"id": "A", "group": "train"})

    def test_data_version_tracks_content(self):
        items = DATA.all_cases()
        changed = [dict(case) for case in items]
        changed[0]["summary"] += "（改）"
        self.assertNotEqual(DATA.data_version(items), DATA.data_version(changed))

    def test_slot_match_scores_partial_credit(self):
        expected = DATA.TRAIN_CASES[0]["summary"]
        partial = expected.replace("未完成：核对编码器极性并跑低速空载测试", "未完成：核对编码器极性")
        self.assertNotEqual(partial, expected)
        score = DATA.slot_match(partial, expected)
        self.assertEqual(score["slots_exact"], 3)
        self.assertEqual(score["per_slot"]["未完成"], 0)
        self.assertFalse(score["all_slots_exact"])
        self.assertEqual(DATA.slot_match(expected, expected)["slots_exact"], 4)
        self.assertEqual(DATA.slot_match("", expected)["slots_exact"], 0)

    def test_slot_match_ignores_unlabelled_prose(self):
        expected = DATA.HELDOUT_CASES[0]["summary"]
        noisy = "好的，以下是记忆：\n" + expected + "\n（完）"
        self.assertTrue(DATA.slot_match(noisy, expected)["all_slots_exact"])

    def test_observation_text_carries_no_labels(self):
        for case in DATA.all_cases():
            text = DATA.source_text(case)
            self.assertIn(case["user"][:6], text)
            self.assertNotIn("主题：", text)
            self.assertNotIn(DATA.memory_text(case), text)

    def test_extra_cases_load_from_jsonl_and_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "extra.jsonl"
            case = dict(DATA.TRAIN_CASES[0])
            case["id"] = "EXTRA1"
            path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
            loaded = DATA.load_events(cases_path=path, template_path=Path(tmp) / "none.json")
            self.assertEqual([c["id"] for c in loaded], ["EXTRA1"])
            path.write_text("{oops}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1 invalid"):
                DATA.load_events(cases_path=path, template_path=Path(tmp) / "none.json")


if __name__ == "__main__":
    unittest.main(verbosity=2)
