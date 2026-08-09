"""Preflight fails on the ground, not in the air.

Everything here is checkable without a GPU and without the real data, because
the point is to run it *before* an allocation exists.
"""

import json

from tools.lora_regret.preflight import (
    STAGE_GPU_REQUIREMENTS,
    check_checkpoints,
    check_data,
    check_matrices,
)


class TestCheckData:
    def test_a_missing_split_fails_by_name(self, tmp_path):
        failures = [c for c in check_data(tmp_path) if not c.ok]
        assert any("tulu3_train.jsonl" in c.detail for c in failures)

    def test_a_truncated_split_fails_even_though_it_exists(self, tmp_path):
        """Existence is not enough: a short split silently changes E1's denominator."""
        (tmp_path / "tulu3_test.jsonl").write_text(
            "".join(json.dumps({"prompt": []}) + "\n" for _ in range(999))
        )
        checks = {c.name: c for c in check_data(tmp_path)}
        assert not checks["tulu3_test.jsonl"].ok
        assert "999" in checks["tulu3_test.jsonl"].detail
        assert "1000" in checks["tulu3_test.jsonl"].detail

    def test_a_correct_split_passes(self, tmp_path):
        (tmp_path / "tulu3_test.jsonl").write_text(
            "".join(json.dumps({"prompt": []}) + "\n" for _ in range(1000))
        )
        checks = {c.name: c for c in check_data(tmp_path)}
        assert checks["tulu3_test.jsonl"].ok

    def test_all_nine_splits_are_checked(self, tmp_path):
        assert len(check_data(tmp_path)) == 9


class TestCheckCheckpoints:
    def test_a_missing_megatron_checkpoint_fails(self, tmp_path):
        checks = {c.name: c for c in check_checkpoints(tmp_path, tmp_path / "nope")}
        assert not checks["megatron_load"].ok

    def test_a_megatron_dir_without_the_iteration_file_fails(self, tmp_path):
        (tmp_path / "mg").mkdir()
        checks = {c.name: c for c in check_checkpoints(tmp_path, tmp_path / "mg")}
        assert not checks["megatron_load"].ok
        assert "latest_checkpointed_iteration.txt" in checks["megatron_load"].detail

    def test_a_complete_megatron_dir_passes(self, tmp_path):
        (tmp_path / "mg").mkdir()
        (tmp_path / "mg" / "latest_checkpointed_iteration.txt").write_text("0")
        checks = {c.name: c for c in check_checkpoints(tmp_path, tmp_path / "mg")}
        assert checks["megatron_load"].ok


class TestCheckMatrices:
    def test_every_matrix_builds_at_its_documented_count(self):
        checks = {c.name: c for c in check_matrices(4096, 14336, 6144)}
        assert checks["matrix:e1"].ok and "45" in checks["matrix:e1"].detail
        assert checks["matrix:e2"].ok and "48" in checks["matrix:e2"].detail
        assert checks["matrix:e3"].ok and "35" in checks["matrix:e3"].detail
        assert checks["matrix:e4"].ok and "70" in checks["matrix:e4"].detail
        assert checks["matrix:e5scout"].ok and "5" in checks["matrix:e5scout"].detail
        assert checks["matrix:e5"].ok and "50" in checks["matrix:e5"].detail

    def test_a_matrix_that_raises_is_reported_not_propagated(self, monkeypatch):
        """A broken matrix must fail the preflight, not crash it.

        Preflight's whole value is telling you every problem at once; an
        uncaught exception in the third matrix hides the fourth.
        """
        import tools.lora_regret.preflight as preflight

        def boom(*_args, **_kwargs):
            raise ValueError("hidden_size and ffn_size must be positive")

        monkeypatch.setitem(preflight.MATRICES, "e1", boom)
        checks = {c.name: c for c in check_matrices(4096, 14336, 6144)}
        assert not checks["matrix:e1"].ok
        assert "ValueError" in checks["matrix:e1"].detail
        assert checks["matrix:e2"].ok  # the rest still ran

    def test_a_wrong_count_fails_even_though_the_matrix_builds(self, monkeypatch):
        """Not a tautology: the counts are pinned, not read back from the builder."""
        import tools.lora_regret.preflight as preflight

        monkeypatch.setitem(preflight.EXPECTED_ARMS, "e1", 46)
        checks = {c.name: c for c in check_matrices(4096, 14336, 6144)}
        assert not checks["matrix:e1"].ok
        assert "45 arms, expected 46" in checks["matrix:e1"].detail


class TestStageRequirements:
    def test_fullft_needs_four_gpus(self):
        assert STAGE_GPU_REQUIREMENTS["e1-full"] == 4

    def test_p3_needs_at_least_two(self):
        """DP=1 makes the reduction a no-op, so a 1-GPU 'P3' proves nothing."""
        assert STAGE_GPU_REQUIREMENTS["p3"] >= 2

    def test_rl_needs_eight(self):
        assert STAGE_GPU_REQUIREMENTS["e4"] == 8


def test_the_new_stages_carry_their_gpu_floors():
    from tools.lora_regret.preflight import STAGE_GPU_REQUIREMENTS

    assert STAGE_GPU_REQUIREMENTS["e1ot"] == 1
    assert STAGE_GPU_REQUIREMENTS["e1short"] == 1
    assert STAGE_GPU_REQUIREMENTS["e4place"] == 8


def test_every_matrix_is_expected_at_its_documented_count():
    from tools.lora_regret.arms import MATRICES
    from tools.lora_regret.preflight import EXPECTED_ARMS

    assert set(EXPECTED_ARMS) == set(MATRICES) - {"e1long"}
    assert EXPECTED_ARMS["e1ot"] == 45
    assert EXPECTED_ARMS["e1short"] == 21
    assert EXPECTED_ARMS["e4place"] == 35
    assert EXPECTED_ARMS["e4lr0"] == 6


def test_the_fullft_stages_agree_with_the_registrys_formula():
    """preflight's floor and the launcher's guard must not drift apart."""
    from tools.lora_regret.models import get
    from tools.lora_regret.preflight import STAGE_GPU_REQUIREMENTS

    assert STAGE_GPU_REQUIREMENTS["e1-full"] == get("llama3.1-8b").min_gpus_fullft()
