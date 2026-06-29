from __future__ import annotations

import unittest

from miles_plugins.envpack_adapter.filters import check_envpack_success_nonzero_std


class _Sample:
    def __init__(self, success: bool | None = None):
        self.metadata = {"envpack": {}}
        if success is not None:
            self.metadata["envpack"]["success"] = success


class EnvpackFiltersTest(unittest.TestCase):
    def test_keeps_mixed_success_group(self) -> None:
        result = check_envpack_success_nonzero_std(None, [_Sample(True), _Sample(False)])

        self.assertTrue(result.keep)
        self.assertIsNone(result.reason)

    def test_rejects_all_solved_group(self) -> None:
        result = check_envpack_success_nonzero_std(None, [_Sample(True), _Sample(True)])

        self.assertFalse(result.keep)
        self.assertEqual(result.reason, "all_solved")

    def test_rejects_none_solved_group(self) -> None:
        result = check_envpack_success_nonzero_std(None, [_Sample(False), _Sample(False)])

        self.assertFalse(result.keep)
        self.assertEqual(result.reason, "none_solved")

    def test_missing_success_fails_loud(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit success metadata"):
            check_envpack_success_nonzero_std(None, [_Sample(None)])


if __name__ == "__main__":
    unittest.main()
