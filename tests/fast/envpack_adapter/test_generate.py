from __future__ import annotations

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-fast")

import asyncio
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch  # noqa: F401
    from miles.rollout.base_types import GenerateFnInput
    from miles.utils.types import Sample
    from miles_plugins.envpack_adapter import generate as generate_mod
    from miles_plugins.envpack_adapter.renderer import RenderedObservation
except ModuleNotFoundError:
    torch = None


class _KwargsObject(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class _Tokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kwargs):
        parts = []
        for message in messages:
            parts.append(f"{message['role']}:{message['content']}")
        if add_generation_prompt:
            parts.append("assistant:")
        return "|".join(parts)

    def encode(self, text, add_special_tokens=False):
        if text == "system:placeholder":
            return [9]
        if text.startswith("system:placeholder|user:next"):
            return [9, 30, 31]
        return [10, 11]

    def decode(self, tokens, skip_special_tokens=False):
        if list(tokens) == [101, 102]:
            return "right"
        return " ".join(str(token) for token in tokens)


class _FakeClient:
    def __init__(self):
        self.cancelled = []
        self.created_requests = []
        self.step_requests = []

    async def create_episode(self, request):
        self.created_requests.append(request)
        return SimpleNamespace(
            owner=SimpleNamespace(
                env_name="sokoban",
                pool_id="sokoban-vision",
                instance_id="instance-0",
                lease_id="lease-0",
            ),
            turn_id=0,
            observation=SimpleNamespace(state={"turn": 0}),
            prompt_bundle=SimpleNamespace(system="system prompt", prompt_bundle_hash="prompt-hash"),
        )

    async def step_episode(self, request):
        self.step_requests.append(request)
        return SimpleNamespace(
            turn_id=1,
            status=SimpleNamespace(value="completed"),
            reward_delta=1.0,
            done=True,
            truncated=False,
            observation=None,
            turn_trace=SimpleNamespace(
                info={
                    "success": True,
                    "format_correct": True,
                    "actions": ["right"],
                    "metrics": {
                        "turn_metrics": {
                            "action_is_valid": True,
                            "action_is_effective": True,
                        }
                    },
                }
            ),
        )

    async def finalize_episode(self, episode_id):
        return SimpleNamespace(
            status=SimpleNamespace(value="completed"),
            reward_report=SimpleNamespace(
                reward=1.0,
                raw_reward={"success": 1.0},
                components={"success": 1.0},
                verifier_outputs={},
                signals=[],
            ),
            credit=SimpleNamespace(
                episode_reward=1.0,
                components={},
                per_turn=[],
                span_hints=[],
                signals=[],
                mode="reward_only",
            ),
            trace_summary={"turns": 1},
        )

    async def cancel_episode(self, episode_id, reason):
        self.cancelled.append((episode_id, reason))


class _StepFailsOnceClient(_FakeClient):
    def __init__(self):
        super().__init__()
        self.remaining_failures = 1

    async def step_episode(self, request):
        if self.remaining_failures:
            self.remaining_failures -= 1
            self.step_requests.append(request)
            raise _RetryableEnvpackError("instance crashed")
        return await super().step_episode(request)


class _RetryableEnvpackError(RuntimeError):
    status_code = 500


class _BuggyStepClient(_FakeClient):
    async def step_episode(self, request):
        self.step_requests.append(request)
        raise RuntimeError("adapter bug")


class _TerminalOutcomeUnknownError(RuntimeError):
    """Mimics envpack's RemoteEnvpackError for a lost terminal response: a transport
    failure (status_code=0) that envpack explicitly marks non-retryable."""

    status_code = 0

    def __init__(self, message: str = "envpack terminal request failed with unknown outcome") -> None:
        super().__init__(message)
        self.error = SimpleNamespace(code="terminal_outcome_unknown", retryable=False, message=message)


class _TerminalFinalizeFailsClient(_FakeClient):
    async def finalize_episode(self, episode_id):
        raise _TerminalOutcomeUnknownError()


class _FakeBundle:
    def __init__(self, client=None):
        self.client = client or _FakeClient()

    def env_config(self, pool_id):
        return {"render_mode": "vision"}


class EnvpackGenerateTest(unittest.TestCase):
    def test_system_message_uses_vlm_content_blocks_when_processor_is_present(self) -> None:
        if torch is None:
            self.skipTest("requires torch because Miles imports torch")

        prompt_bundle = SimpleNamespace(system="system prompt")
        self.assertEqual(
            generate_mod._system_message(prompt_bundle, content_blocks=True),
            {"role": "system", "content": [{"type": "text", "text": "system prompt"}]},
        )
        self.assertEqual(
            generate_mod._system_message(prompt_bundle, content_blocks=False),
            {"role": "system", "content": "system prompt"},
        )

    def test_generate_golden_path_preserves_miles_sample_contract(self) -> None:
        if torch is None:
            self.skipTest("requires torch because Miles Sample imports torch")

        bundle = _FakeBundle()
        calls = []

        async def fake_post(url, payload, headers=None):
            calls.append((url, payload, headers))
            return {
                "meta_info": {
                    "output_token_logprobs": [(-0.1, 101), (-0.2, 102)],
                    "finish_reason": {"type": "stop"},
                    "weight_version": "7",
                    "prompt_tokens": len(payload["input_ids"]),
                    "cached_tokens": 1,
                }
            }

        sample = Sample(
            metadata={
                "envpack": {
                    "env_name": "sokoban",
                    "seed": 123,
                    "pool_id": "sokoban-vision",
                    "env_uuid": "init-sha",
                }
            }
        )
        args = SimpleNamespace(
            envpack={
                "api": "in_process",
                "pools": [{"env": "sokoban", "profile": "vision_free_think_local", "pool_id": "sokoban-vision"}],
                "rollout": {"max_turns": 1, "response_length_per_turn": 8},
            },
            sglang_router_ip="127.0.0.1",
            sglang_router_port=30000,
            sglang_router_policy="consistent_hashing",
            sglang_speculative_algorithm=None,
            apply_chat_template_kwargs={},
            hf_checkpoint="/model",
            chat_template_path=None,
            rollout_max_context_len=64,
            partial_rollout=False,
            group_rm=False,
            rm_type=None,
            custom_rm_path=None,
            rollout_external=False,
            use_rollout_routing_replay=False,
            use_miles_router=False,
            miles_router_middleware_paths=[],
        )
        state = SimpleNamespace(args=args, tokenizer=_Tokenizer(), processor=None)
        fake_envpack = _fake_envpack_modules()

        with patch.dict(sys.modules, fake_envpack):
            with patch.object(generate_mod, "get_client_bundle", return_value=bundle):
                with patch.object(generate_mod, "post", side_effect=fake_post):
                    with patch.object(generate_mod, "is_lora_enabled", return_value=False):
                        encode_patch = patch.object(
                            generate_mod,
                            "encode_image_for_rollout_engine",
                            side_effect=lambda image: image,
                        )
                        with encode_patch:
                            with patch.object(generate_mod, "observation_to_chat_message") as render_obs:
                                render_obs.return_value = RenderedObservation(
                                    message={"role": "user", "content": "init"},
                                    images=[],
                                    videos=[],
                                    media_hashes=["init-sha"],
                                    artifacts=[],
                                )
                                output = asyncio.run(
                                    generate_mod.generate(
                                        GenerateFnInput(
                                            state=state,
                                            sample=sample,
                                            sampling_params={"max_new_tokens": 16},
                                            evaluation=False,
                                        )
                                    )
                                )

        result = output.samples
        self.assertIs(result, sample)
        self.assertEqual(result.status, Sample.Status.COMPLETED)
        self.assertEqual(result.tokens, [10, 11, 101, 102])
        self.assertEqual(result.loss_mask, [1, 1])
        self.assertEqual(result.rollout_log_probs, [-0.1, -0.2])
        self.assertEqual(result.response_length, 2)
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(result.weight_versions, ["7"])
        self.assertEqual(result.metadata["envpack"]["model_generated_token_count"], 2)
        self.assertEqual(result.metadata["envpack"]["env_suffix_token_count"], 0)
        self.assertEqual(result.metadata["envpack"]["trainable_token_count"], 2)
        self.assertEqual(result.metadata["vagen"]["adapter"], "envpack")
        self.assertEqual(result.metadata["vagen"]["env_name"], "sokoban")
        self.assertEqual(result.metadata["vagen"]["env_reward"], 1.0)
        self.assertEqual(result.metadata["vagen"]["num_turns"], 1)
        self.assertEqual(result.metadata["vagen"]["per_turn"][0]["reward"], 1.0)
        self.assertTrue(result.metadata["vagen"]["traj_success"])
        self.assertEqual(calls[0][2], {"X-SMG-Routing-Key": result.session_id})
        self.assertEqual(bundle.client.cancelled, [])
        self.assertEqual(bundle.client.step_requests[0].expected_turn_id, 0)

    def test_generate_refills_same_prompt_after_system_failure(self) -> None:
        if torch is None:
            self.skipTest("requires torch because Miles Sample imports torch")

        client = _StepFailsOnceClient()
        bundle = _FakeBundle(client=client)
        calls = []

        async def fake_post(url, payload, headers=None):
            calls.append((url, payload, headers))
            return {
                "meta_info": {
                    "output_token_logprobs": [(-0.1, 101), (-0.2, 102)],
                    "finish_reason": {"type": "stop"},
                    "weight_version": "7",
                    "prompt_tokens": len(payload["input_ids"]),
                    "cached_tokens": 1,
                }
            }

        sample = Sample(
            group_index=3,
            index=9,
            metadata={
                "envpack": {
                    "env_name": "sokoban",
                    "seed": 123,
                    "pool_id": "sokoban-vision",
                    "env_uuid": "init-sha",
                }
            },
        )
        args = SimpleNamespace(
            envpack={
                "api": "in_process",
                "pools": [{"env": "sokoban", "profile": "vision_free_think_local", "pool_id": "sokoban-vision"}],
                "refill": {"max_attempts": 2, "backoff_s": 0},
                "rollout": {"max_turns": 1, "response_length_per_turn": 8},
            },
            sglang_router_ip="127.0.0.1",
            sglang_router_port=30000,
            sglang_router_policy="consistent_hashing",
            sglang_speculative_algorithm=None,
            apply_chat_template_kwargs={},
            hf_checkpoint="/model",
            chat_template_path=None,
            rollout_max_context_len=64,
            partial_rollout=False,
            group_rm=False,
            rm_type=None,
            custom_rm_path=None,
            rollout_external=False,
            use_rollout_routing_replay=False,
            use_miles_router=False,
            miles_router_middleware_paths=[],
        )
        state = SimpleNamespace(args=args, tokenizer=_Tokenizer(), processor=None)
        fake_envpack = _fake_envpack_modules()

        with patch.dict(sys.modules, fake_envpack):
            with patch.object(generate_mod, "get_client_bundle", return_value=bundle):
                with patch.object(generate_mod, "post", side_effect=fake_post):
                    with patch.object(generate_mod, "is_lora_enabled", return_value=False):
                        with patch.object(
                            generate_mod, "encode_image_for_rollout_engine", side_effect=lambda image: image
                        ):
                            with patch.object(generate_mod, "observation_to_chat_message") as render_obs:
                                render_obs.return_value = RenderedObservation(
                                    message={"role": "user", "content": "init"},
                                    images=[],
                                    videos=[],
                                    media_hashes=["init-sha"],
                                    artifacts=[],
                                )
                                with self.assertLogs("miles_plugins.envpack_adapter.generate", level="WARNING"):
                                    output = asyncio.run(
                                        generate_mod.generate(
                                            GenerateFnInput(
                                                state=state,
                                                sample=sample,
                                                sampling_params={"max_new_tokens": 16},
                                                evaluation=False,
                                            )
                                        )
                                    )

        result = output.samples
        self.assertIs(result, sample)
        self.assertEqual(result.status, Sample.Status.COMPLETED)
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(result.group_index, 3)
        self.assertEqual(result.index, 9)
        self.assertEqual(len(client.created_requests), 2)
        self.assertEqual(client.created_requests[0].seed, client.created_requests[1].seed)
        self.assertEqual(client.created_requests[0].env_config, client.created_requests[1].env_config)
        self.assertNotEqual(client.created_requests[0].episode_id, client.created_requests[1].episode_id)
        self.assertEqual(client.cancelled, [(client.created_requests[0].episode_id, "miles_adapter_cleanup")])
        self.assertEqual(len(calls), 2)
        refill = result.metadata["envpack"]["refill"]
        self.assertEqual(refill["attempt_index"], 1)
        self.assertEqual(refill["max_attempts"], 2)
        self.assertEqual(refill["failed_attempts"][0]["error_type"], "_RetryableEnvpackError")

    def test_generate_does_not_refill_plain_runtime_error(self) -> None:
        if torch is None:
            self.skipTest("requires torch because Miles Sample imports torch")

        client = _BuggyStepClient()
        bundle = _FakeBundle(client=client)

        async def fake_post(url, payload, headers=None):
            return {
                "meta_info": {
                    "output_token_logprobs": [(-0.1, 101), (-0.2, 102)],
                    "finish_reason": {"type": "stop"},
                    "weight_version": "7",
                    "prompt_tokens": len(payload["input_ids"]),
                    "cached_tokens": 1,
                }
            }

        sample = Sample(
            metadata={
                "envpack": {
                    "env_name": "sokoban",
                    "seed": 123,
                    "pool_id": "sokoban-vision",
                    "env_uuid": "init-sha",
                }
            },
        )
        args = SimpleNamespace(
            envpack={
                "api": "in_process",
                "pools": [{"env": "sokoban", "profile": "vision_free_think_local", "pool_id": "sokoban-vision"}],
                "refill": {"max_attempts": 2, "backoff_s": 0},
                "rollout": {"max_turns": 1, "response_length_per_turn": 8},
            },
            sglang_router_ip="127.0.0.1",
            sglang_router_port=30000,
            sglang_router_policy="consistent_hashing",
            sglang_speculative_algorithm=None,
            apply_chat_template_kwargs={},
            hf_checkpoint="/model",
            chat_template_path=None,
            rollout_max_context_len=64,
            partial_rollout=False,
            group_rm=False,
            rm_type=None,
            custom_rm_path=None,
            rollout_external=False,
            use_rollout_routing_replay=False,
            use_miles_router=False,
            miles_router_middleware_paths=[],
        )
        state = SimpleNamespace(args=args, tokenizer=_Tokenizer(), processor=None)
        fake_envpack = _fake_envpack_modules()

        with patch.dict(sys.modules, fake_envpack):
            with patch.object(generate_mod, "get_client_bundle", return_value=bundle):
                with patch.object(generate_mod, "post", side_effect=fake_post):
                    with patch.object(generate_mod, "is_lora_enabled", return_value=False):
                        with patch.object(
                            generate_mod, "encode_image_for_rollout_engine", side_effect=lambda image: image
                        ):
                            with patch.object(generate_mod, "observation_to_chat_message") as render_obs:
                                render_obs.return_value = RenderedObservation(
                                    message={"role": "user", "content": "init"},
                                    images=[],
                                    videos=[],
                                    media_hashes=["init-sha"],
                                    artifacts=[],
                                )
                                with self.assertRaisesRegex(RuntimeError, "adapter bug"):
                                    asyncio.run(
                                        generate_mod.generate(
                                            GenerateFnInput(
                                                state=state,
                                                sample=sample,
                                                sampling_params={"max_new_tokens": 16},
                                                evaluation=False,
                                            )
                                        )
                                    )

        self.assertEqual(len(client.created_requests), 1)
        self.assertEqual(len(client.step_requests), 1)

    def test_generate_does_not_refill_terminal_outcome_unknown(self) -> None:
        if torch is None:
            self.skipTest("requires torch because Miles Sample imports torch")

        client = _TerminalFinalizeFailsClient()
        bundle = _FakeBundle(client=client)

        async def fake_post(url, payload, headers=None):
            return {
                "meta_info": {
                    "output_token_logprobs": [(-0.1, 101), (-0.2, 102)],
                    "finish_reason": {"type": "stop"},
                    "weight_version": "7",
                    "prompt_tokens": len(payload["input_ids"]),
                    "cached_tokens": 1,
                }
            }

        sample = Sample(
            metadata={
                "envpack": {
                    "env_name": "sokoban",
                    "seed": 123,
                    "pool_id": "sokoban-vision",
                    "env_uuid": "init-sha",
                }
            },
        )
        args = SimpleNamespace(
            envpack={
                "api": "in_process",
                "pools": [{"env": "sokoban", "profile": "vision_free_think_local", "pool_id": "sokoban-vision"}],
                "refill": {"max_attempts": 2, "backoff_s": 0},
                "rollout": {"max_turns": 1, "response_length_per_turn": 8},
            },
            sglang_router_ip="127.0.0.1",
            sglang_router_port=30000,
            sglang_router_policy="consistent_hashing",
            sglang_speculative_algorithm=None,
            apply_chat_template_kwargs={},
            hf_checkpoint="/model",
            chat_template_path=None,
            rollout_max_context_len=64,
            partial_rollout=False,
            group_rm=False,
            rm_type=None,
            custom_rm_path=None,
            rollout_external=False,
            use_rollout_routing_replay=False,
            use_miles_router=False,
            miles_router_middleware_paths=[],
        )
        state = SimpleNamespace(args=args, tokenizer=_Tokenizer(), processor=None)
        fake_envpack = _fake_envpack_modules()

        with patch.dict(sys.modules, fake_envpack):
            with patch.object(generate_mod, "get_client_bundle", return_value=bundle):
                with patch.object(generate_mod, "post", side_effect=fake_post):
                    with patch.object(generate_mod, "is_lora_enabled", return_value=False):
                        with patch.object(
                            generate_mod, "encode_image_for_rollout_engine", side_effect=lambda image: image
                        ):
                            with patch.object(generate_mod, "observation_to_chat_message") as render_obs:
                                render_obs.return_value = RenderedObservation(
                                    message={"role": "user", "content": "init"},
                                    images=[],
                                    videos=[],
                                    media_hashes=["init-sha"],
                                    artifacts=[],
                                )
                                with self.assertRaisesRegex(RuntimeError, "unknown outcome"):
                                    asyncio.run(
                                        generate_mod.generate(
                                            GenerateFnInput(
                                                state=state,
                                                sample=sample,
                                                sampling_params={"max_new_tokens": 16},
                                                evaluation=False,
                                            )
                                        )
                                    )

        # A lost terminal outcome must fail loud: the rollout is not rerun from reset.
        self.assertEqual(len(client.created_requests), 1)
        self.assertEqual(len(client.step_requests), 1)
        refill = sample.metadata["envpack"]["refill"]
        self.assertFalse(refill["exhausted"])
        self.assertFalse(refill["failed_attempts"][0]["retryable"])


def _fake_envpack_modules():
    core_mod = types.ModuleType("envpack.core")

    class ActorOutput(_KwargsObject):
        pass

    class EpisodeCreateRequest(_KwargsObject):
        pass

    class EpisodeStepRequest(_KwargsObject):
        pass

    core_mod.ActorOutput = ActorOutput
    core_mod.EpisodeCreateRequest = EpisodeCreateRequest
    core_mod.EpisodeStepRequest = EpisodeStepRequest
    package_mod = types.ModuleType("envpack")
    package_mod.core = core_mod
    return {"envpack": package_mod, "envpack.core": core_mod}


if __name__ == "__main__":
    unittest.main()
