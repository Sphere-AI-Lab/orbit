"""CPU leg of the teacher-logprob equivalence harness (instrumentation I-5).

Pins the numerical equivalences behind orbit's adapter-as-teacher OPD claims
("the base teacher is free", "an adapter teacher is exact") for the
trainer-side plans returned by teacher_forward_plan and dispatched by
MegatronTrainRayActor.compute_teacher_log_probs:

  * alias_ref    — teacher_log_probs ARE the ref logprobs: same list object,
                   no forward runs (the base teacher is free when the ref
                   forward already ran).
  * adapter_off  — one forward under peft.disable_adapter(model) is bitwise
                   equal (CPU float32) to an adapter-free twin built from the
                   same base weights.
  * adapter_swap — one forward inside swap_adapter_tensors is bitwise equal
                   to a module directly constructed with the teacher tensors,
                   and the live student adapter is restored bitwise after.

Already covered elsewhere (deliberately not repeated here):
  * plan-matrix routing (which spec picks which plan): tests/test_opd_teacher_spec.py
  * swap restore-on-exception, key/shape-mismatch validation, multi-chunk
    independence, base params untouched: tests/test_adapter_swap.py

GPU/SGLang leg (runbook only — NOT implemented here):
  The same comparison utility (orbit.utils.logprob_compare) closes the loop
  against a live engine:
    1. launch trainer + SGLang engine with --opd-teacher <spec>;
    2. trainer side: compute_teacher_log_probs on a fixed sampled-token batch
       (plans adapter_off / adapter_swap), collect per-token teacher_log_probs;
    3. engine side: teacher-forcing prefill of the same token sequences —
       against base weights (a request with no lora_path) for base teachers,
       against the reserved OPD_TEACHER_ADAPTER_NAME ("orbit_teacher") slot
       for adapter/self teachers after promotion;
    4. compare_logprob_dicts(trainer_side, engine_side) keyed by sample id,
       gated on summarize_reports(...).within(atol) with a documented
       cross-stack bf16 tolerance — bitwise exactness is claimed only within
       one stack; a megatron-vs-sglang gap is numerics, not an equivalence bug.
"""

from argparse import Namespace
from contextlib import contextmanager
from types import MethodType

import pytest
import torch

from megatron.bridge.peft.base import PEFT as BridgePEFT

from miles.backends.megatron_utils import actor as actor_utils
from orbit.utils.adapter_swap import swap_adapter_tensors
from orbit.utils.logprob_compare import compare_logprobs
from orbit.opd.opd_teacher_spec import TeacherSpec

VOCAB = 7
DIM = 5
SEQ = 6
_ADAPTER_KEY = (0, "wrapped.adapter.delta")  # matches the real is_adapter_param_name (".adapter.")


def _fixtures():
    gen = torch.Generator().manual_seed(1234)
    hidden = torch.randn(SEQ, DIM, generator=gen, dtype=torch.float32)
    tokens = torch.randint(0, VOCAB, (SEQ,), generator=gen)
    base_weight = torch.randn(VOCAB, DIM, generator=gen, dtype=torch.float32)
    student_delta = 0.1 * torch.randn(VOCAB, DIM, generator=gen, dtype=torch.float32)
    teacher_delta = 0.1 * torch.randn(VOCAB, DIM, generator=gen, dtype=torch.float32)
    return hidden, tokens, base_weight, student_delta, teacher_delta


class _ToyAdapter(torch.nn.Module):
    def __init__(self, delta: torch.Tensor):
        super().__init__()
        self.delta = torch.nn.Parameter(delta.clone())

    def forward(self, x):
        return x @ self.delta.T


class _ToyAdapterWrapper(torch.nn.Module):
    """Mimics megatron.bridge.peft.adapter_wrapper.AdapterWrapper's toggle contract.

    to_wrap/adapter submodules, an _adapter_enabled flag flipped by
    enable_adapter_layers()/disable_adapter_layers(); while disabled the
    forward returns only the base module's output.
    """

    def __init__(self, to_wrap: torch.nn.Module, adapter: torch.nn.Module):
        super().__init__()
        self.to_wrap = to_wrap
        self.adapter = adapter
        self._adapter_enabled = True

    def enable_adapter_layers(self):
        self._adapter_enabled = True

    def disable_adapter_layers(self):
        self._adapter_enabled = False

    def forward(self, x):
        out = self.to_wrap(x)
        if self._adapter_enabled:
            out = out + self.adapter(x)
        return out


class _AdaptedLM(torch.nn.Module):
    def __init__(self, base_weight: torch.Tensor, delta: torch.Tensor):
        super().__init__()
        linear = torch.nn.Linear(DIM, VOCAB, bias=False)
        with torch.no_grad():
            linear.weight.copy_(base_weight)
        self.wrapped = _ToyAdapterWrapper(linear, _ToyAdapter(delta))

    def forward(self, h):
        return self.wrapped(h)


class _BaseOnlyLM(torch.nn.Module):
    """Adapter-free twin: same base weights, no adapter modules anywhere."""

    def __init__(self, base_weight: torch.Tensor):
        super().__init__()
        self.linear = torch.nn.Linear(DIM, VOCAB, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(base_weight)

    def forward(self, h):
        return self.linear(h)


def _token_logprobs(model: torch.nn.Module, hidden: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    logits = model(hidden)
    logprobs = torch.log_softmax(logits, dim=-1)
    return logprobs[torch.arange(tokens.numel()), tokens]


class _ToyPeftMimic:
    """Faithful mimic of megatron.bridge.peft.base.PEFT.disable_adapter.

    The bridge contract as dispatched by compute_teacher_log_probs: walk every
    module of every chunk, call disable_adapter_layers()/enable_adapter_layers()
    wherever callable, and re-enable in a finally block.
    """

    def _walk(self, model, method_name: str) -> None:
        chunks = model if isinstance(model, list) else [model]
        for chunk in chunks:
            for module in chunk.modules():
                method = getattr(module, method_name, None)
                if callable(method):
                    method()

    def disable_adapter_layers(self, model) -> None:
        self._walk(model, "disable_adapter_layers")

    def enable_adapter_layers(self, model) -> None:
        self._walk(model, "enable_adapter_layers")

    @contextmanager
    def disable_adapter(self, model):
        try:
            self.disable_adapter_layers(model)
            yield
        finally:
            self.enable_adapter_layers(model)


class _NoopTransformBridgePeft(BridgePEFT):
    """The REAL bridge PEFT.disable_adapter context manager (no-op transform).

    The toy model is pre-wrapped with _ToyAdapterWrapper, which honors the
    AdapterWrapper enable/disable protocol the bridge walk relies on.
    """

    def transform(self, module, name=None, prefix=None):
        return module


def _make_actor(monkeypatch, *, spec, model=None, teacher_tensors=None, self_teacher=None):
    """Actor-level scaffold in the style of tests/fast/test_actor_ref_restore.py."""
    monkeypatch.setattr(actor_utils, "all_replay_managers", [])
    actor = object.__new__(actor_utils.MegatronTrainRayActor)
    actor.args = Namespace(peft_method="lora", opd_type="megatron")  # real is_peft_enabled reads this
    actor._opd_teacher_spec = spec
    actor._opd_teacher_tensors = teacher_tensors
    actor._self_teacher = self_teacher
    actor.model = model
    return actor


def _install_toy_compute_log_prob(actor, hidden, tokens):
    """Stand-in honoring the real compute_log_prob contract.

    Same signature and same return shape ({f"{store_prefix}log_probs":
    [per-sample tensors]}), but scoring the toy model — so the surrounding
    adapter toggling in compute_teacher_log_probs is exercised for real.
    """

    def _compute_log_prob(self, data_iterator, num_microbatches, store_prefix=""):
        return {f"{store_prefix}log_probs": [_token_logprobs(self.model[0], hidden, tokens)]}

    actor.compute_log_prob = MethodType(_compute_log_prob, actor)


# ---------------------------------------------------------------------------
# (a) alias_ref: teacher_log_probs ARE the ref logprobs
# ---------------------------------------------------------------------------


def test_alias_ref_returns_the_ref_logprob_list_object_and_runs_no_forward(monkeypatch):
    actor = _make_actor(monkeypatch, spec=TeacherSpec("base"))
    actor.compute_log_prob = MethodType(
        lambda self, *args, **kwargs: pytest.fail("alias_ref must not run a forward"), actor
    )

    ref_list = [torch.tensor([-0.5, -1.25, -2.0]), torch.tensor([-3.0])]
    ref_data = {"ref_log_probs": ref_list}
    out = actor.compute_teacher_log_probs([], [], ref_data=ref_data)

    assert set(out) == {"teacher_log_probs"}
    # Pinned: the ref list is ALIASED, not copied. A future clone/detach/dtype
    # cast (an accidental transformation) must break this assertion.
    assert out["teacher_log_probs"] is ref_list
    for got, want in zip(out["teacher_log_probs"], ref_list, strict=True):
        report = compare_logprobs(want, got)
        assert report.count == want.numel()
        assert report.max_abs_diff == 0.0


# ---------------------------------------------------------------------------
# (b) adapter_off: disabled-adapter forward == adapter-free twin, bitwise
# ---------------------------------------------------------------------------


def test_disable_adapter_context_reproduces_base_forward_bitwise():
    hidden, tokens, base_weight, student_delta, _ = _fixtures()
    adapted = _AdaptedLM(base_weight, student_delta)
    base_only = _BaseOnlyLM(base_weight)

    enabled = _token_logprobs(adapted, hidden, tokens)
    with _ToyPeftMimic().disable_adapter([adapted]):
        disabled = _token_logprobs(adapted, hidden, tokens)
    reenabled = _token_logprobs(adapted, hidden, tokens)
    base = _token_logprobs(base_only, hidden, tokens)

    assert torch.equal(disabled, base)  # bitwise: the base teacher is exact
    assert not torch.equal(enabled, base)  # the adapter genuinely contributes
    assert torch.equal(reenabled, enabled)  # the context restores the enabled state


def test_disable_adapter_reenables_after_exception():
    hidden, tokens, base_weight, student_delta, _ = _fixtures()
    adapted = _AdaptedLM(base_weight, student_delta)
    enabled = _token_logprobs(adapted, hidden, tokens)

    with pytest.raises(RuntimeError, match="boom"):
        with _ToyPeftMimic().disable_adapter([adapted]):
            raise RuntimeError("boom")

    assert adapted.wrapped._adapter_enabled is True
    assert torch.equal(_token_logprobs(adapted, hidden, tokens), enabled)


def test_compute_teacher_log_probs_adapter_off_matches_base_only_twin(monkeypatch):
    """Through the real actor branch, with the REAL bridge PEFT.disable_adapter."""
    hidden, tokens, base_weight, student_delta, _ = _fixtures()
    adapted = _AdaptedLM(base_weight, student_delta)
    enabled = _token_logprobs(adapted, hidden, tokens)
    base = _token_logprobs(_BaseOnlyLM(base_weight), hidden, tokens)

    actor = _make_actor(monkeypatch, spec=TeacherSpec("base"), model=[adapted])
    monkeypatch.setattr(actor_utils, "create_peft_instance", lambda args: _NoopTransformBridgePeft())
    _install_toy_compute_log_prob(actor, hidden, tokens)

    out = actor.compute_teacher_log_probs([], [], ref_data=None)  # no ref: plan adapter_off

    assert set(out) == {"teacher_log_probs"}
    assert torch.equal(out["teacher_log_probs"][0], base)
    # Adapter re-enabled after the teacher forward: student scoring is intact.
    assert torch.equal(_token_logprobs(adapted, hidden, tokens), enabled)


# ---------------------------------------------------------------------------
# (c) adapter_swap: swapped forward == directly-built teacher module, bitwise
# ---------------------------------------------------------------------------


def test_swap_forward_bitwise_matches_directly_built_teacher_module():
    hidden, tokens, base_weight, student_delta, teacher_delta = _fixtures()
    student = _AdaptedLM(base_weight, student_delta)
    direct_teacher = _AdaptedLM(base_weight, teacher_delta)
    student_delta_before = student.wrapped.adapter.delta.detach().clone()
    before = _token_logprobs(student, hidden, tokens)

    with swap_adapter_tensors([student], {_ADAPTER_KEY: teacher_delta}, lambda name: ".adapter." in name):
        swapped = _token_logprobs(student, hidden, tokens)
    restored = _token_logprobs(student, hidden, tokens)

    assert torch.equal(swapped, _token_logprobs(direct_teacher, hidden, tokens))  # bitwise: exact teacher
    assert torch.equal(restored, before)  # bitwise restore of the student forward
    assert torch.equal(student.wrapped.adapter.delta, student_delta_before)  # bitwise restore of the params


def test_compute_teacher_log_probs_adapter_swap_matches_direct_teacher_and_restores(monkeypatch):
    """Through the real actor branch: real swap util, real is_adapter_param_name."""
    hidden, tokens, base_weight, student_delta, teacher_delta = _fixtures()
    student = _AdaptedLM(base_weight, student_delta)
    direct_teacher_logprobs = _token_logprobs(_AdaptedLM(base_weight, teacher_delta), hidden, tokens)
    student_delta_before = student.wrapped.adapter.delta.detach().clone()
    student_logprobs_before = _token_logprobs(student, hidden, tokens)

    actor = _make_actor(
        monkeypatch,
        spec=TeacherSpec("adapter", "/ckpts/teacher_adapter"),
        model=[student],
        teacher_tensors={_ADAPTER_KEY: teacher_delta},
    )
    _install_toy_compute_log_prob(actor, hidden, tokens)

    out = actor.compute_teacher_log_probs([], [], ref_data=None)

    assert set(out) == {"teacher_log_probs"}
    assert torch.equal(out["teacher_log_probs"][0], direct_teacher_logprobs)
    assert torch.equal(student.wrapped.adapter.delta, student_delta_before)
    assert torch.equal(_token_logprobs(student, hidden, tokens), student_logprobs_before)


def test_compute_teacher_log_probs_self_teacher_tensors_take_the_same_swap_path(monkeypatch):
    """self:* teachers swap the self-teacher buffer tensors: same exactness."""
    hidden, tokens, base_weight, student_delta, teacher_delta = _fixtures()
    student = _AdaptedLM(base_weight, student_delta)
    direct_teacher_logprobs = _token_logprobs(_AdaptedLM(base_weight, teacher_delta), hidden, tokens)

    class _SelfTeacherStub:
        tensors = {_ADAPTER_KEY: teacher_delta}

    actor = _make_actor(
        monkeypatch,
        spec=TeacherSpec("self_ema"),
        model=[student],
        teacher_tensors=None,  # forces the self-teacher fallback branch
        self_teacher=_SelfTeacherStub(),
    )
    _install_toy_compute_log_prob(actor, hidden, tokens)

    out = actor.compute_teacher_log_probs([], [], ref_data=None)

    assert torch.equal(out["teacher_log_probs"][0], direct_teacher_logprobs)
