from __future__ import annotations

import multiprocessing as mp
import os
import re
import sys
import threading
from math import isclose
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

from miles.utils.metric_utils import compute_pass_rate

# sympy's equals/simplify/cancel/solve can run for unbounded time on adversarial
# inputs; latex2sympy internally invokes sympy.solve so even parsing can hang.
# Run symbolic comparison in a child process so native/parser code that holds the
# GIL cannot block the rollout event loop from enforcing the deadline.
_SYMBOLIC_EQUAL_TIMEOUT_S = 5.0
_MAX_SYMBOLIC_COMPARE_CHARS = 2048
_SYMBOLIC_EQUAL_MP_CONTEXT_ENV = "ORBIT_SYMBOLIC_EQUAL_MP_CONTEXT"
_DEFAULT_SYMBOLIC_EQUAL_MAX_PROCESSES = 8
_SYMBOLIC_EQUAL_MAX_PROCESSES_ENV = "ORBIT_SYMBOLIC_EQUAL_MAX_PROCESSES"
_SYMBOLIC_PROCESS_SEMAPHORES: dict[int, threading.BoundedSemaphore] = {}
_SYMBOLIC_PROCESS_SEMAPHORES_LOCK = threading.Lock()

_SUPPORTED_DATASETS = {"math500", "math250", "aime24", "amc23"}


def _ensure_vendored_math_eval_on_path() -> None:
    # Walk up from this file so the anchor survives moves within the repo
    # (a fixed parents[N] silently broke when this module moved homes).
    for root in Path(__file__).resolve().parents:
        math_eval_dir = root / "examples" / "peft_arena" / "backend" / "third_party" / "math_eval"
        if math_eval_dir.is_dir():
            math_eval_path = str(math_eval_dir)
            if math_eval_path not in sys.path:
                sys.path.insert(0, math_eval_path)
            return


_ensure_vendored_math_eval_on_path()
from latex2sympy.latex2sympy2 import latex2sympy  # noqa: E402


def build_cot_prompt(problem: str) -> str:
    return f"Question: {problem.rstrip()}\nAnswer:"


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if not substr:
                return string
            if substr[0] == "{":
                new_str += substr
                continue
            if len(substr) < 2:
                return string
            a = substr[0]
            b = substr[1]
            if b != "{":
                if len(substr) > 2:
                    post_substr = substr[2:]
                    new_str += "{" + a + "}{" + b + "}" + post_substr
                else:
                    new_str += "{" + a + "}{" + b + "}"
            else:
                if len(substr) > 2:
                    post_substr = substr[2:]
                    new_str += "{" + a + "}" + b + post_substr
                else:
                    new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a_i = int(a)
        b_i = int(b)
        if string == f"{a_i}/{b_i}":
            return f"\\frac{{{a_i}}}{{{b_i}}}"
    except Exception:
        return string
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            return string
        if split[0] != "{":
            new_substr = "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(value: Optional[str]) -> str:
    if value is None:
        return ""
    string = str(value).strip()
    string = re.sub(r"√\s*([0-9]+(?:\.[0-9]+)?)", r"\\sqrt{\1}", string)
    string = string.replace("√", "\\sqrt")
    string = string.replace("π", "\\pi")
    string = string.replace("∞", "\\infty")
    string = string.replace("−", "-")
    string = string.replace("∈", "\\in")
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if string.startswith("."):
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = re.sub(r"^[a-zA-Z]\\in(?=[\[\(])", "", string)
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string.strip().rstrip(".").rstrip("/")


def extract_answer(pred_str: str, data_name: str, use_last_number: bool = True) -> str:
    pred_str = pred_str.replace("\u043a\u0438", "")

    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        pred = pred_str.split("final answer is $", 1)[1].split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if not ans:
            return ""
        if ans[0] == "{":
            stack = 1
            extracted = ""
            for ch in ans[1:]:
                if ch == "{":
                    stack += 1
                    extracted += ch
                elif ch == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    extracted += ch
                else:
                    extracted += ch
            pred = extracted
        else:
            pred = ans.split("$")[0].strip()
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "\u7b54\u6848\u662f" in pred_str:
        pred = pred_str.split("\u7b54\u6848\u662f")[1].strip().split("\n\n")[0].strip()
    else:
        if use_last_number:
            candidates = re.findall(r"-?\d*\.?\d+", pred_str.replace(",", ""))
            pred = candidates[-1] if candidates else ""
        else:
            pred = ""

    if data_name not in _SUPPORTED_DATASETS:
        raise NotImplementedError(f"Unsupported math_alignment dataset {data_name}")

    pred = re.sub(r"\n\s*", "", pred)
    if pred.startswith(":"):
        pred = pred[1:]
    return strip_string(pred)


def parse_ground_truth(label: Optional[str], metadata: Optional[dict[str, Any]]) -> tuple[str, str]:
    metadata = metadata or {}
    data_name = str(metadata.get("dataset_name", "")).strip()
    if data_name not in _SUPPORTED_DATASETS:
        raise NotImplementedError(f"Unsupported math_alignment dataset {data_name}")

    gt_cot = ""
    if data_name in {"math500", "math250"}:
        gt_cot = str(metadata.get("solution", "")).strip()
        gt_ans = extract_answer(gt_cot, data_name) if gt_cot else ""
        if not gt_ans:
            gt_ans = str(metadata.get("answer", label or "")).strip()
    else:
        gt_ans = str(metadata.get("answer", label or "")).strip()

    return gt_cot, strip_string(gt_ans)


def _parse_digits(num: str):
    num = re.sub(",", "", str(num))
    try:
        return float(num)
    except Exception:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except Exception:
                pass
    return None


def _is_digit(num: str) -> bool:
    return _parse_digits(num) is not None


def _is_nontrivial_symbolic_relation(expr: str) -> bool:
    if "=" not in expr:
        return False
    lhs, _ = expr.split("=", 1)
    lhs = lhs.strip()
    if len(lhs) <= 2:
        return False
    return True


def _plain_number_cannot_equal_symbolic_relation(prediction: str, reference: str) -> bool:
    return (_is_digit(prediction) and _is_nontrivial_symbolic_relation(reference)) or (
        _is_digit(reference) and _is_nontrivial_symbolic_relation(prediction)
    )


def _too_long_for_symbolic_compare(*values: str) -> bool:
    return any(len(value) > _MAX_SYMBOLIC_COMPARE_CHARS for value in values)


def _numeric_equal(prediction: float, reference: float) -> bool:
    return isclose(reference, prediction, rel_tol=1e-4)


def _str_to_pmatrix(input_str: str) -> str:
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []
    for matrix in matrix_str:
        matrix = matrix.strip("{}")
        pmatrix_list.append(r"\begin{pmatrix}" + matrix.replace(",", "\\") + r"\end{pmatrix}")
    return ", ".join(pmatrix_list)


def _symbolic_mp_context():
    method = os.getenv(_SYMBOLIC_EQUAL_MP_CONTEXT_ENV, "fork")
    try:
        return mp.get_context(method)
    except ValueError:
        return mp.get_context()


def _symbolic_max_processes() -> int:
    value = os.getenv(_SYMBOLIC_EQUAL_MAX_PROCESSES_ENV, str(_DEFAULT_SYMBOLIC_EQUAL_MAX_PROCESSES))
    try:
        max_processes = int(value)
    except (TypeError, ValueError):
        return _DEFAULT_SYMBOLIC_EQUAL_MAX_PROCESSES
    return max(1, max_processes)


def _symbolic_process_semaphore(max_processes: int) -> threading.BoundedSemaphore:
    with _SYMBOLIC_PROCESS_SEMAPHORES_LOCK:
        semaphore = _SYMBOLIC_PROCESS_SEMAPHORES.get(max_processes)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(max_processes)
            _SYMBOLIC_PROCESS_SEMAPHORES[max_processes] = semaphore
        return semaphore


def _symbolic_equal_process_entry(a: str, b: str, conn) -> None:
    try:
        conn.send(("ok", bool(_symbolic_equal_impl(a, b))))
    except BaseException as exc:
        try:
            conn.send(("error", exc))
        except BaseException as send_error:
            conn.send(("error", RuntimeError(f"{type(exc).__name__}: {exc}; failed to serialize: {send_error}")))
    finally:
        conn.close()


def _stop_process(process) -> None:
    process.terminate()
    process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)


def _close_process(process) -> None:
    try:
        process.close()
    except ValueError:
        pass


def _symbolic_equal_in_process(a: str, b: str, timeout: float) -> bool:
    with _symbolic_process_semaphore(_symbolic_max_processes()):
        ctx = _symbolic_mp_context()
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_symbolic_equal_process_entry, args=(a, b, child_conn))
        process.daemon = True
        process.start()
        child_conn.close()
        try:
            process.join(timeout=timeout)
            if process.is_alive():
                _stop_process(process)
                raise TimeoutError
            if not parent_conn.poll():
                raise RuntimeError(f"symbolic compare subprocess exited with code {process.exitcode} without a result")
            status, payload = parent_conn.recv()
            if status == "ok":
                return payload
            raise payload
        finally:
            parent_conn.close()
            _close_process(process)


def _symbolic_equal(a: str, b: str) -> bool:
    """Time-bounded wrapper around the symbolic equivalence check.

    Returns False on timeout. See _SYMBOLIC_EQUAL_TIMEOUT_S for rationale.
    """
    try:
        return _symbolic_equal_in_process(a, b, _SYMBOLIC_EQUAL_TIMEOUT_S)
    except TimeoutError:
        return False
    except Exception:
        return False


def _symbolic_equal_impl(a: str, b: str) -> bool:
    def _parse(expr: str):
        for parse_fn in (parse_latex, parse_expr, latex2sympy):
            try:
                return parse_fn(expr.replace("\\\\", "\\"))
            except Exception:
                try:
                    return parse_fn(expr)
                except Exception:
                    pass
        return expr

    lhs = _parse(a)
    rhs = _parse(b)
    try:
        if str(lhs) == str(rhs) or lhs == rhs:
            return True
    except Exception:
        pass
    try:
        if lhs.equals(rhs) or simplify(lhs - rhs) == 0:
            return True
    except Exception:
        pass
    try:
        if (abs(lhs.lhs - lhs.rhs)).equals(abs(rhs.lhs - rhs.rhs)):
            return True
    except Exception:
        pass
    try:
        if _numeric_equal(float(N(lhs)), float(N(rhs))):
            return True
    except Exception:
        pass
    try:
        if lhs.shape == rhs.shape:
            if lhs.applyfunc(lambda x: round(x, 3)).equals(rhs.applyfunc(lambda x: round(x, 3))):
                return True
    except Exception:
        pass
    return False


def math_equal(
    prediction: Union[bool, float, str, None],
    reference: Union[float, str, None],
    *,
    include_percentage: bool = True,
    is_close: bool = True,
) -> bool:
    if prediction is None or reference is None:
        return False
    if str(prediction).strip().lower() == str(reference).strip().lower():
        return True

    try:
        if _is_digit(str(prediction)) and _is_digit(str(reference)):
            pred_value = _parse_digits(str(prediction))
            ref_value = _parse_digits(str(reference))
            if pred_value is None or ref_value is None:
                return False
            candidates = [ref_value / 100, ref_value, ref_value * 100] if include_percentage else [ref_value]
            for candidate in candidates:
                if is_close:
                    if _numeric_equal(pred_value, candidate):
                        return True
                elif pred_value == candidate:
                    return True
            return False
    except Exception:
        pass

    prediction_str = strip_string(str(prediction).strip())
    reference_str = strip_string(str(reference).strip())
    if not prediction_str:
        return False

    if prediction_str.lower() == reference_str.lower():
        return True

    if "pmatrix" in prediction_str and "pmatrix" not in reference_str:
        reference_str = _str_to_pmatrix(reference_str)

    pred_cmp = prediction_str
    ref_cmp = reference_str
    if (
        prediction_str.startswith("[")
        and prediction_str.endswith("]")
        and not reference_str.startswith("(")
    ) or (
        prediction_str.startswith("(")
        and prediction_str.endswith(")")
        and not reference_str.startswith("[")
    ):
        pred_cmp = pred_cmp.strip("[]()")
        ref_cmp = ref_cmp.strip("[]()")
    for char in ("{", "}", "(", ")"):
        pred_cmp = pred_cmp.replace(char, "")
        ref_cmp = ref_cmp.replace(char, "")
    if pred_cmp.lower() == ref_cmp.lower():
        return True

    if _too_long_for_symbolic_compare(prediction_str, reference_str):
        return False

    if (
        re.match(r"(\(|\[).+(\)|\])", prediction_str) is not None
        and re.match(r"(\(|\[).+(\)|\])", reference_str) is not None
    ):
        pred_parts = prediction_str[1:-1].split(",")
        ref_parts = reference_str[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            math_equal(pred_parts[i], ref_parts[i], include_percentage=include_percentage, is_close=is_close)
            for i in range(len(pred_parts))
        ):
            return True

    if prediction_str.count("=") == 1 and reference_str.count("=") == 1:
        pred_eq = prediction_str.split("=")
        pred_expr = f"{pred_eq[0].strip()} - ({pred_eq[1].strip()})"
        ref_eq = reference_str.split("=")
        ref_expr = f"{ref_eq[0].strip()} - ({ref_eq[1].strip()})"
        if _symbolic_equal(pred_expr, ref_expr) or _symbolic_equal(f"-({pred_expr})", ref_expr):
            return True
    elif prediction_str.count("=") == 1 and len(prediction_str.split("=")[0].strip()) <= 2 and "=" not in reference_str:
        if math_equal(prediction_str.split("=")[1], reference_str, include_percentage=include_percentage, is_close=is_close):
            return True
    elif reference_str.count("=") == 1 and len(reference_str.split("=")[0].strip()) <= 2 and "=" not in prediction_str:
        if math_equal(prediction_str, reference_str.split("=")[1], include_percentage=include_percentage, is_close=is_close):
            return True

    if _plain_number_cannot_equal_symbolic_relation(prediction_str, reference_str):
        return False

    return _symbolic_equal(prediction_str, reference_str)


def grade_math_alignment(response: str, label: Optional[str], metadata: Optional[dict[str, Any]]) -> bool:
    metadata = metadata or {}
    data_name = str(metadata.get("dataset_name", "")).strip()
    prediction = extract_answer(response, data_name)
    _, ground_truth = parse_ground_truth(label, metadata)
    return math_equal(prediction, ground_truth)


def is_math_alignment_sample(sample) -> bool:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return (metadata.get("rm_type") or "").strip() == "math_alignment"


def compute_math_alignment_metrics(
    flat_rewards: list[float],
    group_size: int,
    *,
    k_values: Optional[list[int]] = None,
) -> dict[str, float]:
    if not flat_rewards:
        return {}

    metrics: dict[str, float] = {
        "acc": float(np.mean(flat_rewards) * 100.0),
    }
    if group_size <= 1:
        return metrics

    rewards_of_group = np.array(flat_rewards).reshape(len(flat_rewards) // group_size, group_size)
    metrics["pass_acc"] = float(np.mean(np.any(rewards_of_group == 1, axis=1)) * 100.0)
    metrics |= compute_pass_rate(
        flat_rewards=flat_rewards,
        group_size=group_size,
        k_values=k_values,
        scale=100.0,
    )
    return metrics
