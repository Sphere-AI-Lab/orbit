from sglang.srt.server_args import ServerArgs
from miles.utils.http_utils import _wrap_ipv6


# ORBIT-SEAM: repo-wide comment-style pass (TODO -> Follow-up), no functional change
# Follow-up: use all sglang router arguments with `--sglang-router` prefix
def add_sglang_router_arguments(parser):
    """
    Add arguments to the parser for the SGLang router.
    """
    parser.add_argument(
        "--sglang-router-ip",
        type=str,
        default=None,
        help="IP address of the SGLang router",
    )
    parser.add_argument(
        "--sglang-router-port",
        type=int,
        default=None,
        help="Port of the SGLang router",
    )
    parser.add_argument(
        "--sglang-router-policy",
        type=str,
        default=None,
        help="Routing policy for the SGLang router (e.g., 'consistent_hashing', 'round_robin')",
    )
    parser.add_argument(
        "--sglang-router-request-timeout-secs",
        type=int,
        default=14400,
        help="Timeout for requests to the SGLang router in seconds",
    )
    return parser


def add_sglang_arguments(parser):
    """
    Add arguments to the parser for the SGLang server.
    """
    parser = add_sglang_router_arguments(parser)
    parser.add_argument("--sglang-server-concurrency", type=int, default=512)

    old_add_argument = parser.add_argument

    skipped_args = [
        "model_path",
        "config",
        "trust_remote_code",
        "random_seed",
        # memory
        "enable_memory_saver",
        # distributed
        "tp_size",
        "port",
        "nnodes",
        "node_rank",
        "dist_init_addr",
        "gpu_id_step",
        "base_gpu_id",
        "nccl_port",
        "skip_server_warmup",
        "enable_return_routed_experts",
        # ORBIT-SEAM: skip auto-mirroring sglang's --enforce-piecewise-cuda-graph so orbit's own
        # manually-added --sglang-enforce-piecewise-cuda-graph override (added below) doesn't collide
        # sglang v0.5.14 upstream (PR #28919) added --enforce-piecewise-cuda-graph
        # to ServerArgs; orbit also manually adds --sglang-enforce-piecewise-cuda-graph
        # below as a colocate override. Skip the auto-mirror so the manual override
        # wins instead of colliding (argparse conflicting option string).
        "enforce_piecewise_cuda_graph",
    ]

    def new_add_argument_wrapper(*name_or_flags, **kwargs):
        """
        Add arguments to the parser, ensuring that the server arguments are prefixed and skippable.
        """
        # ORBIT-SEAM: base only checked the flag-derived name; this now also checks the explicit
        # dest (any() over both) so a deprecated-alias arg whose dest points at a different real
        # field (e.g. --enforce-piecewise-cuda-graph -> cuda_graph_backend_prefill) can be skipped too
        # Determine the canonical name(s) for skip check (e.g., "model_path").
        # Check BOTH the explicit dest AND the flag-derived name: sglang v0.5.14
        # deprecated-alias args (e.g. --enforce-piecewise-cuda-graph) carry an
        # explicit dest pointing at the REAL field (cuda_graph_backend_prefill),
        # so a dest-only check can't skip them by their own flag name.
        canonical_names_for_skip_check = []
        if "dest" in kwargs and isinstance(kwargs["dest"], str):
            canonical_names_for_skip_check.append(kwargs["dest"])
        for flag_name_candidate in name_or_flags:
            if isinstance(flag_name_candidate, str) and flag_name_candidate.startswith("--"):
                # Derive from first long flag: --foo-bar -> foo_bar
                canonical_names_for_skip_check.append(flag_name_candidate[2:].replace("-", "_"))
                break

        if any(n in skipped_args for n in canonical_names_for_skip_check):
            return  # Skip this entire argument definition

        # If not skipped, proceed to prefix flags and dest
        new_name_or_flags_list = []
        for item_flag in name_or_flags:
            if isinstance(item_flag, str) and item_flag.startswith("-"):
                original_flag_stem = item_flag.lstrip("-")  # "foo-bar" from "--foo-bar", or "f" from "-f"
                prefixed_item = f"--sglang-{original_flag_stem}"
                new_name_or_flags_list.append(prefixed_item)
            else:
                # Positional arguments or non-string items
                new_name_or_flags_list.append(item_flag)

        # Prepare kwargs for the actual add_argument call.
        # Make a copy to avoid modifying the original kwargs dict.
        final_kwargs = kwargs.copy()

        # If 'dest' is explicitly provided and is a string, prefix it.
        # This ensures the attribute on the args namespace becomes, e.g., args.sglang_dest_name.
        if "dest" in final_kwargs and isinstance(final_kwargs["dest"], str):
            original_dest = final_kwargs["dest"]
            # Avoid double prefixing if dest somehow already starts with sglang_
            if not original_dest.startswith("sglang_"):
                final_kwargs["dest"] = f"sglang_{original_dest}"
        # If 'dest' is not explicitly provided (or is None/not a string),
        # argparse will derive 'dest' from the (now prefixed) flag names.
        # E.g., if the first flag is "--sglang-foo-bar", argparse sets dest to "sglang_foo_bar".

        old_add_argument(*new_name_or_flags_list, **final_kwargs)

    parser.add_argument = new_add_argument_wrapper
    ServerArgs.add_cli_args(parser)
    parser.add_argument = old_add_argument

    # ORBIT-SEAM: two orbit-only CLI overrides - a colocate piecewise-cuda-graph keep-enabled flag,
    # and a force-native-ops compatibility flag for spawned rollout servers
    parser.add_argument(
        "--sglang-enforce-piecewise-cuda-graph",
        action="store_true",
        default=False,
        help=(
            "Orbit colocate override: keep SGLang piecewise CUDA graph enabled "
            "when colocate mode would otherwise disable it."
        ),
    )
    parser.add_argument(
        "--sglang-force-native-ops",
        action="store_true",
        default=False,
        help=(
            "Orbit compatibility override: force selected SGLang MultiPlatformOp "
            "layers onto PyTorch-native forwards inside spawned rollout servers."
        ),
    )
    parser.add_argument(
        "--sglang-config",
        type=str,
        default=None,
        help=(
            "Path to a YAML config for SGLang engine deployment. "
            "Defines server_groups with worker_type (regular/prefill/decode/placeholder), "
            "num_gpus per group, and optional per-group 'overrides' dict of "
            "ServerArgs field names that override the base --sglang-* CLI args. "
            "Placeholder groups reserve GPU slots without creating engines. "
            "Mutually exclusive with --prefill-num-servers."
        ),
    )

    return parser


# ORBIT-SEAM: new function - defaults prefill CUDA graphs off (unusable with orbit's colocate
# memory-saver mode and OFT adapters; see docstring) and enforces the restriction under OFT
def apply_prefill_cuda_graph_policy(args) -> None:
    """Default ``--sglang-cuda-graph-backend-prefill`` to "disabled"; reject any
    other backend under ``--peft-method oft``.

    Prefill CUDA graphs arrived with the sglang v0.5.16 merge, defaulting to the
    "breakable" backend. Phase-0 qualification (2026-08-21, 4xB200) showed that
    backend is unusable for orbit's engines: it refuses memory-saver mode (every
    --colocate engine fails at startup) and its graph replay does not apply OFT
    adapters (NaN logits at the first sample; "tc_piecewise" trips torch.compile
    in the OFT layers). Default to the pre-merge envelope -- no prefill graphs --
    so every arm of a systems comparison runs the same engine config; users may
    opt back in explicitly with ``--sglang-cuda-graph-backend-prefill <backend>``.
    """
    backend = getattr(args, "sglang_cuda_graph_backend_prefill", None)
    if backend is None:
        args.sglang_cuda_graph_backend_prefill = "disabled"
    elif backend != "disabled" and getattr(args, "peft_method", "none") == "oft":
        raise ValueError(
            f"--sglang-cuda-graph-backend-prefill {backend!r} is not supported with "
            "--peft-method oft: the prefill CUDA-graph replay does not apply OFT adapters "
            "(NaN logits). Use 'disabled' (the default)."
        )


def validate_args(args):
    args.sglang_tp_size = args.rollout_num_gpus_per_engine

    # ORBIT-SEAM: applies the prefill-cuda-graph default/OFT guard above; the true-on-policy
    # fallback and attn_cp/moe_dp derivation below replace base's direct
    # sglang_dp_size/pp_size/ep_size assignments (v0.5.14's auto-mirror now produces those directly)
    apply_prefill_cuda_graph_policy(args)

    # Fallback net: --true-on-policy-mode can be set directly, bypassing the
    # --true-on-policy parse-time expansion (orbit/true_on_policy/config.py),
    # which is the primary path that forces this (miles parity).
    if args.true_on_policy_mode:
        args.sglang_enable_deterministic_inference = True

    # sglang v0.5.14 ServerArgs fields are dp_size/pp_size/ep_size (the old
    # data_parallel_size/pipeline_parallel_size/expert_parallel_size are CLI
    # aliases only), so the auto-mirror already produces args.sglang_dp_size /
    # sglang_pp_size / sglang_ep_size directly -- used as-is below (v0.5.14-only).
    args.sglang_attn_cp_size = getattr(
        args,
        "sglang_attn_cp_size",
        getattr(args, "sglang_attention_context_parallel_size", 1),
    )
    args.sglang_moe_dp_size = getattr(
        args,
        "sglang_moe_dp_size",
        getattr(args, "sglang_moe_data_parallel_size", 1),
    )

    if args.sglang_dp_size > 1:
        assert args.sglang_enable_dp_attention

    # ORBIT-SEAM: attention/MoE topology divisibility checks + effective-tp-size bookkeeping, new
    # for the attn_cp_size/moe_dp_size dimensions added above (base validated only dp_attention)
    sglang_attention_dp_size = args.sglang_dp_size if args.sglang_enable_dp_attention else 1
    sglang_attention_group_size = sglang_attention_dp_size * args.sglang_attn_cp_size
    if args.sglang_tp_size % sglang_attention_group_size != 0:
        raise ValueError(
            "SGLang attention topology must divide rollout_num_gpus_per_engine: "
            f"tp_size={args.sglang_tp_size}, attention_dp_size={sglang_attention_dp_size}, "
            f"attn_cp_size={args.sglang_attn_cp_size}"
        )
    args.sglang_effective_attention_tp_size = args.sglang_tp_size // sglang_attention_group_size

    sglang_moe_group_size = args.sglang_ep_size * args.sglang_moe_dp_size
    if args.sglang_tp_size % sglang_moe_group_size != 0:
        raise ValueError(
            "SGLang MoE topology must divide rollout_num_gpus_per_engine: "
            f"tp_size={args.sglang_tp_size}, ep_size={args.sglang_ep_size}, "
            f"moe_dp_size={args.sglang_moe_dp_size}"
        )
    args.sglang_effective_moe_tp_size = args.sglang_tp_size // sglang_moe_group_size

    if args.sglang_router_policy:
        from miles.utils.environ import enable_experimental_rollout_refactor

        assert (
            not enable_experimental_rollout_refactor()
        ), "--sglang-router-policy is not supported with MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1"

    if getattr(args, "sglang_router_ip", None):
        args.sglang_router_ip = _wrap_ipv6(args.sglang_router_ip)
