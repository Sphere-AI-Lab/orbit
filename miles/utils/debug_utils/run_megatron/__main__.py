from miles.utils.debug_utils.run_megatron.cli import app

# ORBIT-SEAM: arms orbit's patches; this debug entrypoint reaches the patched
# load_tokenizer, and a debug run that silently differs from the real one is worse
# than useless. Costs this file its pristine status.
import orbit  # noqa: F401,E402  -- arming side effect only


if __name__ == "__main__":
    app()
