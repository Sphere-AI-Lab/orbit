"""Task-index dataset exporter for legacy Tau-bench."""

import argparse
import json
from pathlib import Path

ALL_DATA_MAPPINGS = {"retail": ["train", "test", "dev"], "airline": ["test"]}


def export_tasks(output_dir: str, *, domains: dict[str, list[str]] | None = None) -> None:
    from tau_bench.envs import get_env
    from tau_bench.types import RunConfig

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config = RunConfig(model_provider="mock", user_model_provider="mock", user_strategy="human", model="mock")

    for env_name, splits in (domains or ALL_DATA_MAPPINGS).items():
        for split in splits:
            config.env = env_name
            config.task_split = split
            env_instance = get_env(
                env_name=config.env,
                user_strategy=config.user_strategy,
                user_model=config.user_model,
                task_split=config.task_split,
                task_index=0,
            )
            path = output_path / f"{env_name}_{split}_tasks.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for idx, task in enumerate(env_instance.tasks):
                    handle.write(json.dumps({"index": idx, "metadata": task.model_dump()}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Tau-bench task-index JSONL files.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    export_tasks(args.output_dir)


if __name__ == "__main__":
    main()
