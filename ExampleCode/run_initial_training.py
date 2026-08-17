#!/usr/bin/env python3
"""
Execution runtime for the full Optiver initial-training Notebook.

The runtime does not implement daily participant fine-tuning or on-board
adaptation. It executes the slow structural training run in which the
Transformer, K recurrent agents, softmax gate, and transition head are
optimized jointly.

Typical usage:

    python run_initial_training.py \
        --notebook 01_full_market_simulator_initial_training_optiver.ipynb \
        --project-root /content/drive/MyDrive/bigalpha_market_project \
        --output-dir /content/drive/MyDrive/bigalpha_market_project/runs/run_001

Use --validate-only to inspect the Notebook contract without training.
"""


from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError
except ImportError as exc:
    raise SystemExit(
        "Missing runtime packages. Install them with:\n"
        "python -m pip install nbformat nbclient jupyter-client ipykernel"
    ) from exc


EXPECTED_FEATURES = [
    "bid_px_rel_1",
    "ask_px_rel_1",
    "log_bid_size_1",
    "log_ask_size_1",
    "bid_px_rel_2",
    "ask_px_rel_2",
    "log_bid_size_2",
    "log_ask_size_2",
    "spread",
    "imbalance_1",
    "return_1",
    "spread_change",
    "imbalance_change",
    "log_trade_size",
    "log_trade_count",
    "real_book_update",
    "trade_occurred",
    "seconds_since_last_real_update",
    "bucket_progress",
]

EXPECTED_TARGETS = [
    "next_return",
    "next_spread_change",
    "next_imbalance_change",
    "next_log_total_depth_change",
    "next_log_trade_size",
    "next_log_trade_count",
]

REQUIRED_NOTEBOOK_MARKERS = [
    "class ConfigurableMarketEncoder",
    "class VectorizedAgentPopulation",
    "class MarketTransitionHead",
    "class MarketSimulationModel",
    "def sequential_objective",
    "def train_model",
    "def evaluate_loader",
    "validate_optiver_raw_schema",
    "processed_schema_version",
    "joint_full_model_initial_training",
    "weighted_aggregate_action_only",
]

FORBIDDEN_RUNTIME_MARKERS = [
    "def bounded_onboard_tune",
    "def set_onboard_trainable_scope",
]


@dataclass
class RuntimeOptions:
    notebook: str
    project_root: str
    output_dir: str
    kernel_name: str
    timeout_seconds: int
    offline: bool
    validate_only: bool
    resume: bool
    install_packages: bool
    batch_size: int | None
    num_workers: int | None
    max_epochs: int | None
    max_train_steps: int | None
    max_validation_batches: int | None
    stock_limit: int | None
    dataset_fraction: float | None
    use_torch_compile: bool | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def notebook_text(notebook: Any) -> str:
    return "\n".join(
        str(cell.get("source", ""))
        for cell in notebook.cells
    )


def validate_notebook_contract(notebook: Any) -> dict[str, Any]:
    text = notebook_text(notebook)
    errors: list[str] = []
    warnings: list[str] = []

    for marker in REQUIRED_NOTEBOOK_MARKERS:
        if marker not in text:
            errors.append(
                f"Missing required Notebook marker: {marker}"
            )

    for marker in FORBIDDEN_RUNTIME_MARKERS:
        if marker in text:
            errors.append(
                "Initial-training Notebook contains runtime adaptation code: "
                f"{marker}"
            )

    feature_match = re.search(
        r"FEATURES\s*=\s*\[(.*?)\]\s*\n\s*TARGETS",
        text,
        flags=re.S,
    )

    target_match = re.search(
        r"TARGETS\s*=\s*\[(.*?)\]\s*\n",
        text,
        flags=re.S,
    )

    if feature_match:
        detected_features = re.findall(
            r'"([^"]+)"',
            feature_match.group(1),
        )

        if detected_features != EXPECTED_FEATURES:
            errors.append(
                "The Notebook feature list does not match the current "
                "19-feature Optiver contract."
            )
    else:
        errors.append("Could not locate FEATURES in the Notebook.")

    if target_match:
        detected_targets = re.findall(
            r'"([^"]+)"',
            target_match.group(1),
        )

        if detected_targets != EXPECTED_TARGETS:
            errors.append(
                "The Notebook target list does not match the current "
                "six-target Optiver contract."
            )
    else:
        errors.append("Could not locate TARGETS in the Notebook.")

    if "if t < cfg.warmup_len:" not in text:
        errors.append(
            "The sequential objective does not contain the corrected "
            "96-position warmup boundary."
        )

    if "bucket_sizes != 599" not in text:
        errors.append(
            "The processed-data validation does not enforce 599 rows "
            "per Optiver bucket."
        )

    if "cfg.sequence_len" not in text:
        warnings.append(
            "Could not confirm that sampled windows use cfg.sequence_len."
        )

    if errors:
        raise ValueError(
            "Notebook contract validation failed:\n"
            + "\n".join(
                f"{index}. {message}"
                for index, message in enumerate(
                    errors,
                    start=1,
                )
            )
        )

    return {
        "feature_count": len(EXPECTED_FEATURES),
        "target_count": len(EXPECTED_TARGETS),
        "warmup_positions": 96,
        "supervised_positions": 32,
        "alignment_rows": 1,
        "sequence_length": 129,
        "processed_rows_per_bucket": 599,
        "warnings": warnings,
    }


def replace_assignment(
    source: str,
    name: str,
    expression: str,
) -> str:
    pattern = re.compile(
        rf"^{re.escape(name)}\s*=\s*.*$",
        flags=re.M,
    )

    if not pattern.search(source):
        raise ValueError(
            f"Could not find assignment for {name}."
        )

    return pattern.sub(
        f"{name} = {expression}",
        source,
        count=1,
    )


def patch_notebook(
    notebook: Any,
    options: RuntimeOptions,
) -> Any:
    patched = copy.deepcopy(notebook)

    override_lines: list[str] = [
        "# Runtime-injected configuration.",
        f"cfg.batch_size = {options.batch_size!r}"
        if options.batch_size is not None
        else "",
        f"cfg.num_workers = {options.num_workers!r}"
        if options.num_workers is not None
        else "",
        f"cfg.max_epochs = {options.max_epochs!r}"
        if options.max_epochs is not None
        else "",
        (
            "cfg.max_train_steps_per_epoch = "
            f"{options.max_train_steps!r}"
        )
        if options.max_train_steps is not None
        else "",
        (
            "cfg.max_validation_batches = "
            f"{options.max_validation_batches!r}"
        )
        if options.max_validation_batches is not None
        else "",
        f"cfg.stock_limit = {options.stock_limit!r}"
        if options.stock_limit is not None
        else "",
        (
            "cfg.dataset_fraction = "
            f"{options.dataset_fraction!r}"
        )
        if options.dataset_fraction is not None
        else "",
        (
            "cfg.use_torch_compile = "
            f"{options.use_torch_compile!r}"
        )
        if options.use_torch_compile is not None
        else "",
        'print("Runtime overrides applied.")',
        "display(pd.Series(asdict(cfg), name='runtime_value').to_frame())",
    ]

    override_source = "\n".join(
        line
        for line in override_lines
        if line
    )

    config_injected = False
    path_patched = False
    resume_patched = False
    checkpoint_cleanup_disabled = False
    install_cell_handled = False

    for cell in patched.cells:
        if cell.cell_type != "code":
            continue

        source = str(cell.source)

        if (
            "@dataclass" in source
            and "class Config" in source
            and "cfg = Config()" in source
        ):
            cell.source = (
                source
                + "\n\n"
                + override_source
                + "\n"
            )
            config_injected = True
            continue

        if (
            "DRIVE_PROJECT_ROOT = Path(" in source
            and "KAGGLE_JSON" in source
        ):
            project_expression = (
                f"Path({options.project_root!r})"
            )

            source = re.sub(
                r'DRIVE_PROJECT_ROOT\s*=\s*Path\([^\n]+\)',
                f"DRIVE_PROJECT_ROOT = {project_expression}",
                source,
                count=1,
            )

            if options.offline:
                source = source.replace(
                    'if not ARCHIVE_PATH.exists():\n'
                    '    print("Downloading Optiver archive...")\n'
                    '    !kaggle competitions download \\\n'
                    '        -c optiver-realized-volatility-prediction \\\n'
                    '        -p "{DRIVE_DOWNLOAD_ROOT}"\n'
                    'else:\n'
                    '    print("Archive already exists in Drive.")',
                    'if not ARCHIVE_PATH.exists():\n'
                    '    raise FileNotFoundError(\n'
                    '        "Offline mode requires the Optiver archive at: "\n'
                    '        f"{ARCHIVE_PATH}"\n'
                    '    )\n'
                    'else:\n'
                    '    print("Offline archive found in project storage.")',
                )

            cell.source = source
            path_patched = True
            continue

        if (
            source.lstrip().startswith("!pip")
            and not options.install_packages
        ):
            lines = source.splitlines()
            retained = [
                line
                for line in lines
                if not line.lstrip().startswith("!pip")
            ]
            cell.source = "\n".join(retained)
            install_cell_handled = True
            continue

        if "RESUME_TRAINING = False" in source:
            cell.source = source.replace(
                "RESUME_TRAINING = False",
                f"RESUME_TRAINING = {options.resume}",
            )
            resume_patched = True
            continue

        if (
            "Remove checkpoints produced by the NaN run"
            in source
        ):
            if options.resume:
                cell.source = (
                    "# Checkpoint cleanup disabled by runtime because "
                    "--resume was requested.\n"
                    "RESUME_TRAINING = True\n"
                )
                checkpoint_cleanup_disabled = True
            continue

    if not config_injected:
        raise ValueError(
            "Could not inject runtime configuration into the Notebook."
        )

    if not path_patched:
        raise ValueError(
            "Could not patch DRIVE_PROJECT_ROOT in the Notebook."
        )

    if not resume_patched:
        raise ValueError(
            "Could not patch RESUME_TRAINING in the Notebook."
        )

    patched.metadata.setdefault(
        "runtime",
        {},
    )

    patched.metadata["runtime"].update(
        {
            "generated_at_utc": utc_now(),
            "project_root": options.project_root,
            "offline": options.offline,
            "resume": options.resume,
            "install_packages":
                options.install_packages,
            "checkpoint_cleanup_disabled":
                checkpoint_cleanup_disabled,
            "pip_install_removed":
                install_cell_handled,
        }
    )

    return patched


def ensure_runtime_inputs(
    options: RuntimeOptions,
) -> dict[str, str]:
    project_root = Path(
        options.project_root
    ).expanduser()

    archive_path = (
        project_root
        / "data"
        / "downloads"
        / "optiver-realized-volatility-prediction.zip"
    )

    processed_root = (
        project_root
        / "data"
        / "processed"
        / "optiver_1s_six_target_v2"
    )

    checkpoint_root = (
        project_root
        / "checkpoints"
    )

    checkpoint_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if options.offline:
        if (
            not archive_path.exists()
            and not processed_root.exists()
        ):
            raise FileNotFoundError(
                "Offline execution requires either:\n"
                f"1. {archive_path}\n"
                f"2. {processed_root}"
            )

    return {
        "project_root":
            str(project_root),
        "archive_path":
            str(archive_path),
        "processed_root":
            str(processed_root),
        "checkpoint_root":
            str(checkpoint_root),
    }


def execute_notebook(
    notebook: Any,
    output_path: Path,
    *,
    kernel_name: str,
    timeout_seconds: int,
    working_directory: Path,
) -> None:
    working_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = NotebookClient(
        notebook,
        timeout=timeout_seconds,
        kernel_name=kernel_name,
        allow_errors=False,
        resources={
            "metadata": {
                "path":
                    str(working_directory)
            }
        },
    )

    client.execute()

    nbformat.write(
        notebook,
        output_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and execute the complete Optiver initial-training "
            "Notebook."
        )
    )

    parser.add_argument(
        "--notebook",
        default=(
            "01_full_market_simulator_"
            "initial_training_optiver.ipynb"
        ),
    )

    parser.add_argument(
        "--project-root",
        required=True,
        help=(
            "Project storage root. In Colab this is normally "
            "/content/drive/MyDrive/bigalpha_market_project."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory for the patched Notebook, executed Notebook, "
            "runtime log, and manifest."
        ),
    )

    parser.add_argument(
        "--kernel-name",
        default="python3",
    )

    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=21600,
        help=(
            "Per-cell timeout. The default is six hours."
        ),
    )

    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Disable Kaggle download. The archive or processed dataset "
            "must already exist."
        ),
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--install-packages",
        action="store_true",
        help=(
            "Retain the Notebook pip-install line. Omit this flag in a "
            "prebuilt competition runtime."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
    )

    parser.add_argument(
        "--max-train-steps",
        type=int,
    )

    parser.add_argument(
        "--max-validation-batches",
        type=int,
    )

    parser.add_argument(
        "--stock-limit",
        type=int,
    )

    parser.add_argument(
        "--dataset-fraction",
        type=float,
    )

    parser.add_argument(
        "--use-torch-compile",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    return parser


def main() -> int:
    parser = build_parser()
    namespace = parser.parse_args()

    options = RuntimeOptions(
        notebook=namespace.notebook,
        project_root=namespace.project_root,
        output_dir=namespace.output_dir,
        kernel_name=namespace.kernel_name,
        timeout_seconds=namespace.timeout_seconds,
        offline=namespace.offline,
        validate_only=namespace.validate_only,
        resume=namespace.resume,
        install_packages=namespace.install_packages,
        batch_size=namespace.batch_size,
        num_workers=namespace.num_workers,
        max_epochs=namespace.max_epochs,
        max_train_steps=namespace.max_train_steps,
        max_validation_batches=(
            namespace.max_validation_batches
        ),
        stock_limit=namespace.stock_limit,
        dataset_fraction=namespace.dataset_fraction,
        use_torch_compile=namespace.use_torch_compile,
    )

    notebook_path = Path(
        options.notebook
    ).expanduser().resolve()

    output_dir = Path(
        options.output_dir
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        output_dir
        / "initial_training_manifest.json"
    )

    patched_path = (
        output_dir
        / "initial_training_patched.ipynb"
    )

    executed_path = (
        output_dir
        / "initial_training_executed.ipynb"
    )

    log_path = (
        output_dir
        / "initial_training_runtime.log"
    )

    manifest: dict[str, Any] = {
        "runtime_started_utc": utc_now(),
        "runtime_options": asdict(options),
        "status": "starting",
    }

    try:
        if not notebook_path.exists():
            raise FileNotFoundError(
                f"Notebook not found: {notebook_path}"
            )

        manifest["notebook_sha256"] = (
            sha256_file(notebook_path)
        )

        with notebook_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            notebook = nbformat.read(
                handle,
                as_version=4,
            )

        contract = validate_notebook_contract(
            notebook
        )

        manifest["notebook_contract"] = contract
        manifest["runtime_inputs"] = (
            ensure_runtime_inputs(options)
        )

        patched = patch_notebook(
            notebook,
            options,
        )

        nbformat.write(
            patched,
            patched_path,
        )

        manifest["patched_notebook"] = (
            str(patched_path)
        )

        if options.validate_only:
            manifest["status"] = "validated"
            manifest["runtime_finished_utc"] = (
                utc_now()
            )

            manifest_path.write_text(
                json.dumps(
                    manifest,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(
                json.dumps(
                    manifest,
                    indent=2,
                )
            )

            return 0

        start = time.time()

        execute_notebook(
            patched,
            executed_path,
            kernel_name=options.kernel_name,
            timeout_seconds=(
                options.timeout_seconds
            ),
            working_directory=output_dir,
        )

        manifest["status"] = "completed"
        manifest["elapsed_seconds"] = (
            time.time() - start
        )

        manifest["executed_notebook"] = (
            str(executed_path)
        )

        manifest["executed_notebook_sha256"] = (
            sha256_file(executed_path)
        )

        manifest["runtime_finished_utc"] = (
            utc_now()
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

        log_path.write_text(
            "Initial training completed successfully.\n"
            f"Executed Notebook: {executed_path}\n"
            f"Manifest: {manifest_path}\n",
            encoding="utf-8",
        )

        print(
            json.dumps(
                manifest,
                indent=2,
            )
        )

        return 0

    except (
        CellExecutionError,
        Exception,
    ) as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = (
            type(exc).__name__
        )
        manifest["error"] = str(exc)
        manifest["traceback"] = (
            traceback.format_exc()
        )
        manifest["runtime_finished_utc"] = (
            utc_now()
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

        log_path.write_text(
            manifest["traceback"],
            encoding="utf-8",
        )

        print(
            manifest["traceback"],
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
