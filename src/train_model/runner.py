from __future__ import annotations

import copy
import hashlib
import json
import platform
import re
import subprocess
import time
import uuid
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent, MODELS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from train_model.config import SUPPORTED_ALGORITHMS, validate_config
from train_model.data import load_train_val_data


def build_environment(frame, environment_config: dict[str, Any]) -> StockTradingEnv:
    indicators = list(environment_config["tech_indicator_list"])
    stock_dimension = frame["tic"].nunique()
    env_kwargs = {
        "df": frame,
        "hmax": int(environment_config["hmax"]),
        "initial_amount": float(environment_config["initial_amount"]),
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [float(environment_config["buy_cost_pct"])] * stock_dimension,
        "sell_cost_pct": [float(environment_config["sell_cost_pct"])] * stock_dimension,
        "reward_scaling": float(environment_config["reward_scaling"]),
        "state_space": 1 + 2 * stock_dimension + len(indicators) * stock_dimension,
        "stock_dim": stock_dimension,
        "action_space": stock_dimension,
        "tech_indicator_list": indicators,
        "risk_indicator_col": environment_config["risk_indicator_col"],
        "print_verbosity": int(environment_config.get("print_verbosity", 10)),
    }
    if environment_config.get("turbulence_threshold") is not None:
        env_kwargs["turbulence_threshold"] = float(
            environment_config["turbulence_threshold"]
        )
    return StockTradingEnv(**env_kwargs)


def build_model(
    algorithm: str,
    vector_environment,
    model_kwargs: dict[str, Any],
    seed: int,
    tensorboard_log: str | Path,
    device: str = "auto",
    policy: str = "MlpPolicy",
    verbose: int = 0,
):
    normalized_algorithm = algorithm.lower()
    if normalized_algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    params = copy.deepcopy(model_kwargs)
    params["device"] = device
    return DRLAgent(env=vector_environment).get_model(
        normalized_algorithm,
        policy=policy,
        model_kwargs=params,
        seed=int(seed),
        tensorboard_log=str(tensorboard_log),
        verbose=int(verbose),
    )


def _resolve_path(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def _slug(value: Any) -> str:
    if isinstance(value, float):
        value = format(value, ".8g")
    text = str(value).replace("-", "m").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9]+", "", text) or "none"


def _compact_count(value: int) -> str:
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000}m"
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}k"
    return str(value)


def build_model_stem(
    algorithm: str,
    seed: int,
    total_timesteps: int,
    model_kwargs: dict[str, Any],
) -> str:
    visible_keys = {
        "ppo": (
            ("learning_rate", "lr"),
            ("n_steps", "n"),
            ("batch_size", "b"),
            ("ent_coef", "ent"),
        ),
        "ddpg": (
            ("learning_rate", "lr"),
            ("batch_size", "b"),
            ("buffer_size", "buf"),
            ("learning_starts", "start"),
        ),
        "td3": (
            ("learning_rate", "lr"),
            ("batch_size", "b"),
            ("buffer_size", "buf"),
            ("learning_starts", "start"),
        ),
        "sac": (
            ("learning_rate", "lr"),
            ("batch_size", "b"),
            ("buffer_size", "buf"),
            ("learning_starts", "start"),
            ("ent_coef", "ent"),
        ),
    }
    normalized_algorithm = algorithm.lower()
    if normalized_algorithm not in visible_keys:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    parts = [normalized_algorithm, f"s{seed}", f"t{_compact_count(total_timesteps)}"]
    for key, label in visible_keys[normalized_algorithm]:
        if key in model_kwargs:
            value = model_kwargs[key]
            if key == "buffer_size":
                value = _compact_count(int(value))
            parts.append(f"{label}{_slug(value)}")
    return "_".join(parts)


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _git_revision(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_archive_metadata(path: Path, metadata_content: dict[str, Any]) -> None:
    content = "FinRL T04 model metadata\n\n"
    content += json.dumps(metadata_content, indent=2, sort_keys=True, default=str)
    with zipfile.ZipFile(path, mode="a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.txt", content)


def run_training(
    config: dict[str, Any],
    dataset_name: str,
    algorithm: str,
    seed: int,
    project_root: str | Path,
    total_timesteps: int | None = None,
    run_type: str = "main",
    device: str | None = None,
) -> dict[str, Any]:
    validate_config(config)
    algorithm = algorithm.lower()
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    if run_type not in ("main", "pilot", "smoke"):
        raise ValueError("run_type must be one of: main, pilot, smoke")

    training_config = config["training"]
    configured_timesteps = int(training_config["total_timesteps"])
    effective_timesteps = int(
        configured_timesteps if total_timesteps is None else total_timesteps
    )
    if effective_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if effective_timesteps != configured_timesteps and run_type == "main":
        raise ValueError("Changing total_timesteps requires run_type='pilot' or 'smoke'")

    if (
        not dataset_name
        or Path(dataset_name).name != dataset_name
        or "/" in dataset_name
        or "\\" in dataset_name
        or dataset_name in (".", "..")
    ):
        raise ValueError("dataset_name must be a single directory name")

    root = Path(project_root).resolve()
    data_config = config["data"]
    dataset_dir = _resolve_path(root, data_config["root_dir"]) / dataset_name
    file_names = (data_config["train_file"], data_config["val_file"], data_config["test_file"])
    if any(
        Path(file_name).name != file_name or "/" in file_name or "\\" in file_name
        for file_name in file_names
    ):
        raise ValueError("Dataset CSV configuration values must be file names, not paths")
    train_path = dataset_dir / data_config["train_file"]
    val_path = dataset_dir / data_config["val_file"]
    test_path = dataset_dir / data_config["test_file"]
    environment_config = copy.deepcopy(config["environment"])
    model_kwargs = copy.deepcopy(config["algorithms"][algorithm])
    effective_device = device or training_config.get("device", "auto")
    train_frame, val_frame = load_train_val_data(
        train_path,
        val_path,
        environment_config["tech_indicator_list"],
        environment_config["risk_indicator_col"],
    )
    data_hashes = {"train_csv": _sha256(train_path), "val_csv": _sha256(val_path)}
    configuration_fingerprint = _json_hash(
        {
            "algorithm": algorithm,
            "seed": int(seed),
            "total_timesteps": effective_timesteps,
            "model_kwargs": model_kwargs,
            "environment": environment_config,
            "data_hashes": data_hashes,
            "dataset_name": dataset_name,
        }
    )
    model_stem = build_model_stem(
        algorithm, int(seed), effective_timesteps, model_kwargs
    )
    run_id = f"{model_stem}__run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"

    output_config = config["output"]
    model_dir = _resolve_path(root, output_config["trained_models_dir"]) / dataset_name / algorithm
    tensorboard_dir = _resolve_path(root, output_config["tensorboard_log_dir"]) / dataset_name / algorithm / run_id
    result_dir = _resolve_path(root, output_config["results_dir"]) / dataset_name / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_stem}.zip"
    if model_path.exists():
        raise FileExistsError(
            f"Model archive already exists; refusing to overwrite: {model_path}"
        )
    manifest_path = result_dir / "run_manifest.json"

    effective_config = {
        "algorithm": algorithm,
        "dataset_name": dataset_name,
        "seed": int(seed),
        "run_type": run_type,
        "total_timesteps": effective_timesteps,
        "policy": training_config["policy"],
        "device": effective_device,
        "model_kwargs": model_kwargs,
        "environment": environment_config,
        "data_paths": {
            "train_csv": str(train_path),
            "val_csv": str(val_path),
            "test_csv": str(test_path),
        },
        "data_hashes": data_hashes,
        "configuration_fingerprint": configuration_fingerprint,
    }
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "effective_config": effective_config,
        "artifacts": {
            "model_zip": str(model_path),
            "tensorboard_log_dir": str(tensorboard_dir),
            "results_dir": str(result_dir),
            "manifest": str(manifest_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_revision": _git_revision(root),
            "packages": {
                "finrl": _package_version("finrl"),
                "stable-baselines3": _package_version("stable-baselines3"),
                "torch": _package_version("torch"),
                "pandas": _package_version("pandas"),
            },
        },
    }
    _write_json(result_dir / "effective_config.json", effective_config)
    _write_json(manifest_path, manifest)
    started = time.perf_counter()

    try:
        train_environment = build_environment(train_frame, environment_config)
        validation_environment = build_environment(val_frame, environment_config)
        vector_environment, _ = train_environment.get_sb_env()
        model = build_model(
            algorithm=algorithm,
            vector_environment=vector_environment,
            model_kwargs=model_kwargs,
            seed=int(seed),
            tensorboard_log=tensorboard_dir,
            device=effective_device,
            policy=training_config["policy"],
            verbose=int(training_config.get("verbose", 1)),
        )
        logger = configure(
            str(tensorboard_dir),
            training_config.get("logger_formats", ["stdout", "csv", "tensorboard"]),
        )
        model.set_logger(logger)
        trained_model = DRLAgent.train_model(
            model=model,
            tb_log_name=run_id,
            total_timesteps=effective_timesteps,
        )
        trained_model.save(str(model_path.with_suffix("")))
        if not model_path.is_file():
            raise RuntimeError(f"SB3 model archive was not created: {model_path}")

        reloaded_model = MODELS[algorithm].load(
            str(model_path), device=effective_device
        )
        account_values, actions = DRLAgent.DRL_prediction(
            model=reloaded_model,
            environment=validation_environment,
            deterministic=True,
        )
        account_values.to_csv(result_dir / "validation_account_value.csv", index=False)
        actions.to_csv(result_dir / "validation_actions.csv", index=False)

        manifest["status"] = "completed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["duration_seconds"] = round(time.perf_counter() - started, 3)
        manifest["artifacts"].update(
            {
                "validation_account_value": str(result_dir / "validation_account_value.csv"),
                "validation_actions": str(result_dir / "validation_actions.csv"),
                "training_progress": str(tensorboard_dir / "progress.csv"),
                "archive_metadata": "metadata.txt",
            }
        )
        archive_metadata = {
            "run_id": run_id,
            "status": manifest["status"],
            "started_at_utc": manifest["started_at_utc"],
            "completed_at_utc": manifest["completed_at_utc"],
            "duration_seconds": manifest["duration_seconds"],
            "effective_config": effective_config,
            "runtime": manifest["runtime"],
            "artifacts": manifest["artifacts"],
        }
        _write_archive_metadata(model_path, archive_metadata)
        MODELS[algorithm].load(str(model_path), device=effective_device)
        _write_json(manifest_path, manifest)
        return manifest
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["duration_seconds"] = round(time.perf_counter() - started, 3)
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        _write_json(manifest_path, manifest)
        raise
