#!/usr/bin/env python3
"""
Simple HTTP server that proxies Miles evaluation requests to `tb run` (1.0)
or `harbor run` (2.0), depending on the request payload.

Usage:
    python examples/eval/terminal_bench/tb_server.py \
        --host 0.0.0.0 --port 9050

Miles (or Miles-compatible runners) should POST the payload described in
`EvalRequestPayload` to http://<host>:<port>/evaluate. The server blocks until
the run finishes, then returns aggregated metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pty
import shlex
import statistics
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

logger = logging.getLogger("terminal_bench_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Request payload helpers
# ---------------------------------------------------------------------------


@dataclass
class EvalRequestPayload:
    model_name: str = ""
    agent_name: str | None = None
    api_base: str = ""
    runner: str | None = None
    dataset_name: str | None = None
    dataset_version: str | None = None
    n_concurrent: int | None = None
    metric_prefix: str | None = None
    output_path: str | None = None
    runner_kwargs: dict[str, Any] | None = None


@dataclass
class JobRecord:
    job_id: str
    status: str
    run_id: str
    command: str
    output_dir: str
    raw_metrics: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
            "run_id": self.run_id,
            "command": self.command,
            "output_dir": self.output_dir,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if self.raw_metrics is not None:
            payload["raw_metrics"] = self.raw_metrics
        if self.error:
            payload["error"] = self.error
        return payload


# ---------------------------------------------------------------------------
# Configuration + command helpers
# ---------------------------------------------------------------------------


class Runner(str, Enum):
    TB = "tb"
    HARBOR = "harbor"


def _normalize_model_name(model_name: str) -> str:
    name = (model_name or "").strip()
    if not name:
        return ""
    if "/" in name:
        return name
    return f"openai/{name}"


def _snake_to_kebab(value: str) -> str:
    return value.replace("_", "-")


def _json_value(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def _append_runner_kwargs(cmd: list[str], runner_kwargs: Mapping[str, Any]) -> None:
    for key, value in runner_kwargs.items():
        flag = f"--{_snake_to_kebab(str(key))}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, (dict, list)):
                    cmd.extend([flag, _json_value(item)])
                else:
                    cmd.extend([flag, str(item)])
            continue
        if isinstance(value, dict):
            if key == "agent_kwarg":
                for agent_key, agent_value in value.items():
                    if isinstance(agent_value, (dict, list)):
                        agent_value_str = _json_value(agent_value)
                    else:
                        agent_value_str = str(agent_value)
                    cmd.extend([flag, f"{agent_key}={agent_value_str}"])
            else:
                cmd.extend([flag, _json_value(value)])
            continue
        cmd.extend([flag, str(value)])


@dataclass
class ServerConfig:
    output_root: Path

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ServerConfig:
        return cls(output_root=Path(args.output_root).expanduser().resolve())


class TerminalBenchEvaluator:
    def __init__(self, config: ServerConfig):
        self._config = config
        self._lock = threading.Lock()
        self._jobs_lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}

    def evaluate(self, payload: EvalRequestPayload) -> dict[str, Any]:
        if not payload.model_name:
            raise ValueError("Missing `model_name` in request payload.")
        if not payload.api_base:
            raise ValueError("Missing `api_base` in request payload.")

        job_id = uuid.uuid4().hex
        run_id = f"{int(time.time())}-{job_id[:8]}"
        runner = Runner(payload.runner)
        run_dir, job_name = self._prepare_run_dir(payload, runner, run_id)

        command = self._build_command(payload, run_id, runner, job_name)
        command_str = self._format_command(command)

        record = JobRecord(
            job_id=job_id,
            status="queued",
            run_id=run_id,
            command=command_str,
            output_dir=str(run_dir),
        )
        with self._jobs_lock:
            self._jobs[job_id] = record

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, payload, run_dir, command, runner),
            daemon=True,
        )
        thread.start()

        return {
            "job_id": job_id,
            "status": "queued",
            "status_url": f"/status/{job_id}",
            "run_id": run_id,
            "command": command_str,
            "output_dir": str(run_dir),
        }

    def _run_job(
        self,
        job_id: str,
        payload: EvalRequestPayload,
        run_dir: Path,
        command: list[str],
        runner: str,
    ) -> None:
        self._update_job(job_id, status="running", started_at=time.time())

        env = self._build_env()
        logger.info("Starting Terminal Bench run: %s", " ".join(shlex.quote(part) for part in command))
        try:
            with self._lock:
                self._run_command(
                    command,
                    env=env,
                )
            metrics = self._collect_metrics(run_dir, runner, payload)
            if payload.metric_prefix:
                metrics = {payload.metric_prefix: metrics}
            self._update_job(
                job_id,
                status="completed",
                raw_metrics=metrics,
                finished_at=time.time(),
            )
        except Exception as exc:  # noqa: BLE001
            self._update_job(
                job_id,
                status="failed",
                error=str(exc),
                finished_at=time.time(),
            )

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        with self._jobs_lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            return record.to_dict()

    def _build_command(
        self,
        payload: EvalRequestPayload,
        run_id: str,
        runner: Runner,
        job_name: str | None,
    ) -> list[str]:
        if runner is Runner.HARBOR:
            cmd = self._build_harbor_command(payload, job_name)
        else:
            cmd = self._build_tb_command(payload, run_id)

        model_name = _normalize_model_name(payload.model_name)
        if model_name:
            cmd.extend(["--model", model_name])

        agent_name = (payload.agent_name or "terminus-2").strip()
        if agent_name:
            cmd.extend(["--agent", agent_name])

        if payload.api_base:
            cmd.extend(["--agent-kwarg", f"api_base={payload.api_base}"])

        n_concurrent = payload.n_concurrent if payload.n_concurrent is not None else 1
        cmd.extend(["--n-concurrent", str(n_concurrent)])

        return cmd

    def _build_harbor_command(self, payload: EvalRequestPayload, job_name: str | None) -> list[str]:
        dataset_name = (payload.dataset_name or "terminal-bench").strip() or "terminal-bench"
        dataset_version = (payload.dataset_version or "2.0").strip() or "2.0"
        cmd = [
            "harbor",
            "run",
            "-d",
            f"{dataset_name}@{dataset_version}",
        ]
        jobs_dir = payload.output_path
        if jobs_dir:
            cmd.extend(["--jobs-dir", jobs_dir])
        if job_name:
            cmd.extend(["--job-name", job_name])

        if payload.runner_kwargs:
            _append_runner_kwargs(cmd, payload.runner_kwargs)

        return cmd

    def _build_tb_command(self, payload: EvalRequestPayload, run_id: str) -> list[str]:
        dataset_name = (payload.dataset_name or "terminal-bench-core").strip() or "terminal-bench-core"
        dataset_version = (payload.dataset_version or "0.1.1").strip() or "0.1.1"
        cmd = [
            "tb",
            "run",
            "-d",
            f"{dataset_name}=={dataset_version}",
        ]
        output_root = str(Path(payload.output_path or self._config.output_root).expanduser())
        Path(output_root).mkdir(parents=True, exist_ok=True)
        cmd.extend(["--output-path", output_root, "--run-id", run_id])
        if payload.runner_kwargs:
            _append_runner_kwargs(cmd, payload.runner_kwargs)

        return cmd

    def _prepare_run_dir(
        self,
        payload: EvalRequestPayload,
        runner: Runner,
        run_id: str,
    ) -> tuple[Path, str | None]:
        if runner is Runner.HARBOR:
            jobs_dir = Path(payload.output_path or "jobs").expanduser()
            jobs_dir.mkdir(parents=True, exist_ok=True)
            return jobs_dir / run_id, run_id

        tb_root = Path(payload.output_path or self._config.output_root).expanduser()
        tb_root.mkdir(parents=True, exist_ok=True)
        return tb_root / run_id, None

    def _update_job(self, job_id: str, **updates: Any) -> None:
        with self._jobs_lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            for key, value in updates.items():
                setattr(record, key, value)

    @staticmethod
    def _format_command(command: list[str]) -> str:
        return " ".join(shlex.quote(part) for part in command)

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        # Inject env var to simulate "OPENAI_API_KEY=EMPTY"
        env["OPENAI_API_KEY"] = "EMPTY"
        return env

    @staticmethod
    def _run_command(
        cmd: list[str],
        *,
        env: dict[str, str],
    ):
        env = env.copy()
        env.setdefault("TERM", "xterm-256color")
        env.setdefault("RICH_FORCE_TERMINAL", "1")
        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(
            cmd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
        )
        os.close(slave_fd)
        try:
            while True:
                try:
                    data = os.read(master_fd, 1024)
                except OSError:
                    break
                if not data:
                    break
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
        finally:
            os.close(master_fd)
        retcode = process.wait()
        if retcode != 0:
            raise RuntimeError(f"Command failed with exit code {retcode}.")

    @staticmethod
    def _collect_metrics(run_dir: Path, runner: Runner, payload: EvalRequestPayload) -> dict[str, Any]:
        if runner is Runner.HARBOR:
            metrics_path = run_dir / "result.json"
            if not metrics_path.exists():
                fallback = TerminalBenchEvaluator._find_latest_result(
                    Path(payload.output_path or "jobs").expanduser()
                )
                if fallback is not None:
                    metrics_path = fallback
            if not metrics_path.exists():
                logger.warning("Results file missing at %s", metrics_path)
                return {}
            metrics = TerminalBenchEvaluator._extract_harbor_metrics(
                metrics_path,
                model_name=_normalize_model_name(payload.model_name),
                dataset_name=(payload.dataset_name or "terminal-bench"),
                agent_name=(payload.agent_name or "terminus-2"),
            )
        else:
            metrics_path = run_dir / "results.json"
            if not metrics_path.exists():
                logger.warning("Results file missing at %s", metrics_path)
                return {}
            metrics = TerminalBenchEvaluator._extract_metrics(metrics_path)
        if not metrics:
            logger.warning("No accuracy/n_resolved metrics found in %s", metrics_path)
        return metrics

    @staticmethod
    def _extract_metrics(metrics_path: Path) -> dict[str, Any]:
        try:
            with open(metrics_path, encoding="utf-8") as fp:
                metrics_data = json.load(fp)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse %s: %s", metrics_path, exc)
            return {}

        metrics: dict[str, Any] = {}

        # core metrics
        accuracy = metrics_data.get("accuracy")
        if isinstance(accuracy, (int, float)):
            metrics["accuracy"] = float(accuracy)

        n_resolved = metrics_data.get("n_resolved")
        if isinstance(n_resolved, (int, float)):
            metrics["n_resolved"] = int(n_resolved)

        n_unresolved = metrics_data.get("n_unresolved")
        if isinstance(n_unresolved, (int, float)):
            metrics["n_unresolved"] = int(n_unresolved)

        # pass@k flatten
        pass_at_k = metrics_data.get("pass_at_k")
        if isinstance(pass_at_k, dict):
            for k, v in pass_at_k.items():
                if isinstance(v, (int, float)):
                    metrics[f"pass_at_k/{k}"] = float(v)

        # token stats from per-task results
        metrics.update(TerminalBenchEvaluator._extract_token_stats(metrics_data.get("results")))

        return metrics

    @staticmethod
    def _find_latest_result(jobs_dir: Path) -> Path | None:
        if not jobs_dir.exists():
            return None
        candidates = list(jobs_dir.glob("**/result.json"))
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    @staticmethod
    def _extract_harbor_metrics(
        metrics_path: Path,
        *,
        model_name: str,
        dataset_name: str,
        agent_name: str,
    ) -> dict[str, Any]:
        try:
            with open(metrics_path, encoding="utf-8") as fp:
                metrics_data = json.load(fp)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse %s: %s", metrics_path, exc)
            return {}

        metrics: dict[str, Any] = {}
        stats = metrics_data.get("stats")
        if isinstance(stats, dict):
            evals = stats.get("evals")
        else:
            evals = None

        accuracy = None
        if isinstance(evals, dict):
            candidates = [
                f"{agent_name}__{model_name}__{dataset_name}",
                f"{agent_name}__{model_name}__terminal-bench",
            ]
            for key in candidates:
                entry = evals.get(key)
                accuracy = TerminalBenchEvaluator._extract_harbor_accuracy(entry)
                if accuracy is not None:
                    break
            if accuracy is None:
                for entry in evals.values():
                    accuracy = TerminalBenchEvaluator._extract_harbor_accuracy(entry)
                    if accuracy is not None:
                        break

        if isinstance(accuracy, (int, float)):
            metrics["accuracy"] = float(accuracy)
            metrics["pass_at_k/1"] = float(accuracy)

        reward_stats = metrics_data.get("reward_stats")
        if isinstance(reward_stats, dict):
            reward_counts = reward_stats.get("reward")
        else:
            reward_counts = None

        if isinstance(reward_counts, dict):
            resolved = TerminalBenchEvaluator._extract_reward_count(reward_counts, 1.0)
            unresolved = TerminalBenchEvaluator._extract_reward_count(reward_counts, 0.0)
            if resolved is not None:
                metrics["n_resolved"] = resolved
            if unresolved is not None:
                metrics["n_unresolved"] = unresolved

        metrics.update(TerminalBenchEvaluator._extract_token_stats(metrics_data.get("results")))

        return metrics

    @staticmethod
    def _extract_token_stats(results: Any) -> dict[str, float]:
        if not isinstance(results, list):
            return {}

        input_tokens = [
            r.get("total_input_tokens")
            for r in results
            if isinstance(r, dict) and isinstance(r.get("total_input_tokens"), (int, float))
        ]
        output_tokens = [
            r.get("total_output_tokens")
            for r in results
            if isinstance(r, dict) and isinstance(r.get("total_output_tokens"), (int, float))
        ]

        metrics: dict[str, float] = {}
        if input_tokens:
            metrics["total_input_tokens_mean"] = float(statistics.mean(input_tokens))
            metrics["total_input_tokens_median"] = float(statistics.median(input_tokens))
        if output_tokens:
            metrics["total_output_tokens_mean"] = float(statistics.mean(output_tokens))
            metrics["total_output_tokens_median"] = float(statistics.median(output_tokens))
        return metrics

    @staticmethod
    def _extract_harbor_accuracy(entry: Any) -> float | None:
        if not isinstance(entry, dict):
            return None
        metrics_block = entry.get("metrics")
        if isinstance(metrics_block, list) and metrics_block:
            first_metric = metrics_block[0]
            if isinstance(first_metric, dict):
                mean = first_metric.get("mean")
                if isinstance(mean, (int, float)):
                    return float(mean)
        mean = entry.get("mean")
        if isinstance(mean, (int, float)):
            return float(mean)
        return None

    @staticmethod
    def _extract_reward_count(reward_counts: dict[Any, Any], reward_value: float) -> int | None:
        for key, value in reward_counts.items():
            try:
                key_value = float(key)
            except (TypeError, ValueError):
                continue
            if key_value == reward_value and isinstance(value, (int, float)):
                return int(value)
        return None


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


def build_app(evaluator: TerminalBenchEvaluator) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health_check():
        return jsonify({"status": "ok"})

    @app.post("/evaluate")
    def evaluate_endpoint():
        try:
            raw_payload = request.get_json(force=True, silent=False)
            cfg = OmegaConf.merge(
                OmegaConf.structured(EvalRequestPayload),
                OmegaConf.create(raw_payload or {}),
            )
            payload = OmegaConf.to_object(cfg)
            result = evaluator.evaluate(payload)
            return jsonify(result)
        except OmegaConfBaseException as exc:
            logger.exception("Invalid request payload")
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            logger.exception("Evaluation failed")
            return jsonify({"error": str(exc)}), 500

    @app.get("/status/<job_id>")
    def status_endpoint(job_id: str):
        status = evaluator.get_job_status(job_id)
        if status is None:
            return jsonify({"error": "job not found"}), 404
        return jsonify(status)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Terminal Bench evaluation HTTP server.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9050)
    parser.add_argument(
        "--output-root",
        type=str,
        default="./terminal-bench-output",
        help="Directory to store `tb run` outputs (Terminal Bench 1.0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ServerConfig.from_args(args)
    evaluator = TerminalBenchEvaluator(config)
    app = build_app(evaluator)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logger.info(
        "Starting Terminal Bench evaluation server on %s:%s (output root=%s)",
        args.host,
        args.port,
        config.output_root,
    )
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
