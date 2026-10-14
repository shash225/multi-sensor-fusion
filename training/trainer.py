"""Stage-aware trainer for the 7-stage curriculum (and the single-stage ablation runs).

A *stage* (``configs/training.yaml: stages``) defines: config overrides (e.g. which
sensors), checkpoints to initialise from, frozen modules, active losses, epochs and an
LR scale. Checkpoints go to ``checkpoints/<experiment>/<stage>/{last,best}.pt`` (shared
single-sensor stages to ``checkpoints/_shared/<stage>/``) and embed the full config.

Memory / speed features: AMP (auto-disabled where fp16 is broken), gradient
accumulation, gradient checkpointing, gradient clipping, non-finite-loss guard,
persistent DataLoader workers, cuDNN autotuning.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from datasets.nuscenes_dataset import NuScenesFusionDataset, collate_batch
from evaluation.evaluator import evaluate_model, summary_line
from models.fusion_model import MultiSensorFusionNet, build_model
from utils.checkpoint import load_checkpoint, load_model_weights, save_checkpoint
from utils.config import Config, deep_merge, load_config, parse_override, resolve_path
from utils.device import amp_enabled, get_device, gpu_memory_report, move_to_device
from utils.logging_utils import setup_logging
from utils.seed import seed_everything, worker_init_fn

logger = logging.getLogger(__name__)


def stage_checkpoint_dir(cfg: Config, experiment: str, stage: str) -> Path:
    spec = cfg.stages[stage]
    owner = "_shared" if spec.get("shared") else experiment
    return resolve_path(cfg.runtime.checkpoint_dir) / owner / stage


def build_stage_config(experiment: Optional[str], stage: str, overrides: Optional[list[str]] = None) -> Config:
    cfg = load_config(experiment=experiment, overrides=overrides)
    if stage not in cfg.stages:
        raise KeyError(f"Unknown stage {stage!r}; available: {list(cfg.stages)}")
    merged = cfg.to_dict()
    deep_merge(merged, cfg.stages[stage].get("overrides", {}) or {})
    # CLI overrides win over stage overrides.
    cfg2 = Config(merged)
    for item in overrides or []:
        cfg2.set_path(*parse_override(item))
    cfg2.runtime.stage = stage
    return cfg2


class Trainer:
    def __init__(self, experiment: str, stage: str, overrides: Optional[list[str]] = None, epochs: Optional[int] = None,
                 resume: Optional[str] = None) -> None:
        self.experiment, self.stage = experiment, stage
        self.cfg = build_stage_config(experiment, stage, overrides)
        self.spec = self.cfg.stages[stage]
        self.ckpt_dir = stage_checkpoint_dir(self.cfg, experiment, stage)
        self.log_dir = resolve_path(self.cfg.runtime.output_dir) / "logs" / experiment / stage
        setup_logging(self.log_dir)
        seed_everything(int(self.cfg.runtime.seed), bool(self.cfg.runtime.deterministic))
        self.device = get_device(self.cfg.runtime.device)
        self.use_amp = amp_enabled(self.device, bool(self.cfg.train.amp))
        self.epochs = int(epochs or self.spec.get("epochs") or self.cfg.train.epochs)
        self.active_losses = list(self.spec.losses)
        logger.info("Experiment %s | stage %s: %s", experiment, stage, self.spec.get("description", ""))
        logger.info("Device %s | AMP %s | epochs %d | losses %s", self.device, self.use_amp, self.epochs, self.active_losses)

        self.train_set = NuScenesFusionDataset(self.cfg, self.cfg.dataset.train_split, train=True)
        self.val_set = NuScenesFusionDataset(self.cfg, self.cfg.dataset.val_split, train=False)
        nw = int(self.cfg.dataset.num_workers)
        common = dict(collate_fn=collate_batch, num_workers=nw, pin_memory=bool(self.cfg.dataset.pin_memory) and self.device.type == "cuda",
                      persistent_workers=nw > 0, worker_init_fn=worker_init_fn)
        self.train_loader = DataLoader(self.train_set, batch_size=int(self.cfg.train.batch_size), shuffle=True,
                                       drop_last=True, **common)
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, **common)

        self.model: MultiSensorFusionNet = build_model(self.cfg).to(self.device)
        self.model.set_grad_checkpoint(bool(self.cfg.train.grad_checkpoint))
        self._initialise_from_stages()
        self._freeze()
        params = [p for p in self.model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in params)
        n_all = sum(p.numel() for p in self.model.parameters())
        logger.info("Parameters: %.2fM total, %.2fM trainable", n_all / 1e6, n_train / 1e6)
        lr = float(self.cfg.train.lr) * float(self.spec.get("lr_scale", 1.0))
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=float(self.cfg.train.weight_decay))
        self.accum = max(int(self.cfg.train.grad_accum), 1)
        steps_per_epoch = max(len(self.train_loader) // self.accum, 1)
        total = steps_per_epoch * self.epochs
        warmup = min(int(self.cfg.train.warmup_iters), max(total // 10, 1))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s / max(total, 1), 1.0))))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.start_epoch, self.step = 0, 0
        self.best_score = -float("inf")
        if resume:
            self._resume(resume)

    # ----------------------------------------------------------------- setup
    def _initialise_from_stages(self) -> None:
        for item in self.spec.get("init_from", []) or []:
            src = stage_checkpoint_dir(self.cfg, self.experiment, item["stage"])
            path = src / "best.pt" if (src / "best.pt").exists() else src / "last.pt"
            if not path.exists():
                logger.warning("Init checkpoint for stage %s not found (%s) - training those weights from scratch.",
                               item["stage"], path)
                continue
            state = load_checkpoint(path)["model"]
            report = load_model_weights(self.model, state, item.get("prefixes"))
            logger.info("Initialised %d tensors from %s (%d shape-mismatch skipped)", len(report["loaded"]), path,
                        len(report["skipped_shape"]))

    def _freeze(self) -> None:
        only = self.spec.get("trainable_only")
        if only:
            keep = tuple(p.rstrip(".") + "." for p in only)
            for name, p in self.model.named_parameters():
                p.requires_grad_(name.startswith(keep))
            logger.info("Training only %s", only)
        frozen = self.spec.get("frozen") or []
        if frozen:
            names = self.model.freeze(frozen)
            logger.info("Frozen %d tensors under %s", len(names), frozen)

    def _resume(self, path: str) -> None:
        ckpt = load_checkpoint(Path(path), map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch, self.step = int(ckpt.get("epoch", 0)), int(ckpt.get("step", 0))
        self.best_score = float(ckpt.get("extra", {}).get("best_score", -float("inf")))
        logger.info("Resumed from %s at epoch %d", path, self.start_epoch)

    # ----------------------------------------------------------------- train
    def _score(self, res: dict[str, Any]) -> float:
        if "detection" in self.active_losses:
            return float(res["detection"]["NDS"])
        if "trajectory" in self.active_losses and "planning_learned" in res:
            return -float(res["planning_learned"].get("ade", float("inf")))
        return -float(res.get("val_losses", {}).get("total", float("inf")))

    def _set_train_mode(self) -> None:
        """``model.train()``, but BatchNorm layers whose parameters are frozen stay in eval mode,
        so frozen modules do not drift through their running statistics."""
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                params = list(m.parameters())
                if params and not any(p.requires_grad for p in params):
                    m.eval()

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self._set_train_mode()
        sums: dict[str, float] = {}
        n, skipped = 0, 0
        t_start = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        for it, batch in enumerate(self.train_loader):
            batch = move_to_device(batch, self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                out = self.model(batch)
                losses = self.model.compute_losses(out, batch, self.cfg.loss, self.active_losses)
            loss = losses["total"] / self.accum
            if not torch.isfinite(loss):
                skipped += 1
                logger.warning("Non-finite loss at iteration %d (%s) - batch skipped", it,
                               {k: float(v) for k, v in losses.items()})
                self.optimizer.zero_grad(set_to_none=True)
                continue
            self.scaler.scale(loss).backward()
            if (it + 1) % self.accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                               float(self.cfg.train.grad_clip))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            n += 1
            if (it + 1) % int(self.cfg.train.log_interval) == 0:
                rate = (it + 1) / (time.perf_counter() - t_start)
                mem = gpu_memory_report(self.device).get("peak_allocated_mb", 0)
                logger.info("ep %d it %d/%d | loss %.3f | %s | lr %.2e | %.2f it/s | peak %.0f MB", epoch + 1, it + 1,
                            len(self.train_loader), sums["total"] / n,
                            " ".join(f"{k} {sums[k] / n:.3f}" for k in self.active_losses if k in sums),
                            self.optimizer.param_groups[0]["lr"], rate, mem)
        avg = {k: v / max(n, 1) for k, v in sums.items()}
        avg["skipped_batches"] = skipped
        avg["epoch_time_s"] = time.perf_counter() - t_start
        return avg

    def fit(self) -> dict[str, Any]:
        history_path = self.log_dir / "history.jsonl"
        last_res: dict[str, Any] = {}
        for epoch in range(self.start_epoch, self.epochs):
            train_stats = self.train_epoch(epoch)
            record: dict[str, Any] = {"epoch": epoch + 1, "train": train_stats}
            if (epoch + 1) % int(self.cfg.train.val_interval) == 0 or epoch + 1 == self.epochs:
                res = evaluate_model(self.model, self.val_loader, self.cfg, self.device, self.use_amp)
                logger.info("VAL ep %d | %s", epoch + 1, summary_line(res))
                record["val"] = {"detection": {k: v for k, v in res["detection"].items() if k != "per_class"},
                                 **{k: v for k, v in res.items() if k != "detection"}}
                score = self._score(res)
                last_res = res
                if score > self.best_score:
                    self.best_score = score
                    self._save("best.pt", epoch + 1, res)
                    logger.info("New best (%.4f) -> %s", score, self.ckpt_dir / "best.pt")
            self._save("last.pt", epoch + 1, last_res)
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=float) + "\n")
        return last_res

    def _save(self, name: str, epoch: int, res: dict[str, Any]) -> None:
        metrics = {}
        if res:
            metrics = {"detection": {k: v for k, v in res["detection"].items() if k != "per_class"},
                       **{k: v for k, v in res.items() if k in ("planning_learned", "feature_selection", "latency_ms")}}
        save_checkpoint(self.ckpt_dir / name, self.model, self.cfg.to_dict(), optimizer=self.optimizer,
                        scheduler=self.scheduler, scaler=self.scaler, epoch=epoch, step=self.step, metrics=metrics,
                        extra={"best_score": self.best_score, "experiment": self.experiment, "stage": self.stage})
