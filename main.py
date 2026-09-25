"""PAST-NLinear-EOF with optimized acoustic and static-stability physics.

Final model: rank-24 causal-low-rank NLinear-EOF, training-fitted persistence
prior, optimized acoustic observation consistency, train-supported N2-like
static-stability regularization, sparse upper-ocean assimilation, and
deduplicated BLTS. The final evaluation is repeated independently for three
random seeds (default: 42, 52 and 62). Primary numerical results are reported
as mean ± standard deviation across the three seeds.

The optimized physics module uses only TRAIN-derived calibration, robust
scales, reliability gates and N2-proxy envelopes. CTRP is not part of the
current physics module.
"""

import argparse
import copy
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset


SEQ, HORIZON, STRIDE = 168, 72, 6
NT, NS = 6, 4
DEFAULT_SEEDS = (42, 52, 62)
FINAL_MODEL_NAME = "PAST-NLinear-EOF"
FINAL_MODEL_FULL_NAME = (
    "Physics-Aware Sparse-Assimilation and Temporal-Refinement NLinear-EOF"
)
BASELINE_MODEL_NAME = "NLinear-EOF"
PHYSICS_CONSTRAINT_NAME = "Optimized Acoustic + Static-Stability Physics (OASP)"
DEFAULT_OPT_ACOUSTIC_WEIGHT = 0.003
DEFAULT_STABILITY_WEIGHT = 0.001
GRAVITY = 9.81


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("/opt/data/private/hwh/DeepLearning_Final_MultiLayer.csv"))
    p.add_argument("--anchor", type=Path, default=Path("/opt/data/private/hwh/hourly_2024_profile_anchor.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("/opt/data/private/hwh/final_locked_model"))
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help=(
            "Independent training seeds. The publication default is "
            "42 52 62; primary results are summarized as mean ± std."
        ),
    )
    p.add_argument(
        "--figure-seed",
        type=int,
        default=None,
        help=(
            "Optional seed used for Figures 8/9. If omitted, a representative "
            "seed is selected from --seeds using validation performance only."
        ),
    )
    p.add_argument("--figure-dpi", type=int, default=600)
    p.add_argument(
        "--plot-only", action="store_true",
        help="Regenerate figures/tables from final_test_predictions.npz.",
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument(
        "--opt-acoustic-weight",
        type=float,
        default=DEFAULT_OPT_ACOUSTIC_WEIGHT,
        help=(
            "Weight of the optimized acoustic consistency loss: level + "
            "window-centered + six-hour-change components, all calibrated "
            "from training data only."
        ),
    )
    p.add_argument(
        "--stability-weight",
        type=float,
        default=DEFAULT_STABILITY_WEIGHT,
        help=(
            "Weight of the train-supported two-sided N2-like static-stability "
            "envelope regularizer."
        ),
    )
    p.add_argument("--physical-epochs", type=int, default=30)
    p.add_argument("--physical-patience", type=int, default=8)
    p.add_argument("--vsc-weight", type=float, default=0.03)
    p.add_argument("--receiver-depth", type=float, default=1515.0)
    p.add_argument("--station-spacing", type=float, default=3186.0)
    p.add_argument("--salt-physics-gradient", type=float, default=0.25)
    p.add_argument("--calibration-fraction", type=float, default=0.50)
    p.add_argument("--persistence-ridge", type=float, default=0.05)
    p.add_argument("--physics-max-grad-ratio", type=float, default=0.20)
    p.add_argument("--physics-warmup-epochs", type=int, default=5)
    p.add_argument(
        "--model-variant", choices=("legacy", "causal-lowrank"),
        default="causal-lowrank",
        help="Legacy NLinear or the regularized causal multiscale model.",
    )
    p.add_argument(
        "--model-rank", type=int, default=24,
        help="Travel-time bottleneck rank for causal-lowrank.",
    )
    p.add_argument(
        "--persistence-prior", dest="persistence_prior",
        action="store_true",
        help="Add a training-fitted current-anomaly persistence prior.",
    )
    p.add_argument(
        "--no-persistence-prior", dest="persistence_prior",
        action="store_false",
        help="Disable the current-anomaly persistence prior for ablation.",
    )
    p.set_defaults(persistence_prior=True)
    p.add_argument(
        "--persistence-ridge-ratio", type=float, default=0.05,
        help="Training-only ridge used to fit the persistence prior.",
    )
    p.add_argument(
        "--physics-suite", choices=("compact", "full"), default="compact",
        help="Compact keeps only supervised, legacy VSC and multiscale VSC.",
    )
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_depth(name):
    match = re.search(r"_([0-9.]+)m(?:_|$)", name)
    if not match:
        raise ValueError(f"Cannot parse depth from {name}")
    return float(match.group(1))


def prefixed(columns, prefix):
    result = [c for c in columns if c.startswith(prefix)]
    if not result:
        raise ValueError(f"No columns start with {prefix}")
    return sorted(result, key=get_depth)


def interpolate_profiles(frame, columns, target_depths):
    source_depths = np.array([get_depth(c) for c in columns])
    order = np.argsort(source_depths)
    source_depths = source_depths[order]
    values = frame[columns].to_numpy(float)[:, order]
    if source_depths[0] - target_depths[0] > 25 or target_depths[-1] - source_depths[-1] > 25:
        raise ValueError("Anchor and target depth ranges do not align")
    return np.vstack([np.interp(target_depths, source_depths, row, left=row[0], right=row[-1]) for row in values])


def load_data(data_path, anchor_path):
    header = pd.read_csv(data_path, nrows=0).columns.tolist()
    temp_cols = prefixed(header, "True_Temp_L")
    salt_cols = [c.replace("True_Temp", "True_Salt", 1) for c in temp_cols]
    depths = np.array([get_depth(c) for c in temp_cols])
    required = ["Datetime", "Travel_Time_sec"] + temp_cols + salt_cols

    frame = pd.read_csv(data_path)
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError(f"Missing columns: {missing[:5]}")
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], errors="coerce")
    frame[required[1:]] = frame[required[1:]].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=required).sort_values("Datetime").drop_duplicates("Datetime", keep="last").reset_index(drop=True)
    frame["Month"], frame["Day"], frame["Hour"] = frame.Datetime.dt.month, frame.Datetime.dt.day, frame.Datetime.dt.hour

    anchor = pd.read_csv(anchor_path)
    keys = ["Month", "Day", "Hour"]
    if anchor.duplicated(keys).any():
        raise ValueError("Anchor contains duplicate Month/Day/Hour keys")
    at, ass = prefixed(anchor.columns.tolist(), "Temp_"), prefixed(anchor.columns.tolist(), "Salt_")
    anchor[at + ass] = anchor[at + ass].apply(pd.to_numeric, errors="coerce")
    profiles = np.concatenate([interpolate_profiles(anchor, at, depths), interpolate_profiles(anchor, ass, depths)], axis=1)
    names = [f"A{i}" for i in range(profiles.shape[1])]
    lookup = anchor[keys].copy()
    lookup[names] = profiles
    frame = frame.merge(lookup, on=keys, how="left", validate="many_to_one")
    if frame[names].isna().any().any():
        raise ValueError("Some timestamps have no exact hourly 2024 anchor")
    # `profiles` comes exclusively from the independent 2024 anchor file.
    # It is returned separately for provenance/diagnostics; the current OASP
    # module does not fit a climatological T-S prior from it.
    return frame, temp_cols + salt_cols, names, depths, profiles


class Scaler:
    def fit(self, value):
        self.mean = np.mean(value, axis=0)
        self.std = np.std(value, axis=0)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        return self

    def transform(self, value):
        return (value - self.mean) / self.std


class EOF:
    def __init__(self, modes):
        self.modes = modes

    def fit(self, residual):
        self.basis = np.linalg.svd(residual, full_matrices=False)[2][: self.modes]
        coefficient = residual @ self.basis.T
        self.scale = np.std(coefficient, axis=0)
        self.scale = np.where(self.scale < 1e-8, 1.0, self.scale)
        return self

    def transform(self, residual):
        return residual @ self.basis.T

    def inverse(self, coefficient):
        return coefficient @ self.basis


def fit_current_eof_projection(
    eof_t, eof_s, upper, scale_t, scale_s, ridge=0.05,
):
    """Project the six current sparse anomalies into standardized EOF space.

    The projection is deterministic and fitted only from training-derived EOF
    bases/scalers.  It therefore provides a causal current-state prior without
    using any future profile value.
    """
    weight = np.zeros((6, NT + NS), dtype=np.float64)
    bias = np.zeros(NT + NS, dtype=np.float64)
    specifications = (
        (eof_t, scale_t, 0, NT, 0),
        (eof_s, scale_s, NT, NT + NS, 3),
    )
    for eof, scaler, out_start, out_end, input_start in specifications:
        sparse_basis = eof.basis[:, upper]
        gram = sparse_basis @ sparse_basis.T
        inverse = np.linalg.inv(
            gram + float(ridge) * np.eye(eof.modes, dtype=np.float64),
        )
        projection = sparse_basis.T @ inverse
        projection /= eof.scale[None]
        weight[input_start:input_start + 3, out_start:out_end] = (
            scaler.std[:, None] * projection
        )
        bias[out_start:out_end] = scaler.mean @ projection
    return weight.astype(np.float32), bias.astype(np.float32)


def fit_persistence_prior(
    windows, current_weight, current_bias, ridge_ratio=0.05,
):
    """Fit a conservative horizon/mode persistence coefficient on train only."""
    context = windows["train"][1][:, :6].astype(np.float64)
    future_z = windows["train"][2].astype(np.float64)
    current_z = context @ current_weight.astype(np.float64)
    current_z += current_bias.astype(np.float64)
    denominator = np.sum(current_z * current_z, axis=0)
    numerator = np.sum(future_z * current_z[:, None, :], axis=0)
    ridge = float(ridge_ratio) * np.maximum(denominator, 1e-8)
    alpha = numerator / (denominator[None] + ridge[None])
    # A persistence prior should decay or vanish, never reverse/amplify the
    # current anomaly.  The trainable network learns the remaining correction.
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha = np.minimum.accumulate(alpha, axis=0)
    return alpha.astype(np.float32)


def make_windows(history, context, coefficient, target, anchor, times, train_end,
                 val_end, future_tt, return_meta=False):
    buckets = {name: [[], [], [], [], [], []] for name in ("train", "val", "test")}
    meta = {
        name: {"origin_ns": [], "valid_ns": [], "start_idx": []}
        for name in ("train", "val", "test")
    }
    times = pd.DatetimeIndex(times)
    times_ns = times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    # Prefix-count irregular gaps once instead of rescanning all 239 gaps in
    # every overlapping window.  A window is hourly iff its gap count is zero.
    hourly = np.diff(times_ns) == np.int64(3_600_000_000_000)
    irregular_prefix = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(~hourly, dtype=np.int64)],
    )
    skipped = 0
    for start in range(0, len(times) - SEQ - HORIZON + 1, STRIDE):
        origin, end = start + SEQ, start + SEQ + HORIZON
        if irregular_prefix[end - 1] != irregular_prefix[start]:
            skipped += 1
            continue
        split = "train" if end <= train_end else "val" if origin >= train_end and end <= val_end else "test" if origin >= val_end else None
        if split is None:
            continue
        values = history[start:origin], context[origin - 1], coefficient[origin:end], target[origin:end], anchor[origin:end], future_tt[origin:end]
        for destination, value in zip(buckets[split], values):
            destination.append(value)
        meta[split]["origin_ns"].append(times_ns[origin - 1])
        meta[split]["valid_ns"].append(times_ns[origin:end].copy())
        meta[split]["start_idx"].append(start)
    if skipped:
        print(f"Skipped non-hourly windows: {skipped}")
    result = {name: tuple(np.asarray(x, np.float32) for x in values) for name, values in buckets.items()}
    if any(len(result[name][0]) == 0 for name in result):
        raise ValueError("At least one time split has no valid window")
    packed_meta = {
        name: {
            "origin_ns": np.asarray(values["origin_ns"], dtype=np.int64),
            "valid_ns": np.asarray(values["valid_ns"], dtype=np.int64),
            "start_idx": np.asarray(values["start_idx"], dtype=np.int64),
        }
        for name, values in meta.items()
    }
    return (result, packed_meta) if return_meta else result


def make_loaders(windows, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    return {
        name: DataLoader(
            TensorDataset(*(torch.from_numpy(x) for x in arrays)),
            batch_size=batch_size,
            shuffle=name == "train",
            generator=generator if name == "train" else None,
            pin_memory=pin_memory,
        )
        for name, arrays in windows.items()
    }


def density_np(temp, salt):
    rw = 999.842594 + 6.793952e-2 * temp - 9.095290e-3 * temp**2 + 1.001685e-4 * temp**3 - 1.120083e-6 * temp**4 + 6.536332e-9 * temp**5
    a = 0.824493 - 4.0899e-3 * temp + 7.6438e-5 * temp**2 - 8.2467e-7 * temp**3 + 5.3875e-9 * temp**4
    b = -5.72466e-3 + 1.0227e-4 * temp - 1.6546e-6 * temp**2
    return rw + a * salt + b * np.maximum(salt, 0.0) ** 1.5 + 4.8314e-4 * salt**2

def density_torch(temp, salt):
    rw = 999.842594 + 6.793952e-2 * temp - 9.095290e-3 * temp.square() + 1.001685e-4 * temp.pow(3) - 1.120083e-6 * temp.pow(4) + 6.536332e-9 * temp.pow(5)
    a = 0.824493 - 4.0899e-3 * temp + 7.6438e-5 * temp.square() - 8.2467e-7 * temp.pow(3) + 5.3875e-9 * temp.pow(4)
    b = -5.72466e-3 + 1.0227e-4 * temp - 1.6546e-6 * temp.square()
    return rw + a * salt + b * torch.clamp(salt, min=0).pow(1.5) + 4.8314e-4 * salt.square()


def robust_depth_scale(value):
    """Training-only MAD scale for each vertical derivative coordinate."""
    value = np.asarray(value, dtype=np.float64)
    center = np.nanmedian(value, axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(value - center[None]), axis=0)
    positive = scale[np.isfinite(scale) & (scale > 0)]
    reference = float(np.quantile(positive, 0.10)) if len(positive) else 1e-8
    floor = max(reference * 0.10, 1e-10)
    return np.where(np.isfinite(scale) & (scale >= floor), scale, floor)


def vertical_second_np(value, depths, step):
    """Second vertical derivative on a nonuniform grid at one scale."""
    z = np.asarray(depths, dtype=np.float64)
    left = (value[..., step:-step] - value[..., :-2 * step]) / (
        z[step:-step] - z[:-2 * step]
    )
    right = (value[..., 2 * step:] - value[..., step:-step]) / (
        z[2 * step:] - z[step:-step]
    )
    return 2.0 * (right - left) / (z[2 * step:] - z[:-2 * step])


def vertical_second_torch(value, depths, step):
    left = (value[..., step:-step] - value[..., :-2 * step]) / (
        depths[step:-step] - depths[:-2 * step]
    )
    right = (value[..., 2 * step:] - value[..., step:-step]) / (
        depths[2 * step:] - depths[step:-step]
    )
    return 2.0 * (right - left) / (
        depths[2 * step:] - depths[:-2 * step]
    )


def sound_speed_np(temp, salt, depths):
    d = np.asarray(depths, dtype=np.float64).reshape((1,) * (temp.ndim - 1) + (-1,))
    return (
        1448.96 + 4.591 * temp - 5.304e-2 * temp**2 + 2.374e-4 * temp**3
        + 1.340 * (salt - 35.0) + 1.630e-2 * d + 1.675e-7 * d**2
        - 1.025e-2 * temp * (salt - 35.0) - 7.139e-13 * temp * d**3
    )


def sound_speed_torch(temp, salt, depths):
    d = depths.view(1, 1, -1)
    return (
        1448.96 + 4.591 * temp - 5.304e-2 * temp.square() + 2.374e-4 * temp.pow(3)
        + 1.340 * (salt - 35.0) + 1.630e-2 * d + 1.675e-7 * d.square()
        - 1.025e-2 * temp * (salt - 35.0) - 7.139e-13 * temp * d.pow(3)
    )


def physical_travel_time_np(temp, salt, depths, sec_theta):
    c = sound_speed_np(temp, salt, depths)
    dz = np.diff(np.asarray(depths, dtype=np.float64))
    return sec_theta * np.sum(dz * (1.0 / c[..., :-1] + 1.0 / c[..., 1:]), axis=-1)


def physical_travel_time_torch(temp, salt, depths, sec_theta):
    c = sound_speed_torch(temp, salt, depths)
    dz = torch.diff(depths)
    return sec_theta * torch.sum(dz * (1.0 / c[..., :-1] + 1.0 / c[..., 1:]), dim=-1)



def prepare(args, seed):
    (
        frame,
        target_cols,
        anchor_cols,
        depths,
        _anchor_reference_profiles,
    ) = load_data(args.data, args.anchor)
    layers = len(depths)
    train_end, val_end = int(0.70 * len(frame)), int(0.85 * len(frame))
    target = frame[target_cols].to_numpy(float)
    anchor = frame[anchor_cols].to_numpy(float)
    residual = target - anchor
    upper = np.array([int(np.argmin(np.abs(depths - z))) for z in (0.5, 20.0, 50.0)])
    if len(np.unique(upper)) != 3:
        raise ValueError("The 0.5/20/50 m inputs do not map to three distinct layers")


    # The accepted 0.307859 baseline is deliberately uncentered.
    mean_residual = np.zeros(residual.shape[1], dtype=float)
    anomaly = residual
    eof_t = EOF(NT).fit(anomaly[:train_end, :layers])
    eof_s = EOF(NS).fit(anomaly[:train_end, layers:])
    coefficient = np.concatenate([eof_t.transform(anomaly[:, :layers]), eof_s.transform(anomaly[:, layers:])], axis=1)
    coefficient_scale = np.concatenate([eof_t.scale, eof_s.scale])
    coefficient_z = coefficient / coefficient_scale

    travel_time = frame[["Travel_Time_sec"]].to_numpy(float)
    tt_scaler = Scaler().fit(travel_time[:train_end])
    tt = tt_scaler.transform(travel_time)
    scale_t = Scaler().fit(residual[:train_end, upper])
    scale_s = Scaler().fit(residual[:train_end, layers + upper])
    upper_t, upper_s = scale_t.transform(residual[:, upper]), scale_s.transform(residual[:, layers + upper])
    history = np.concatenate([tt, upper_t, upper_s], axis=1)
    # Causal origin-time encoding.  These features are known at forecast issue
    # time and contain no future target information.
    timestamp = pd.DatetimeIndex(frame.Datetime)
    day = (
        timestamp.dayofyear.to_numpy(dtype=float) - 1.0
        + timestamp.hour.to_numpy(dtype=float) / 24.0
        + timestamp.minute.to_numpy(dtype=float) / 1440.0
    )
    hour = (
        timestamp.hour.to_numpy(dtype=float)
        + timestamp.minute.to_numpy(dtype=float) / 60.0
    )
    cycle = np.column_stack([
        np.sin(2.0 * np.pi * day / 365.25),
        np.cos(2.0 * np.pi * day / 365.25),
        np.sin(2.0 * np.pi * hour / 24.0),
        np.cos(2.0 * np.pi * hour / 24.0),
    ])
    context = np.concatenate([upper_t, upper_s, cycle], axis=1)
    windows, window_meta = make_windows(
        history, context, coefficient_z, target, anchor, frame.Datetime.to_numpy(),
        train_end, val_end, travel_time[:, 0], return_meta=True,
    )
    current_eof_weight, current_eof_bias = fit_current_eof_projection(
        eof_t, eof_s, upper, scale_t, scale_s,
    )
    persistence_prior = fit_persistence_prior(
        windows, current_eof_weight, current_eof_bias,
        ridge_ratio=args.persistence_ridge_ratio,
    )

    train_t, train_s = target[:train_end, :layers], target[:train_end, layers:]
    temp_depth_scale = np.std(anomaly[:train_end, :layers], axis=0)
    temp_depth_scale = np.where(temp_depth_scale < 1e-4, 1.0, temp_depth_scale)
    salt_depth_scale = np.std(anomaly[:train_end, layers:], axis=0)
    salt_depth_scale = np.where(salt_depth_scale < 1e-4, 1.0, salt_depth_scale)
    if args.receiver_depth <= 0 or args.station_spacing < 0:
        raise ValueError("receiver-depth must be positive and station-spacing non-negative")
    sec_theta = math.sqrt(1.0 + (0.5 * args.station_spacing / args.receiver_depth) ** 2)
    train_tt_physical = physical_travel_time_np(train_t, train_s, depths, sec_theta)
    train_tt_obs = travel_time[:train_end, 0]
    design = np.column_stack([np.ones(len(train_tt_obs)), train_tt_physical])

    # Optimized acoustic calibration. If the raw slope is clipped, re-fit the
    # intercept for the clipped slope to avoid an artificial constant bias.
    opt_raw_tt_offset, opt_raw_tt_slope = np.linalg.lstsq(
        design, train_tt_obs, rcond=None,
    )[0]
    opt_tt_slope = float(np.clip(opt_raw_tt_slope, 0.05, 20.0))
    opt_tt_offset = float(np.mean(
        train_tt_obs - opt_tt_slope * train_tt_physical
    ))
    tt_scale = max(float(np.std(train_tt_obs)), 1e-8)

    train_anchor_t = anchor[:train_end, :layers]
    train_anchor_s = anchor[:train_end, layers:]
    anchor_tt_physical = physical_travel_time_np(
        train_anchor_t, train_anchor_s, depths, sec_theta,
    )
    opt_observed_tt_anomaly = train_tt_obs - (
        opt_tt_offset + opt_tt_slope * anchor_tt_physical
    )
    opt_tt_anomaly_scale = max(
        float(1.4826 * np.median(np.abs(
            opt_observed_tt_anomaly - np.median(opt_observed_tt_anomaly)
        ))),
        1e-8,
    )

    # Reliability is estimated from TRAIN-only observed vs forward travel time
    # without subtracting a shared time-varying anchor.
    if np.std(train_tt_physical) > 1e-10 and np.std(train_tt_obs) > 1e-10:
        opt_acoustic_correlation = float(np.corrcoef(
            train_tt_obs, train_tt_physical
        )[0, 1])
    else:
        opt_acoustic_correlation = 0.0
    if not np.isfinite(opt_acoustic_correlation):
        opt_acoustic_correlation = 0.0
    opt_acoustic_reliability = float(np.clip(
        opt_acoustic_correlation / 0.25, 0.0, 1.0
    ))

    # Dynamic scales/reliability from chronological TRAIN forecast windows.
    train_windows_target = windows["train"][3].astype(np.float64)
    train_windows_obs = windows["train"][5].astype(np.float64)
    train_windows_forward = (
        opt_tt_offset
        + opt_tt_slope * physical_travel_time_np(
            train_windows_target[..., :layers],
            train_windows_target[..., layers:],
            depths, sec_theta,
        )
    )
    observed_centered = (
        train_windows_obs - train_windows_obs.mean(axis=1, keepdims=True)
    )
    forward_centered = (
        train_windows_forward - train_windows_forward.mean(axis=1, keepdims=True)
    )
    observed_change = train_windows_obs[:, 6:] - train_windows_obs[:, :-6]
    forward_change = train_windows_forward[:, 6:] - train_windows_forward[:, :-6]

    def dynamic_scale(value):
        flat = value.ravel()
        robust = 1.4826 * np.median(np.abs(flat - np.median(flat)))
        return max(float(robust), 0.1 * tt_scale, 1e-8)

    def dynamic_reliability(predicted, observed):
        if min(float(np.std(predicted)), float(np.std(observed))) <= 1e-10:
            return 0.0
        correlation = float(np.corrcoef(
            predicted.ravel(), observed.ravel()
        )[0, 1])
        if not np.isfinite(correlation):
            return 0.0
        return float(np.clip(correlation / 0.25, 0.0, 1.0))

    opt_center_scale = dynamic_scale(observed_centered)
    opt_change_scale = dynamic_scale(observed_change)
    opt_center_reliability = dynamic_reliability(
        forward_centered, observed_centered
    )
    opt_change_reliability = dynamic_reliability(
        forward_change, observed_change
    )

    rho = density_np(train_t, train_s)
    dz = np.diff(depths)
    curvature_dz = 0.5 * (dz[:-1] + dz[1:])
    curvature = np.diff(np.diff(rho, axis=1) / dz[None], axis=1) / curvature_dz[None]
    anchor_rho = density_np(train_anchor_t, train_anchor_s)
    anchor_curvature = np.diff(np.diff(anchor_rho, axis=1) / dz[None], axis=1) / curvature_dz[None]
    curvature_scale = np.std(curvature - anchor_curvature, axis=0)
    curvature_scale = np.where(curvature_scale < 1e-10, 1.0, curvature_scale)
    curvature_q95 = np.quantile(np.abs(curvature), 0.95, axis=0)
    density_delta = rho - anchor_rho
    density_delta_gradient = np.diff(density_delta, axis=1) / dz[None]
    sound_delta = sound_speed_np(train_t, train_s, depths) - sound_speed_np(
        train_anchor_t, train_anchor_s, depths,
    )
    sound_train = sound_speed_np(train_t, train_s, depths)
    sound_gradient = np.diff(sound_train, axis=1) / dz[None]
    density_gradient = np.diff(rho, axis=1) / dz[None]

    # N2-like static-stability proxy (depth positive downward). This is an
    # EOS-based differentiable proxy rather than full TEOS-10 N2.
    rho_mid = 0.5 * (rho[:, 1:] + rho[:, :-1])
    n2_proxy = GRAVITY * density_gradient / np.maximum(rho_mid, 1.0)
    # Preserve train-supported weak inversions, penalizing only states outside
    # the empirical 1%-99% TRAIN envelope.
    n2_floor = np.minimum(np.quantile(n2_proxy, 0.01, axis=0), 0.0)
    n2_ceiling = np.maximum(np.quantile(n2_proxy, 0.99, axis=0), 0.0)
    n2_scale = robust_depth_scale(n2_proxy)
    structure_stats = {}
    for step in (1, 2, 4):
        temp_second = vertical_second_np(train_t, depths, step)
        structure_stats[f"temp_second_scale_{step}"] = robust_depth_scale(
            temp_second,
        )
        structure_stats[f"temp_second_q99_{step}"] = np.quantile(
            np.abs(temp_second), 0.99, axis=0,
        )
    train_times_ns = pd.DatetimeIndex(
        frame.Datetime.iloc[:train_end],
    ).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    contiguous = np.diff(train_times_ns) == np.int64(3_600_000_000_000)
    temp_tendency = np.diff(train_t, axis=0)[contiguous]
    salt_tendency = np.diff(train_s, axis=0)[contiguous]
    depth_loss_weight = np.full(layers, 0.75, dtype=np.float32)
    depth_loss_weight[depths <= 500] = 1.00
    depth_loss_weight[(depths > 50) & (depths <= 150)] = 1.50
    depth_loss_weight[upper] = 0.25
    depth_loss_weight /= depth_loss_weight.mean()

    return {
        "loaders": make_loaders(windows, args.batch_size, seed),
        "window_meta": window_meta,
        "train_series_z": coefficient_z[:train_end].astype(np.float32),
        "train_times_ns": train_times_ns,
        "depths": depths,
        "layers": layers,
        "upper": upper,
        "eof_t": eof_t,
        "eof_s": eof_s,
        "coefficient_scale": coefficient_scale,
        "current_eof_weight": current_eof_weight,
        "current_eof_bias": current_eof_bias,
        "persistence_prior": persistence_prior,
        "persistence_prior_h1_mean": float(persistence_prior[0].mean()),
        "persistence_prior_h24_mean": float(persistence_prior[23].mean()),
        "persistence_prior_h72_mean": float(persistence_prior[-1].mean()),
        "mean_residual": mean_residual,
        "temp_scale": max(float(residual[:train_end, :layers].std()), 1e-8),
        "salt_scale": max(float(residual[:train_end, layers:].std()), 1e-8),
        "temp_depth_scale": temp_depth_scale,
        "salt_depth_scale": salt_depth_scale,
        "sec_theta": sec_theta,
        "dz": dz,
        "tt_mean": float(tt_scaler.mean[0]),
        "tt_std": float(tt_scaler.std[0]),
        # Optimized acoustic calibration/statistics, all TRAIN-only.
        "tt_offset": opt_tt_offset,  # compatibility alias for auxiliary code
        "tt_slope": opt_tt_slope,    # compatibility alias for auxiliary code
        "tt_scale": tt_scale,
        "opt_tt_raw_offset": float(opt_raw_tt_offset),
        "opt_tt_raw_slope": float(opt_raw_tt_slope),
        "opt_tt_offset": opt_tt_offset,
        "opt_tt_slope": opt_tt_slope,
        "opt_tt_slope_was_clipped": bool(
            abs(float(opt_raw_tt_slope) - opt_tt_slope) > 1e-12
        ),
        "opt_tt_anomaly_scale": opt_tt_anomaly_scale,
        "opt_acoustic_correlation": opt_acoustic_correlation,
        "opt_acoustic_reliability": opt_acoustic_reliability,
        "opt_center_scale": opt_center_scale,
        "opt_change_scale": opt_change_scale,
        "opt_center_reliability": opt_center_reliability,
        "opt_change_reliability": opt_change_reliability,
        "n2_proxy_floor": n2_floor,
        "n2_proxy_ceiling": n2_ceiling,
        "n2_proxy_scale": n2_scale,
        "upper_t_mean": scale_t.mean,
        "upper_t_std": scale_t.std,
        "upper_s_mean": scale_s.mean,
        "upper_s_std": scale_s.std,
        "curvature_dz": curvature_dz,
        "curvature_q95": curvature_q95,
        "curvature_scale": curvature_scale,
        "density_grad_scale": robust_depth_scale(density_delta_gradient),
        "density_gradient_scale": robust_depth_scale(density_gradient),
        "sound_speed_scale": robust_depth_scale(sound_delta),
        "sound_gradient_scale": robust_depth_scale(sound_gradient),
        "temp_tendency_scale": robust_depth_scale(temp_tendency),
        "salt_tendency_scale": robust_depth_scale(salt_tendency),
        "depth_loss_weight": depth_loss_weight,
        **structure_stats,
    }


class NLinearEOF(nn.Module):
    def __init__(self):
        super().__init__()
        self.tt_t = nn.Linear(SEQ, HORIZON * NT)
        self.level_t = nn.Linear(1, HORIZON * NT, bias=False)
        self.context_t = nn.Linear(3, HORIZON * NT, bias=False)
        self.tt_s = nn.Linear(SEQ, HORIZON * NS)
        self.level_s = nn.Linear(1, HORIZON * NS, bias=False)
        self.context_s = nn.Linear(3, HORIZON * NS, bias=False)
    def forward(self, history, context):
        tt = history[..., 0]
        level = tt[:, -1:]
        temp = self.tt_t(tt - level) + self.level_t(level) + self.context_t(context[:, :3])
        salt = self.tt_s(tt - level) + self.level_s(level) + self.context_s(context[:, 3:6])
        output = torch.cat(
            [temp.view(-1, HORIZON, NT), salt.view(-1, HORIZON, NS)], dim=-1,
        )
        return output


class CausalLowRankNLinearEOF(nn.Module):
    """Regularized NLinear with causal multiscale and calendar features.

    The legacy 168-to-720 dense mapping is replaced by separate low-rank
    temperature/salinity projections.  Small deterministic multiscale features
    retain recent trend and variability information that can be lost by a
    bottleneck.  No feature uses a timestamp or observation after the origin.
    """

    MULTISCALE_DIM = 14

    def __init__(self, rank=24, prepared=None, use_persistence=False):
        super().__init__()
        if not 4 <= int(rank) <= 96:
            raise ValueError("model-rank must be between 4 and 96")
        rank = int(rank)
        self.rank = rank
        self.encoder_t = nn.Linear(SEQ, rank, bias=False)
        self.encoder_s = nn.Linear(SEQ, rank, bias=False)
        self.tt_t = nn.Linear(rank, HORIZON * NT)
        self.tt_s = nn.Linear(rank, HORIZON * NS)
        self.level_t = nn.Linear(1, HORIZON * NT, bias=False)
        self.level_s = nn.Linear(1, HORIZON * NS, bias=False)
        self.context_t = nn.Linear(3, HORIZON * NT, bias=False)
        self.context_s = nn.Linear(3, HORIZON * NS, bias=False)
        self.multiscale_t = nn.Linear(
            self.MULTISCALE_DIM, HORIZON * NT, bias=False,
        )
        self.multiscale_s = nn.Linear(
            self.MULTISCALE_DIM, HORIZON * NS, bias=False,
        )
        self.calendar_t = nn.Linear(4, HORIZON * NT, bias=False)
        self.calendar_s = nn.Linear(4, HORIZON * NS, bias=False)
        if use_persistence:
            if prepared is None:
                raise ValueError("prepared data are required for persistence")
            current_weight = prepared["current_eof_weight"]
            current_bias = prepared["current_eof_bias"]
            persistence = prepared["persistence_prior"]
        else:
            current_weight = np.zeros((6, NT + NS), dtype=np.float32)
            current_bias = np.zeros(NT + NS, dtype=np.float32)
            persistence = np.zeros((HORIZON, NT + NS), dtype=np.float32)
        self.register_buffer(
            "current_eof_weight",
            torch.as_tensor(current_weight, dtype=torch.float32),
        )
        self.register_buffer(
            "current_eof_bias",
            torch.as_tensor(current_bias, dtype=torch.float32),
        )
        self.register_buffer(
            "persistence_prior",
            torch.as_tensor(persistence, dtype=torch.float32),
        )

    @staticmethod
    def multiscale_features(tt):
        level = tt[:, -1:]
        centered_means, variability = [], []
        for width in (6, 24, 72, 168):
            segment = tt[:, -width:]
            centered_means.append(segment.mean(dim=1, keepdim=True) - level)
            variability.append(segment.std(dim=1, keepdim=True, unbiased=False))
        trends = []
        for width in (24, 72, 168):
            segment = tt[:, -width:]
            half = width // 2
            trends.append(
                segment[:, half:].mean(dim=1, keepdim=True)
                - segment[:, :half].mean(dim=1, keepdim=True)
            )
        lag_changes = [
            level - tt[:, -lag - 1:-lag]
            for lag in (6, 24, 72)
        ]
        return torch.cat(
            centered_means + trends + lag_changes + variability, dim=1,
        )

    def forward(self, history, context):
        tt = history[..., 0]
        level = tt[:, -1:]
        centered = tt - level
        multiscale = self.multiscale_features(tt)
        calendar = context[:, 6:10]
        temp = (
            self.tt_t(self.encoder_t(centered))
            + self.level_t(level)
            + self.context_t(context[:, :3])
            + self.multiscale_t(multiscale)
            + self.calendar_t(calendar)
        )
        salt = (
            self.tt_s(self.encoder_s(centered))
            + self.level_s(level)
            + self.context_s(context[:, 3:6])
            + self.multiscale_s(multiscale)
            + self.calendar_s(calendar)
        )
        network = torch.cat([
            temp.view(-1, HORIZON, NT),
            salt.view(-1, HORIZON, NS),
        ], dim=-1)
        current_z = (
            context[:, :6] @ self.current_eof_weight
            + self.current_eof_bias
        )
        prior = current_z[:, None, :] * self.persistence_prior[None]
        return network + prior


def build_forecast_model(args, prepared):
    if args.model_variant == "legacy":
        return NLinearEOF()
    return CausalLowRankNLinearEOF(
        args.model_rank, prepared,
        use_persistence=args.persistence_prior,
    )


def decode_np(z, anchor, prepared):
    coefficient = z * prepared["coefficient_scale"].reshape(1, 1, -1)
    residual = np.concatenate([
        prepared["eof_t"].inverse(coefficient[..., :NT]),
        prepared["eof_s"].inverse(coefficient[..., NT:]),
    ], axis=-1)
    return anchor + prepared["mean_residual"].reshape(1, 1, -1) + residual


class BaseLoss(nn.Module):
    """Supervised forecast loss plus the current optimized physics module.

    OASP = optimized acoustic observation consistency + train-supported
    two-sided N2-proxy static-stability envelope. CTRP is intentionally not
    part of the current module.
    """

    def __init__(
        self, prepared, acoustic_weight=0.0, stability_weight=0.0,
        physics_mode="none", loss_mode="global_huber",
        temp_latent_weight=0.03, salt_latent_weight=0.03,
    ):
        super().__init__()
        if physics_mode not in ("none", "optimized"):
            raise ValueError(f"Unknown physics_mode={physics_mode!r}")
        self.layers = prepared["layers"]
        self.acoustic_weight = float(acoustic_weight)
        self.stability_weight = float(stability_weight)
        self.physics_mode = str(physics_mode)
        self.loss_mode = loss_mode
        self.temp_latent_weight = float(temp_latent_weight)
        self.salt_latent_weight = float(salt_latent_weight)

        self.register_buffer("scale", torch.as_tensor(
            prepared["coefficient_scale"], dtype=torch.float32
        ).view(1, 1, -1))
        self.register_buffer("basis_t", torch.as_tensor(
            prepared["eof_t"].basis, dtype=torch.float32
        ))
        self.register_buffer("basis_s", torch.as_tensor(
            prepared["eof_s"].basis, dtype=torch.float32
        ))
        self.register_buffer("mean_residual", torch.as_tensor(
            prepared["mean_residual"], dtype=torch.float32
        ))
        self.register_buffer("temp_global_scale", torch.tensor(
            prepared["temp_scale"], dtype=torch.float32
        ))
        self.register_buffer("salt_global_scale", torch.tensor(
            prepared["salt_scale"], dtype=torch.float32
        ))
        self.register_buffer("depths", torch.as_tensor(
            prepared["depths"], dtype=torch.float32
        ))
        self.register_buffer("dz", torch.as_tensor(
            prepared["dz"], dtype=torch.float32
        ))

        for key in (
            "opt_tt_offset", "opt_tt_slope", "opt_tt_anomaly_scale",
            "opt_acoustic_reliability", "opt_center_scale",
            "opt_change_scale", "opt_center_reliability",
            "opt_change_reliability",
        ):
            self.register_buffer(
                key, torch.as_tensor(prepared[key], dtype=torch.float64)
            )
        for key in ("n2_proxy_floor", "n2_proxy_ceiling", "n2_proxy_scale"):
            self.register_buffer(
                key, torch.as_tensor(prepared[key], dtype=torch.float64)
            )
        self.sec_theta = float(prepared["sec_theta"])

    def _decode_profiles(self, residual, anchor):
        temp = (
            anchor[..., :self.layers]
            + self.mean_residual[:self.layers]
            + residual[..., :self.layers]
        )
        salt = (
            anchor[..., self.layers:]
            + self.mean_residual[self.layers:]
            + residual[..., self.layers:]
        )
        return temp, salt

    def forward(self, prediction_z, true_z, target, anchor, history=None, future_tt=None):
        coefficient = prediction_z * self.scale
        residual = torch.cat([
            coefficient[..., :NT] @ self.basis_t,
            coefficient[..., NT:] @ self.basis_s,
        ], dim=-1)
        truth = target - anchor - self.mean_residual

        temp_huber = F.huber_loss(
            residual[..., :self.layers] / self.temp_global_scale,
            truth[..., :self.layers] / self.temp_global_scale,
        )
        temp_mse = F.mse_loss(
            residual[..., :self.layers] / self.temp_global_scale,
            truth[..., :self.layers] / self.temp_global_scale,
        )
        if self.loss_mode == "global_mse":
            profile = temp_mse
        elif self.loss_mode == "mixed_global":
            profile = 0.7 * temp_mse + 0.3 * temp_huber
        else:
            profile = temp_huber
        profile += 0.5 * F.huber_loss(
            residual[..., self.layers:] / self.salt_global_scale,
            truth[..., self.layers:] / self.salt_global_scale,
        )
        latent_t = F.huber_loss(prediction_z[..., :NT], true_z[..., :NT])
        latent_s = F.huber_loss(prediction_z[..., NT:], true_z[..., NT:])
        loss = (
            profile
            + self.temp_latent_weight * latent_t
            + 0.5 * self.salt_latent_weight * latent_s
        )

        if self.physics_mode != "optimized":
            return loss

        needs_profiles = self.acoustic_weight > 0 or self.stability_weight > 0
        if not needs_profiles:
            return loss
        temp, salt = self._decode_profiles(residual, anchor)

        # Optimized acoustic consistency: level + centered dynamics + 6h changes.
        if self.acoustic_weight > 0 and future_tt is not None:
            c = sound_speed_torch(
                temp.double(), salt.double(), self.depths.double()
            )
            physical_tt = self.sec_theta * torch.sum(
                self.dz.double()
                * (1.0 / c[..., :-1] + 1.0 / c[..., 1:]),
                dim=-1,
            )
            predicted_tt = self.opt_tt_offset + self.opt_tt_slope * physical_tt
            observed_tt = future_tt.double()
            error = predicted_tt - observed_tt

            level = F.smooth_l1_loss(
                error / self.opt_tt_anomaly_scale,
                torch.zeros_like(error),
            )
            centered_error = error - error.mean(dim=1, keepdim=True)
            centered = F.smooth_l1_loss(
                centered_error / self.opt_center_scale,
                torch.zeros_like(centered_error),
            )
            change_error = error[:, 6:] - error[:, :-6]
            change = F.smooth_l1_loss(
                change_error / self.opt_change_scale,
                torch.zeros_like(change_error),
            )
            acoustic_loss = (
                0.25 * self.opt_acoustic_reliability * level
                + 0.50 * self.opt_center_reliability * centered
                + 0.25 * self.opt_change_reliability * change
            )
            loss = loss + self.acoustic_weight * acoustic_loss

        # Train-supported N2-like static-stability envelope.
        if self.stability_weight > 0:
            rho = density_torch(temp.double(), salt.double())
            density_gradient = (rho[..., 1:] - rho[..., :-1]) / self.dz.double()
            rho_mid = 0.5 * (rho[..., 1:] + rho[..., :-1])
            n2_proxy = GRAVITY * density_gradient / torch.clamp(rho_mid, min=1.0)

            lower_violation = F.relu(
                (self.n2_proxy_floor - n2_proxy) / self.n2_proxy_scale
            )
            stability_loss = F.smooth_l1_loss(
                lower_violation, torch.zeros_like(lower_violation)
            )
            upper_violation = F.relu(
                (n2_proxy - self.n2_proxy_ceiling) / self.n2_proxy_scale
            )
            stability_loss = stability_loss + 0.25 * F.smooth_l1_loss(
                upper_violation, torch.zeros_like(upper_violation)
            )
            loss = loss + self.stability_weight * stability_loss

        return loss

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def anomaly_r2(prediction, target, anchor):
    """R2 of anomalies relative to the independent 2024 anchor."""
    predicted_anomaly = np.asarray(prediction, dtype=np.float64) - anchor
    true_anomaly = np.asarray(target, dtype=np.float64) - anchor
    denominator = np.sum(
        (true_anomaly - np.mean(true_anomaly)) ** 2,
        dtype=np.float64,
    )
    if denominator <= 1e-12:
        return float("nan")
    numerator = np.sum(
        (predicted_anomaly - true_anomaly) ** 2,
        dtype=np.float64,
    )
    return float(1.0 - numerator / denominator)


def metrics(prediction, target, prepared, anchor=None):
    layers, depths, selected = prepared["layers"], prepared["depths"], prepared["upper"]
    tp, sp = prediction[..., :layers], prediction[..., layers:]
    tt, st = target[..., :layers], target[..., layers:]
    surface, deep = depths <= 150, depths > 150
    unobserved = surface.copy()
    unobserved[selected] = False
    result = {
        "Temp_RMSE": rmse(tp, tt),
        "Temp_MAE": mae(tp, tt),
        "Salt_RMSE": rmse(sp, st),
        "Salt_MAE": mae(sp, st),
        "Temp_Surface_RMSE": rmse(tp[..., surface], tt[..., surface]),
        "Temp_UpperUnobserved_RMSE": rmse(tp[..., unobserved], tt[..., unobserved]),
        "Temp_Deep_RMSE": rmse(tp[..., deep], tt[..., deep]),
        "Temp_Selected3_RMSE": rmse(tp[..., selected], tt[..., selected]),
    }
    if anchor is not None:
        result["Temp_Anomaly_R2"] = anomaly_r2(
            tp, tt, anchor[..., :layers],
        )
        result["Salt_Anomaly_R2"] = anomaly_r2(
            sp, st, anchor[..., layers:],
        )
    for name, start, end in (("h1_24", 0, 24), ("h25_48", 24, 48), ("h49_72", 48, 72)):
        result[f"Temp_RMSE_{name}"] = rmse(tp[:, start:end], tt[:, start:end])
        result[f"Salt_RMSE_{name}"] = rmse(sp[:, start:end], st[:, start:end])
    return result


@torch.no_grad()
def collect_latent(model, loader, device):
    model.eval()
    output = [[] for _ in range(5)]
    for history, context, true_z, target, anchor, _ in loader:
        values = (
            model(
                history.to(device, non_blocking=True),
                context.to(device, non_blocking=True),
            ).cpu().numpy(),
            true_z.numpy(), target.numpy(), anchor.numpy(), context.numpy(),
        )
        for destination, value in zip(output, values):
            destination.append(value)
    return tuple(np.concatenate(x) for x in output)


def validation_score(model, prepared, device, anchor_metrics, loader=None):
    if loader is None:
        loader = prepared["loaders"]["val"]
    pred_z, _, target, anchor, _ = collect_latent(model, loader, device)
    score = metrics(
        decode_np(pred_z, anchor, prepared), target, prepared, anchor,
    )
    return score["Temp_RMSE"] / anchor_metrics["Temp_RMSE"] + 0.2 * score["Salt_RMSE"] / anchor_metrics["Salt_RMSE"]


def train_base(
    seed, prepared, args, device, anchor_metrics, spec,
    validation_loader=None,
):
    seed_all(seed)
    model = build_forecast_model(args, prepared).to(device)
    criterion = BaseLoss(
        prepared,
        acoustic_weight=spec.get("acoustic", 0.0),
        stability_weight=spec.get("stability", 0.0),
        physics_mode=spec.get("physics_mode", "none"),
        loss_mode=spec["loss_mode"],
        temp_latent_weight=spec["temp_latent"],
        salt_latent_weight=spec["salt_latent"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)
    loader = prepared["loaders"]["train"]
    loader.generator.manual_seed(seed)
    best, best_score, best_epoch, bad = copy.deepcopy(model.state_dict()), math.inf, 0, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for history, context, true_z, target, anchor, future_tt in loader:
            history = history.to(device, non_blocking=True)
            context = context.to(device, non_blocking=True)
            true_z = true_z.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            anchor = anchor.to(device, non_blocking=True)
            future_tt = (
                future_tt.to(device, non_blocking=True)
                if criterion.acoustic_weight > 0 else None
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                model(history, context), true_z, target, anchor,
                future_tt=future_tt,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        score = validation_score(
            model, prepared, device, anchor_metrics, validation_loader,
        )
        scheduler.step(score)
        if score < best_score - 1e-8:
            best, best_score, best_epoch, bad = copy.deepcopy(model.state_dict()), score, epoch, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best)
    return model, best_epoch, time.perf_counter() - started

class Refiner(nn.Module):
    def __init__(self, method="identity", transition=None, bias=None, q=None, r=None):
        super().__init__()
        self.method = method
        zero, one = np.zeros(NT + NS, np.float32), np.ones(NT + NS, np.float32)
        for name, value in (("transition", zero if transition is None else transition), ("bias", zero if bias is None else bias), ("q", one if q is None else q), ("r", one if r is None else r)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))

    def forward(self, observation):
        if self.method == "identity":
            return observation
        def observation_variance(horizon):
            if self.r.ndim == 1:
                return self.r
            if self.r.shape[0] == 3:
                return self.r[min(horizon // 24, 2)]
            return self.r[horizon]

        first_r = observation_variance(0)
        filtered, variance = [observation[:, 0]], [first_r]
        priors, prior_variance = [observation[:, 0]], [first_r]
        for horizon in range(1, observation.shape[1]):
            prior = self.transition * filtered[-1] + self.bias
            pv = self.transition.square() * variance[-1] + self.q
            current_r = observation_variance(horizon)
            gain = pv / torch.clamp(pv + current_r, min=1e-8)
            filtered.append(prior + gain * (observation[:, horizon] - prior))
            variance.append((1 - gain) * pv)
            priors.append(prior)
            prior_variance.append(pv)
        smooth = list(filtered)
        for horizon in range(observation.shape[1] - 2, -1, -1):
            gain = variance[horizon] * self.transition / torch.clamp(prior_variance[horizon + 1], min=1e-8)
            smooth[horizon] = filtered[horizon] + gain * (smooth[horizon + 1] - priors[horizon + 1])
        return torch.stack(smooth, dim=1)


class DualRefiner(nn.Module):
    def __init__(self, temp, salt):
        super().__init__()
        self.temp, self.salt = temp, salt

    def forward(self, value):
        return torch.cat([self.temp(value)[..., :NT], self.salt(value)[..., NT:]], dim=-1)








@torch.no_grad()
def refine_np(refiner, value):
    return refiner(torch.as_tensor(value, dtype=torch.float32)).cpu().numpy()


def fit_ar_pairs(x, y):
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    denominator = np.sum(xc * xc, axis=0)
    transition = np.divide(np.sum(xc * yc, axis=0), denominator, out=np.zeros_like(denominator), where=denominator > 1e-10)
    transition = np.clip(transition, -0.995, 0.995)
    bias = ym - transition * xm
    q = np.maximum(np.mean((y - transition * x - bias) ** 2, axis=0), 1e-5)
    return transition, bias, q


def fit_dynamics(true_z):
    x = true_z[:, :-1].reshape(-1, NT + NS)
    y = true_z[:, 1:].reshape(-1, NT + NS)
    return fit_ar_pairs(x, y)


def fit_dynamics_unique(series_z, times_ns):
    contiguous = np.diff(times_ns) == np.int64(3_600_000_000_000)
    if contiguous.sum() < 100:
        raise ValueError("Too few contiguous hourly transitions for deduplicated BLTS")
    return fit_ar_pairs(series_z[:-1][contiguous], series_z[1:][contiguous])


def fit_temp_gate(prediction, target, anchor, depths):
    residual, truth = prediction - anchor, target - anchor
    gate = np.ones((HORIZON, len(depths)))
    for mask in ((depths > 150) & (depths <= 500), depths > 500):
        p, y = residual[..., mask], truth[..., mask]
        value = np.sum(p * y) / max(float(np.sum(p * p)), 1e-12)
        gate[:, mask] = np.clip(value, 0, 1)
    return gate


def fit_salt_gate(prediction, target, anchor, depths):
    p, y = prediction - anchor, target - anchor
    denominator = np.sum(p * p, axis=(0, 1))
    raw = np.clip(np.divide(np.sum(p * y, axis=(0, 1)), denominator, out=np.zeros(len(depths)), where=denominator > 1e-12), 0, 1)
    coordinate, best = np.log1p(depths), None
    for bandwidth in (0.0, 0.15, 0.30, 0.60):
        if bandwidth == 0:
            smooth = raw
        else:
            weight = np.exp(-0.5 * ((coordinate[:, None] - coordinate[None, :]) / bandwidth) ** 2) * np.maximum(denominator[None], 1e-12)
            smooth = weight @ raw / np.maximum(weight.sum(1), 1e-12)
        for strength in (0.25, 0.5, 0.75, 1.0):
            gate = np.clip(1 + strength * (smooth - 1), 0, 1)
            score = rmse(anchor + p * gate, target)
            if best is None or score < best[0]:
                best = score, gate
    return np.broadcast_to(best[1], (HORIZON, len(depths))).copy()


def pseudo_observation(context, anchor, prepared):
    index, layers = prepared["upper"], prepared["layers"]
    anomaly_t = context[:, :3] * prepared["upper_t_std"] + prepared["upper_t_mean"]
    anomaly_s = context[:, 3:6] * prepared["upper_s_std"] + prepared["upper_s_mean"]
    return anchor[..., index] + anomaly_t[:, None], anchor[..., layers + index] + anomaly_s[:, None]


def sparse_blend(base, target, pseudo, indices):
    return min(
        ((rmse(base[..., indices] + blend * (pseudo - base[..., indices]), target[..., indices]), blend) for blend in (0, 0.25, 0.5, 0.75, 1)),
        key=lambda item: item[0],
    )[1]


def fit_gain(innovation, base, target, depths, ridge, max_depth):
    features = innovation.shape[-1]
    gain = np.zeros((HORIZON, features, len(depths)))
    active = np.flatnonzero(depths <= max_depth)
    for start, end in ((0, 24), (24, 48), (48, 72)):
        x = innovation[:, start:end].reshape(-1, features)
        y = (target[:, start:end] - base[:, start:end]).reshape(-1, len(depths))[:, active]
        gram = x.T @ x
        penalty = ridge * max(float(np.trace(gram) / features), 1e-8)
        beta = np.linalg.solve(gram + penalty * np.eye(features), x.T @ y)
        gain[start:end, :, active] = beta.T.reshape(1, len(active), features).transpose(0, 2, 1)
    return gain


def stratification_state(context, anchor, prepared):
    index, layers = prepared["upper"], prepared["layers"]
    anomaly_t = context[:, :3] * prepared["upper_t_std"] + prepared["upper_t_mean"]
    anomaly_s = context[:, 3:6] * prepared["upper_s_std"] + prepared["upper_s_mean"]
    rho = density_np(anchor[:, 0, index] + anomaly_t, anchor[:, 0, layers + index] + anomaly_s)
    return (rho[:, -1] - rho[:, 0]) / max(float(prepared["depths"][index[-1]] - prepared["depths"][index[0]]), 1e-6)


def weights(state, centers, bandwidth):
    distance = np.abs(state[:, None] - centers[None])
    if bandwidth <= 0:
        return np.eye(len(centers))[np.argmin(distance, axis=1)]
    value = -0.5 * (distance / bandwidth) ** 2
    value -= value.max(axis=1, keepdims=True)
    value = np.exp(value)
    return value / np.maximum(value.sum(1, keepdims=True), 1e-12)


def fit_regime_sua(train, val, prepared, regimes=2, evaluation_mask=None):
    tr_base, tr_target, tr_innovation, tr_state = train
    va_base, va_target, va_innovation, va_state = val
    index, depths = prepared["upper"], prepared["depths"]
    if regimes == 1:
        labels = np.zeros(len(tr_state), dtype=int)
        centers = np.array([tr_state.mean()])
    elif regimes == 2:
        threshold = np.median(tr_state)
        labels = (tr_state > threshold).astype(int)
        centers = np.array([tr_state[labels == r].mean() for r in range(2)])
    else:
        raise ValueError("regimes must be 1 or 2")
    labels = np.argmin(np.abs(tr_state[:, None] - centers[None]), axis=1)
    if evaluation_mask is None:
        evaluation = np.ones(len(depths), bool)
        evaluation[index] = False
    else:
        evaluation = np.asarray(evaluation_mask, dtype=bool)
        if evaluation.shape != depths.shape or not evaluation.any():
            raise ValueError("evaluation_mask must select at least one target depth")
    best = None
    for max_depth in (50.0, 150.0, 500.0, np.inf):
        for ridge in (0.01, 0.1, 1.0, 10.0):
            gains = np.stack([fit_gain(tr_innovation[labels == r], tr_base[labels == r], tr_target[labels == r], depths, ridge, max_depth) for r in range(regimes)])
            spacing = max(float(np.ptp(centers)), float(np.std(tr_state)) * 0.1, 1e-8)
            for softness in (0.0, 0.25, 0.5, 1.0, 2.0):
                bandwidth = softness * spacing
                correction = np.einsum("nr,nhk,rhkl->nhl", weights(va_state, centers, bandwidth), va_innovation, gains)
                for gamma in (0.25, 0.5, 0.75, 1.0, 1.25):
                    score = rmse((va_base + gamma * correction)[..., evaluation], va_target[..., evaluation])
                    if best is None or score < best[0]:
                        best = score, gains, bandwidth, gamma, ridge, max_depth
    return best[1], centers, best[2], best[3], {
        "SUA_Val_RMSE": best[0], "SUA_Ridge": best[4], "SUA_MaxDepth": best[5],
        "SUA_Gamma": best[3], "SUA_Regimes": regimes,
    }






class Processor(nn.Module):
    def __init__(self, prepared, refiner, temp_gate, salt_gate, gains, centers,
                 bandwidth, gamma, salt_blend, sua_features="temp", innovation_scale=None):
        super().__init__()
        self.refiner, self.layers = refiner, prepared["layers"]
        self.bandwidth, self.gamma, self.salt_blend = float(bandwidth), float(gamma), float(salt_blend)
        self.sua_features = sua_features
        if innovation_scale is None:
            innovation_scale = np.ones(gains.shape[2], dtype=np.float32)
        buffers = {
            "scale": prepared["coefficient_scale"].reshape(1, 1, -1), "basis_t": prepared["eof_t"].basis,
            "basis_s": prepared["eof_s"].basis, "temp_gate": temp_gate, "salt_gate": salt_gate,
            "mean_residual": prepared["mean_residual"],
            "gains": gains, "centers": centers, "innovation_scale": innovation_scale,
            "indices": prepared["upper"], "upper_depths": prepared["depths"][prepared["upper"]],
            "upper_t_mean": prepared["upper_t_mean"], "upper_t_std": prepared["upper_t_std"],
            "upper_s_mean": prepared["upper_s_mean"], "upper_s_std": prepared["upper_s_std"],
        }
        for name, value in buffers.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.long if name == "indices" else torch.float32))

    def forward(self, z, anchor, context):
        coefficient = self.refiner(z) * self.scale
        raw_t = anchor[..., :self.layers] + self.mean_residual[:self.layers] + coefficient[..., :NT] @ self.basis_t
        raw_s = anchor[..., self.layers:] + self.mean_residual[self.layers:] + coefficient[..., NT:] @ self.basis_s
        base_t = anchor[..., :self.layers] + (raw_t - anchor[..., :self.layers]) * self.temp_gate
        base_s = anchor[..., self.layers:] + (raw_s - anchor[..., self.layers:]) * self.salt_gate
        anomaly_t = context[:, :3] * self.upper_t_std + self.upper_t_mean
        anomaly_s = context[:, 3:6] * self.upper_s_std + self.upper_s_mean
        pseudo_t = anchor.index_select(-1, self.indices) + anomaly_t[:, None]
        pseudo_s = anchor[..., self.layers:].index_select(-1, self.indices) + anomaly_s[:, None]
        innovation_t = pseudo_t - base_t.index_select(-1, self.indices)
        innovation_s = pseudo_s - base_s.index_select(-1, self.indices)
        innovation = torch.cat([innovation_t, innovation_s], dim=-1) if self.sua_features == "ts" else innovation_t
        innovation = innovation / self.innovation_scale
        origin_t = anchor[:, 0, :self.layers].index_select(-1, self.indices) + anomaly_t
        origin_s = anchor[:, 0, self.layers:].index_select(-1, self.indices) + anomaly_s
        origin_density = density_torch(origin_t, origin_s)
        state = (origin_density[:, -1] - origin_density[:, 0]) / torch.clamp(self.upper_depths[-1] - self.upper_depths[0], min=1e-6)
        distance = torch.abs(state[:, None] - self.centers[None])
        if self.bandwidth <= 0:
            weight = F.one_hot(torch.argmin(distance, dim=1), num_classes=len(self.centers)).to(base_t.dtype)
        else:
            weight = torch.softmax(-0.5 * (distance / self.bandwidth).square(), dim=1)
        output_t = base_t + self.gamma * torch.einsum("br,bhk,rhkl->bhl", weight, innovation, self.gains)
        scatter = self.indices.view(1, 1, -1).expand(output_t.shape[0], output_t.shape[1], -1)
        output_t = output_t.scatter(-1, scatter, pseudo_t)
        selected_s = base_s.index_select(-1, self.indices)
        output_s = base_s.scatter(-1, scatter, selected_s + self.salt_blend * (pseudo_s - selected_s))
        return torch.cat([output_t, output_s], dim=-1)


@torch.no_grad()
def predict_processed(model, loader, processor, device):
    model.eval(), processor.eval()
    prediction, target, anchor, future_tt = [], [], [], []
    for history, context, _, y, a, tt in loader:
        history_device = history.to(device, non_blocking=True)
        context_device = context.to(device, non_blocking=True)
        anchor_device = a.to(device, non_blocking=True)
        result = processor(
            model(history_device, context_device), anchor_device, context_device,
        )
        prediction.append(result.cpu().numpy())
        target.append(y.numpy())
        anchor.append(a.numpy())
        future_tt.append(tt.numpy())
    return np.concatenate(prediction), np.concatenate(target), np.concatenate(anchor), np.concatenate(future_tt)


def calibrate_processor(
    refiner, split_data, prepared, device, sua_features="temp", regimes=2,
    evaluation_mask=None,
):
    profiles = {}
    for name, (pred_z, _, target, anchor, context) in split_data.items():
        profiles[name] = [decode_np(refine_np(refiner, pred_z), anchor, prepared), target, anchor, context]
    layers, depths, index = prepared["layers"], prepared["depths"], prepared["upper"]
    val_prediction, val_target, val_anchor, _ = profiles["val"]
    temp_gate = fit_temp_gate(val_prediction[..., :layers], val_target[..., :layers], val_anchor[..., :layers], depths)
    salt_gate = fit_salt_gate(val_prediction[..., layers:], val_target[..., layers:], val_anchor[..., layers:], depths)
    bundles = {}
    for name, (prediction, target, anchor, context) in profiles.items():
        base_t = anchor[..., :layers] + (prediction[..., :layers] - anchor[..., :layers]) * temp_gate
        base_s = anchor[..., layers:] + (prediction[..., layers:] - anchor[..., layers:]) * salt_gate
        pseudo_t, pseudo_s = pseudo_observation(context, anchor, prepared)
        bundles[name] = (
            base_t, base_s, target, anchor, context, pseudo_t, pseudo_s,
            pseudo_t - base_t[..., index], pseudo_s - base_s[..., index],
        )
    salt_blend = sparse_blend(bundles["val"][1], bundles["val"][2][..., layers:], bundles["val"][6], index)
    train = bundles["train"]
    val = bundles["val"]
    if sua_features == "ts":
        train_raw = np.concatenate([train[7], train[8]], axis=-1)
        val_raw = np.concatenate([val[7], val[8]], axis=-1)
        innovation_scale = np.std(train_raw, axis=(0, 1))
        innovation_scale = np.maximum(innovation_scale, np.median(innovation_scale) * 0.05 + 1e-8)
    elif sua_features == "temp":
        train_raw, val_raw = train[7], val[7]
        innovation_scale = np.ones(train_raw.shape[-1], dtype=np.float32)
    else:
        raise ValueError("sua_features must be 'temp' or 'ts'")
    gains, centers, bandwidth, gamma, meta = fit_regime_sua(
        (train[0], train[2][..., :layers], train_raw / innovation_scale,
         stratification_state(train[4], train[3], prepared)),
        (val[0], val[2][..., :layers], val_raw / innovation_scale,
         stratification_state(val[4], val[3], prepared)),
        prepared, regimes, evaluation_mask,
    )
    processor = Processor(
        prepared, copy.deepcopy(refiner), temp_gate, salt_gate, gains, centers,
        bandwidth, gamma, salt_blend, sua_features, innovation_scale,
    ).to(device)
    return processor, {
        **meta, "Assimilation_Method": "SUA",
        "SUA_Features": sua_features.upper(), "Salt_AP3_Blend": salt_blend,
        "Temp_Gate_Deep": float(temp_gate[:, depths > 150].mean()),
        "Salt_Gate_Deep": float(salt_gate[:, depths > 150].mean()),
    }


@torch.no_grad()
def process_z(processor, z, anchor, context, device, batch_size=64):
    processor.eval()
    output = []
    for start in range(0, len(z), batch_size):
        end = start + batch_size
        output.append(
            processor(
                torch.as_tensor(z[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(anchor[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(context[start:end], dtype=torch.float32, device=device),
            ).cpu().numpy()
        )
    return np.concatenate(output)


def processor_with_refiner(base_processor, refiner, device):
    processor = copy.deepcopy(base_processor)
    processor.refiner = copy.deepcopy(refiner)
    return processor.to(device)


class DepthRouteProcessor(nn.Module):
    """Use thermohaline SUA only where a validation-selected depth mask permits it."""
    def __init__(self, baseline, candidate, depth_mask, beta):
        super().__init__()
        self.baseline, self.candidate = baseline, candidate
        self.beta = float(beta)
        self.register_buffer("depth_mask", torch.as_tensor(depth_mask, dtype=torch.float32).view(1, 1, -1))

    def forward(self, z, anchor, context):
        safe = self.baseline(z, anchor, context)
        proposed = self.candidate(z, anchor, context)
        return safe + self.beta * self.depth_mask * (proposed - safe)


def route_mask(depths, transition, width, layers):
    if width <= 0:
        temp_mask = (depths > transition).astype(np.float32)
    else:
        value = np.clip((depths - transition) / width, -20.0, 20.0)
        temp_mask = (1.0 / (1.0 + np.exp(-value))).astype(np.float32)
    # The candidate only supplies a temperature correction. Salt remains the B1 estimate.
    return np.concatenate([temp_mask, np.zeros(layers, dtype=np.float32)])


def variable_score(prediction, target, prepared, variable, start=0, end=HORIZON):
    layers = prepared["layers"]
    if variable == "temp":
        return rmse(prediction[:, start:end, :layers], target[:, start:end, :layers])
    return rmse(prediction[:, start:end, layers:], target[:, start:end, layers:])


def select_blts_from_dynamics(
    dynamics, validation, base_processor, prepared, device,
    fit_fraction=0.60, selection_end_fraction=1.00,
):
    """Estimate R early and select Q/R on a later chronological segment."""
    predicted_z, true_z, target, anchor, context = validation
    fit_end = int(fit_fraction * len(predicted_z))
    selection_end = int(selection_end_fraction * len(predicted_z))
    if fit_end < 20 or selection_end - fit_end < 20:
        raise ValueError("Validation set is too small for BLTS validation")

    transition, bias, q = dynamics
    r = np.maximum(np.mean((predicted_z[:fit_end] - true_z[:fit_end]) ** 2, axis=(0, 1)), 1e-5)
    identity = Refiner()
    selected = {}

    for variable in ("temp", "salt"):
        best = None
        for q_scale in (0.25, 0.5, 1.0, 2.0, 4.0):
            for r_scale in (0.25, 0.5, 1.0, 2.0, 4.0):
                candidate = Refiner("rts", transition, bias, q * q_scale, r * r_scale)
                dual = DualRefiner(candidate, copy.deepcopy(identity)) if variable == "temp" else DualRefiner(copy.deepcopy(identity), candidate)
                processor = processor_with_refiner(base_processor, dual, device)
                prediction = process_z(
                    processor, predicted_z[fit_end:selection_end],
                    anchor[fit_end:selection_end],
                    context[fit_end:selection_end], device,
                )
                score = variable_score(
                    prediction, target[fit_end:selection_end], prepared, variable,
                )
                if best is None or score < best[0]:
                    best = score, q_scale, r_scale
        selected[variable] = {
            "refiner": Refiner("rts", transition, bias, q * best[1], r * best[2]),
            "q": best[1], "r": best[2], "score": best[0],
        }

    return DualRefiner(selected["temp"]["refiner"], selected["salt"]["refiner"]), {
        "Temp_Q_Scale": selected["temp"]["q"], "Temp_R_Scale": selected["temp"]["r"],
        "Salt_Q_Scale": selected["salt"]["q"], "Salt_R_Scale": selected["salt"]["r"],
        "Temp_BLTS_Val_RMSE": selected["temp"]["score"],
        "Salt_BLTS_Val_RMSE": selected["salt"]["score"],
        "BLTS_Fit_Fraction": fit_fraction,
        "BLTS_Selection_End_Fraction": selection_end_fraction,
    }


def select_full_blts(train_true_z, validation, base_processor, prepared, device):
    return select_blts_from_dynamics(
        fit_dynamics(train_true_z), validation, base_processor, prepared, device,
    )


def select_unique_blts(prepared, validation, base_processor, device):
    dynamics = fit_dynamics_unique(prepared["train_series_z"], prepared["train_times_ns"])
    refiner, meta = select_blts_from_dynamics(
        dynamics, validation, base_processor, prepared, device,
        fit_fraction=0.50, selection_end_fraction=0.75,
    )
    meta["BLTS_Dynamics"] = "deduplicated_hourly"
    return refiner, meta




def horizon_error_variance(predicted_z, true_z):
    """Stable horizon/mode error variance estimated on a calibration segment."""
    raw = np.maximum(np.mean((predicted_z - true_z) ** 2, axis=0), 1e-5)
    smooth = np.empty_like(raw)
    for horizon in range(HORIZON):
        start, end = max(0, horizon - 5), min(HORIZON, horizon + 7)
        smooth[horizon] = raw[start:end].mean(axis=0)
    global_mode = raw.mean(axis=0, keepdims=True)
    return np.maximum(0.50 * smooth + 0.50 * global_mode, 1e-5)


def causal_vintage_fusion(z, origin_ns, valid_ns, variance, gamma, beta):
    """Fuse only forecasts issued before the current origin; salt is untouched."""
    z = np.asarray(z)
    origin_ns, valid_ns = np.asarray(origin_ns), np.asarray(valid_ns)
    if z.ndim != 3 or z.shape[1:] != (HORIZON, NT + NS):
        raise ValueError("z must have shape [N, 72, NT+NS]")
    if len(z) != len(origin_ns) or valid_ns.shape != (len(z), HORIZON):
        raise ValueError("Forecast arrays and timestamp metadata do not align")
    if not np.isfinite(z).all() or not np.isfinite(variance).all():
        raise ValueError("Fusion inputs must be finite")
    if np.any(variance <= 0):
        raise ValueError("Fusion variances must be positive")
    if len(origin_ns) > 1 and not np.all(np.diff(origin_ns) > 0):
        raise ValueError("Forecast origins must be strictly increasing")
    if variance.shape != (HORIZON, NT + NS):
        raise ValueError("variance must have shape [72, NT+NS]")
    if beta == 0:
        return z.copy()

    output = z.copy()
    bank = {}
    hour_ns = 3_600_000_000_000
    for sample, origin in enumerate(origin_ns):
        pending = []
        for horizon, valid_time in enumerate(valid_ns[sample]):
            candidates = [z[sample, horizon, :NT]]
            weights_ = [1.0 / variance[horizon, :NT]]
            for old_origin, old_horizon, old_value in bank.get(int(valid_time), ()):
                if old_origin >= origin:
                    raise AssertionError("Causal fusion encountered a future origin")
                age_hours = max(float(origin - old_origin) / hour_ns, 1.0)
                age_steps = age_hours / STRIDE
                candidates.append(old_value)
                weights_.append(
                    gamma ** age_steps / variance[old_horizon, :NT]
                )
            candidates = np.stack(candidates)
            weights_ = np.stack(weights_)
            fused = np.sum(candidates * weights_, axis=0) / np.maximum(
                weights_.sum(axis=0), 1e-12,
            )
            output[sample, horizon, :NT] = (
                z[sample, horizon, :NT]
                + beta * (fused - z[sample, horizon, :NT])
            )
            pending.append((int(valid_time), horizon, z[sample, horizon, :NT].copy()))
        # Write the current forecast only after every horizon has been read.
        for valid_time, horizon, value in pending:
            bank.setdefault(valid_time, []).append((int(origin), horizon, value))

    if not np.array_equal(output[..., NT:], z[..., NT:]):
        raise AssertionError("Causal fusion must not modify salinity modes")
    return output


def select_causal_fusion(gated_z, validation, metadata, identity_route, prepared, device):
    """Fit variance on V1 and select fusion on a cold-start V3 segment."""
    _, true_z, target, anchor, context = validation
    fit_end = int(0.50 * len(gated_z))
    selection_start = int(0.75 * len(gated_z))
    variance = horizon_error_variance(gated_z[:fit_end], true_z[:fit_end])
    best = None
    for gamma in (0.50, 0.75, 0.90, 1.00):
        fully_fused_z = causal_vintage_fusion(
            gated_z[selection_start:],
            metadata["origin_ns"][selection_start:],
            metadata["valid_ns"][selection_start:],
            variance, gamma, 1.0,
        )
        base_z = gated_z[selection_start:]
        fusion_delta = fully_fused_z - base_z
        for beta in (0.0, 0.25, 0.50, 0.75, 1.00):
            candidate_z = (
                base_z.copy() if beta == 0.0
                else base_z + beta * fusion_delta
            )
            prediction = process_z(
                identity_route, candidate_z, anchor[selection_start:],
                context[selection_start:], device,
            )
            score = variable_score(
                prediction, target[selection_start:], prepared, "temp",
            )
            if best is None or score < best[0]:
                best = score, gamma, beta
    return variance, best[1], best[2], {
        "Fusion_Val_Temp_RMSE": best[0], "Fusion_Gamma": best[1],
        "Fusion_Beta": best[2], "Fusion_Causal": True,
        "Fusion_Fit_Fraction": 0.50, "Fusion_Selection_Start_Fraction": 0.75,
        "Fusion_Selection_Cold_Start": True,
    }


def select_fusion_gamma_v2(gated_z, validation, metadata, identity_route,
                           prepared, device):
    """Fit variance on V1 and select only age decay on the disjoint V2."""
    _, true_z, target, anchor, context = validation
    fit_end = int(0.50 * len(gated_z))
    selection_end = int(0.75 * len(gated_z))
    variance = horizon_error_variance(gated_z[:fit_end], true_z[:fit_end])
    best = None
    for gamma in (0.50, 0.75, 0.90, 1.00):
        candidate_z = causal_vintage_fusion(
            gated_z[fit_end:selection_end],
            metadata["origin_ns"][fit_end:selection_end],
            metadata["valid_ns"][fit_end:selection_end],
            variance, gamma, 1.0,
        )
        prediction = process_z(
            identity_route, candidate_z, anchor[fit_end:selection_end],
            context[fit_end:selection_end], device,
        )
        score = variable_score(
            prediction, target[fit_end:selection_end], prepared, "temp",
        )
        if best is None or score < best[0]:
            best = score, gamma
    return variance, best[1], {
        "Fusion_V2_Val_Temp_RMSE": best[0],
        "Fusion_Gamma": best[1],
        "Fusion_Variance_Fit_Fraction": 0.50,
        "Fusion_Gamma_Selection_End_Fraction": 0.75,
    }


def eof_deep_reliability(prepared):
    """Deterministic mode weights from train-only EOF energy below 150 m."""
    basis = np.asarray(prepared["eof_t"].basis, dtype=np.float64)
    deep = prepared["depths"] > 150.0
    ratio = np.sum(basis[:, deep] ** 2, axis=1) / np.maximum(
        np.sum(basis**2, axis=1), 1e-12,
    )
    weight = 0.25 + 0.75 * np.sqrt(ratio / max(float(ratio.max()), 1e-12))
    return weight.astype(np.float32)


def apply_eof_fusion_gate(base_z, fused_z, mode_weight):
    if base_z.shape != fused_z.shape:
        raise ValueError("EOF fusion arrays must have identical shapes")
    weight = np.asarray(mode_weight, dtype=np.float32)
    if weight.shape != (NT,) or np.any((weight < 0) | (weight > 1)):
        raise ValueError("Temperature EOF reliability must have shape [NT] in [0,1]")
    output = base_z.copy()
    output[..., :NT] += weight.reshape(1, 1, -1) * (
        fused_z[..., :NT] - base_z[..., :NT]
    )
    output[..., NT:] = fused_z[..., NT:]
    return output

def fit_profile_reliability_router(current, candidate, target, prepared,
                                   depth_aware):
    """Analytic ridge blend on V3; returns only 3 or 12 small coefficients."""
    if current.shape != candidate.shape or current.shape != target.shape:
        raise ValueError("Router arrays must have identical profile shapes")
    layers, depths = prepared["layers"], prepared["depths"]
    horizon_groups = (
        np.arange(0, 24), np.arange(24, 48), np.arange(48, 72),
    )
    if depth_aware:
        depth_groups = (
            depths <= 50,
            (depths > 50) & (depths <= 150),
            (depths > 150) & (depths <= 500),
            depths > 500,
        )
    else:
        depth_groups = (np.ones(layers, dtype=bool),)

    alpha = np.zeros((HORIZON, layers), dtype=np.float32)
    selected = np.zeros(layers, dtype=bool)
    selected[prepared["upper"]] = True
    values = []
    for horizons in horizon_groups:
        for depth_mask in depth_groups:
            active = depth_mask & ~selected
            if not active.any():
                continue
            difference = (
                candidate[:, horizons, :layers][..., active]
                - current[:, horizons, :layers][..., active]
            )
            desired = (
                target[:, horizons, :layers][..., active]
                - current[:, horizons, :layers][..., active]
            )
            denominator = float(np.sum(difference**2))
            if depth_aware:
                prior = 1.0 if np.all(depths[active] > 150) else 0.0
            else:
                prior = 0.5
            ridge = 0.10 * denominator + 1e-10
            value = np.clip(
                (float(np.sum(difference * desired)) + ridge * prior)
                / (denominator + ridge),
                0.0, 1.0,
            )
            alpha[np.ix_(horizons, np.flatnonzero(active))] = value
            values.append(value)
    alpha[:, selected] = 0.0
    return alpha, {
        "Router_Depth_Aware": bool(depth_aware),
        "Router_Alpha_Mean": float(alpha[:, ~selected].mean()),
        "Router_Alpha_Min": float(np.min(values)),
        "Router_Alpha_Max": float(np.max(values)),
        "Router_Ridge_Ratio": 0.10,
        "Router_Fit_Segment": "validation_last_25pct",
    }


def apply_profile_reliability_router(current, candidate, alpha, prepared):
    layers = prepared["layers"]
    if alpha.shape != (HORIZON, layers):
        raise ValueError("Router alpha must have shape [72,layers]")
    temp = current[..., :layers] + alpha.reshape(1, HORIZON, layers) * (
        candidate[..., :layers] - current[..., :layers]
    )
    return np.concatenate([temp, candidate[..., layers:]], axis=-1)


def assert_overlap_consistency(true_z, valid_ns):
    """The same valid hour must carry the same target in overlapping windows."""
    seen = {}
    for sample in range(len(true_z)):
        for horizon, valid_time in enumerate(valid_ns[sample]):
            key = int(valid_time)
            value = true_z[sample, horizon]
            if key in seen and not np.allclose(seen[key], value, rtol=1e-5, atol=1e-6):
                raise AssertionError("Overlapping forecast windows contain different targets")
            seen.setdefault(key, value.copy())














def acoustic_metrics(prediction, future_tt, prepared):
    """Diagnostics matching the optimized acoustic module."""
    layers, depths = prepared["layers"], prepared["depths"]
    physical = physical_travel_time_np(
        prediction[..., :layers], prediction[..., layers:],
        depths, prepared["sec_theta"],
    )
    predicted = prepared["opt_tt_offset"] + prepared["opt_tt_slope"] * physical
    error = predicted - future_tt
    centered_error = error - error.mean(axis=1, keepdims=True)
    change_error = error[:, 6:] - error[:, :-6]
    return {
        "Acoustic_TT_RMSE_s": float(np.sqrt(np.mean(error ** 2))),
        "Acoustic_Centered_RMSE_s": float(np.sqrt(np.mean(centered_error ** 2))),
        "Acoustic_6hChange_RMSE_s": float(np.sqrt(np.mean(change_error ** 2))),
    }

def collect_future_tt(loader):
    """Return future observed travel times in deterministic loader order."""
    values = []
    for batch in loader:
        values.append(batch[-1].numpy())
    return np.concatenate(values)


def physics_metrics(prediction, target, anchor, prepared):
    layers, depths = prepared["layers"], prepared["depths"]
    dz = np.diff(depths)[None, None]
    rho_p = density_np(prediction[..., :layers], prediction[..., layers:])
    rho_t = density_np(target[..., :layers], target[..., layers:])
    grad_p = np.diff(rho_p, axis=-1) / dz
    grad_t = np.diff(rho_t, axis=-1) / dz
    curv_p = np.diff(grad_p, axis=-1) / prepared["curvature_dz"]
    curv_t = np.diff(grad_t, axis=-1) / prepared["curvature_dz"]
    excess = np.maximum(
        (np.abs(curv_p) - prepared["curvature_q95"])
        / prepared["curvature_scale"],
        0,
    )

    rho_mid_p = 0.5 * (rho_p[..., 1:] + rho_p[..., :-1])
    rho_mid_t = 0.5 * (rho_t[..., 1:] + rho_t[..., :-1])
    n2_p = GRAVITY * grad_p / np.maximum(rho_mid_p, 1.0)
    n2_t = GRAVITY * grad_t / np.maximum(rho_mid_t, 1.0)
    floor = np.asarray(prepared["n2_proxy_floor"])[None, None, :]
    ceiling = np.asarray(prepared["n2_proxy_ceiling"])[None, None, :]
    envelope_violation = (n2_p < floor) | (n2_p > ceiling)

    return {
        "Curvature_RMSE": rmse(curv_p, curv_t),
        "Curvature_Severity": float(np.mean(excess ** 2)),
        "Density_Inversion_Fraction": float(np.mean(grad_p < 0)),
        "N2Proxy_RMSE_s2": rmse(n2_p, n2_t),
        "N2Proxy_MAE_s2": mae(n2_p, n2_t),
        "StaticInstability_Rate": float(np.mean(n2_p < 0.0)),
        "Reference_StaticInstability_Rate": float(np.mean(n2_t < 0.0)),
        "StaticInstability_Rate_Gap": abs(float(
            np.mean(n2_p < 0.0) - np.mean(n2_t < 0.0)
        )),
        "N2Envelope_Violation_Fraction": float(np.mean(envelope_violation)),
    }

def depth_safe_fusion(current, fused, prepared):
    """Keep current-C2 upper temperature and fused deep temperature/salinity."""
    if current.shape != fused.shape:
        raise ValueError("Current and fused profiles must have identical shapes")
    layers = prepared["layers"]
    deep = prepared["depths"] > 150.0
    temp = current[..., :layers].copy()
    temp[..., deep] = fused[..., :layers][..., deep]
    return np.concatenate([temp, fused[..., layers:]], axis=-1)


class PhysicalLoss(nn.Module):
    """Optional physical fine-tuning loss using the same current OASP module."""

    def __init__(
        self, processor, prepared, weight, salt_gradient,
        acoustic_weight=0.0, stability_weight=0.0, physics_mode="none",
        loss_mode="global_huber", depth_balanced=False, sound_weight=0.0,
        stratification_weight=0.0, vsc_mode="legacy",
    ):
        super().__init__()
        if physics_mode not in ("none", "optimized"):
            raise ValueError(f"Unknown physics_mode={physics_mode!r}")
        self.processor, self.layers = processor, prepared["layers"]
        self.weight, self.salt_gradient = float(weight), float(salt_gradient)
        self.acoustic_weight = float(acoustic_weight)
        self.stability_weight = float(stability_weight)
        self.physics_mode = str(physics_mode)
        self.loss_mode = loss_mode
        self.depth_balanced = bool(depth_balanced)
        self.sound_weight = float(sound_weight)
        self.stratification_weight = float(stratification_weight)
        self.vsc_mode = str(vsc_mode)
        self.register_buffer("temp_depth_scale", torch.tensor(prepared["temp_depth_scale"], dtype=torch.float32))
        self.register_buffer("salt_depth_scale", torch.tensor(prepared["salt_depth_scale"], dtype=torch.float32))
        self.register_buffer("temp_global_scale", torch.tensor(prepared["temp_scale"], dtype=torch.float32))
        self.register_buffer("salt_global_scale", torch.tensor(prepared["salt_scale"], dtype=torch.float32))
        self.register_buffer("depths", torch.tensor(prepared["depths"], dtype=torch.float32))
        self.register_buffer("dz", torch.tensor(prepared["dz"], dtype=torch.float32))
        self.register_buffer("depth_loss_weight", torch.tensor(prepared["depth_loss_weight"], dtype=torch.float32))
        self.register_buffer("sound_speed_scale", torch.tensor(prepared["sound_speed_scale"], dtype=torch.float32))
        self.register_buffer("density_grad_scale", torch.tensor(prepared["density_grad_scale"], dtype=torch.float32))
        for key in (
            "opt_tt_offset", "opt_tt_slope", "opt_tt_anomaly_scale",
            "opt_acoustic_reliability", "opt_center_scale", "opt_change_scale",
            "opt_center_reliability", "opt_change_reliability",
            "n2_proxy_floor", "n2_proxy_ceiling", "n2_proxy_scale",
        ):
            self.register_buffer(key, torch.as_tensor(prepared[key], dtype=torch.float64))
        self.sec_theta = float(prepared["sec_theta"])
        for name in ("curvature_dz", "curvature_q95", "curvature_scale"):
            self.register_buffer(name, torch.as_tensor(prepared[name], dtype=torch.float32))
        for step in (1, 2, 4):
            self.register_buffer(
                f"temp_second_scale_{step}",
                torch.as_tensor(prepared[f"temp_second_scale_{step}"], dtype=torch.float32),
            )
            self.register_buffer(
                f"temp_second_q99_{step}",
                torch.as_tensor(prepared[f"temp_second_q99_{step}"], dtype=torch.float32),
            )

    def forward(self, z, target, anchor, context, history=None, future_tt=None):
        prediction = self.processor(z, anchor, context)
        temp, salt = prediction[..., :self.layers], prediction[..., self.layers:]
        target_t, target_s = target[..., :self.layers], target[..., self.layers:]
        temp_element = F.huber_loss(
            temp / self.temp_global_scale,
            target_t / self.temp_global_scale,
            reduction="none",
        )
        temp_huber = (
            (temp_element * self.depth_loss_weight).mean()
            if self.depth_balanced else temp_element.mean()
        )
        temp_mse = F.mse_loss(
            temp / self.temp_global_scale,
            target_t / self.temp_global_scale,
        )
        if self.loss_mode == "global_mse":
            loss = temp_mse
        elif self.loss_mode == "mixed_global":
            loss = 0.7 * temp_mse + 0.3 * temp_huber
        else:
            loss = temp_huber
        loss += 0.5 * F.huber_loss(
            salt / self.salt_global_scale,
            target_s / self.salt_global_scale,
        )

        # Legacy diagnostic/fine-tuning options retained, but OASP is the only
        # acoustic/stability module used by the current configuration.
        if self.sound_weight > 0:
            sound_prediction = sound_speed_torch(temp, salt, self.depths)
            sound_target = sound_speed_torch(target_t, target_s, self.depths)
            loss += self.sound_weight * F.huber_loss(
                (sound_prediction - sound_target) / self.sound_speed_scale,
                torch.zeros_like(sound_prediction),
            )
        if self.stratification_weight > 0:
            rho_prediction = density_torch(temp, salt)
            rho_target = density_torch(target_t, target_s)
            gradient_prediction = torch.diff(rho_prediction, dim=-1) / self.dz
            gradient_target = torch.diff(rho_target, dim=-1) / self.dz
            loss += self.stratification_weight * F.huber_loss(
                (gradient_prediction - gradient_target) / self.density_grad_scale,
                torch.zeros_like(gradient_prediction),
            )
        if self.weight > 0 and self.vsc_mode == "legacy":
            physics_salt = salt.detach() + self.salt_gradient * (salt - salt.detach())
            rho = density_torch(temp, physics_salt)
            gradient = torch.diff(rho, dim=-1) / torch.diff(self.depths)
            curvature = torch.diff(gradient, dim=-1) / self.curvature_dz
            excess = F.relu(
                (torch.abs(curvature) - self.curvature_q95) / self.curvature_scale
            )
            loss += self.weight * excess.square().mean()
        elif self.weight > 0 and self.vsc_mode == "multiscale":
            multiscale = 0.0
            for step in (1, 2, 4):
                second = vertical_second_torch(temp, self.depths, step)
                threshold = getattr(self, f"temp_second_q99_{step}")
                scale = getattr(self, f"temp_second_scale_{step}")
                excess = F.relu(torch.abs(second) - threshold) / scale
                multiscale = multiscale + excess.square().mean()
            loss += self.weight * multiscale / 3.0

        if self.physics_mode == "optimized" and self.acoustic_weight > 0 and future_tt is not None:
            c = sound_speed_torch(temp.double(), salt.double(), self.depths.double())
            physical_tt = self.sec_theta * torch.sum(
                self.dz.double() * (1.0 / c[..., :-1] + 1.0 / c[..., 1:]), dim=-1,
            )
            predicted_tt = self.opt_tt_offset + self.opt_tt_slope * physical_tt
            error = predicted_tt - future_tt.double()
            level = F.smooth_l1_loss(
                error / self.opt_tt_anomaly_scale, torch.zeros_like(error)
            )
            centered_error = error - error.mean(dim=1, keepdim=True)
            centered = F.smooth_l1_loss(
                centered_error / self.opt_center_scale, torch.zeros_like(centered_error)
            )
            change_error = error[:, 6:] - error[:, :-6]
            change = F.smooth_l1_loss(
                change_error / self.opt_change_scale, torch.zeros_like(change_error)
            )
            acoustic_loss = (
                0.25 * self.opt_acoustic_reliability * level
                + 0.50 * self.opt_center_reliability * centered
                + 0.25 * self.opt_change_reliability * change
            )
            loss = loss + self.acoustic_weight * acoustic_loss

        if self.physics_mode == "optimized" and self.stability_weight > 0:
            rho = density_torch(temp.double(), salt.double())
            density_gradient = (rho[..., 1:] - rho[..., :-1]) / self.dz.double()
            rho_mid = 0.5 * (rho[..., 1:] + rho[..., :-1])
            n2_proxy = GRAVITY * density_gradient / torch.clamp(rho_mid, min=1.0)
            lower = F.relu((self.n2_proxy_floor - n2_proxy) / self.n2_proxy_scale)
            upper = F.relu((n2_proxy - self.n2_proxy_ceiling) / self.n2_proxy_scale)
            stability_loss = F.smooth_l1_loss(lower, torch.zeros_like(lower))
            stability_loss = stability_loss + 0.25 * F.smooth_l1_loss(
                upper, torch.zeros_like(upper)
            )
            loss = loss + self.stability_weight * stability_loss
        return loss

def fine_tune(
    base_model, processor, prepared, args, device, seed,
    weight, validation_loader, spec,
):
    seed_all(seed)
    model, processor = copy.deepcopy(base_model).to(device), copy.deepcopy(processor).to(device)
    criterion = PhysicalLoss(
        processor, prepared, weight, args.salt_physics_gradient,
        acoustic_weight=spec.get("acoustic", 0.0),
        stability_weight=spec.get("stability", 0.0),
        physics_mode=spec.get("physics_mode", "none"),
        loss_mode=spec["loss_mode"],
        depth_balanced=spec.get("depth_balanced", False),
        sound_weight=spec.get("sound_weight", 0.0),
        stratification_weight=spec.get("stratification_weight", 0.0),
        vsc_mode=spec.get("vsc_mode", "legacy"),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate * 0.25, weight_decay=1e-2,
    )
    loader = prepared["loaders"]["train"]
    loader.generator.manual_seed(seed)
    best, best_score, best_epoch, bad = copy.deepcopy(model.state_dict()), math.inf, 0, 0
    for epoch in range(1, args.physical_epochs + 1):
        model.train()
        for history, context, _, target, anchor, future_tt in loader:
            history = history.to(device, non_blocking=True)
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            anchor = anchor.to(device, non_blocking=True)
            future_tt = (
                future_tt.to(device, non_blocking=True)
                if criterion.acoustic_weight > 0 else None
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                model(history, context), target, anchor, context,
                future_tt=future_tt,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        total = count = 0
        with torch.no_grad():
            for history, context, _, target, anchor, future_tt in validation_loader:
                history = history.to(device, non_blocking=True)
                context = context.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                anchor = anchor.to(device, non_blocking=True)
                future_tt = (
                    future_tt.to(device, non_blocking=True)
                    if criterion.acoustic_weight > 0 else None
                )
                value = criterion(
                    model(history, context), target, anchor, context,
                    future_tt=future_tt,
                )
                total += value.item() * len(history)
                count += len(history)
        score = total / max(count, 1)
        if score < best_score - 1e-8:
            best_score = score
            best, best_epoch, bad = copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= args.physical_patience:
                break
    model.load_state_dict(best)
    return model, processor, best_epoch






def smoke_test():
    rows = 600
    windows, window_meta = make_windows(
        np.zeros((rows, 7)), np.zeros((rows, 6)), np.zeros((rows, NT + NS)),
        np.zeros((rows, 76)), np.zeros((rows, 76)),
        pd.date_range("2025-01-01", periods=rows, freq="h"),
        300, 450, np.zeros(rows), return_meta=True,
    )
    model = NLinearEOF()
    assert model(torch.randn(2, SEQ, 7), torch.randn(2, 6)).shape == (2, HORIZON, NT + NS)
    lowrank = CausalLowRankNLinearEOF(rank=24)
    lowrank_output = lowrank(
        torch.randn(2, SEQ, 7), torch.randn(2, 10),
    )
    assert lowrank_output.shape == (2, HORIZON, NT + NS)
    assert sum(parameter.numel() for parameter in lowrank.parameters()) == 41904
    value = torch.randn(2, HORIZON, NT + NS)
    rts = Refiner("rts", np.full(NT + NS, 0.9), np.zeros(NT + NS), np.ones(NT + NS), np.ones(NT + NS))
    blts = DualRefiner(rts, copy.deepcopy(rts))
    assert blts(value).shape == value.shape
    horizon_rts = Refiner(
        "rts", np.full(NT + NS, 0.9), np.zeros(NT + NS),
        np.ones(NT + NS), np.ones((3, NT + NS)),
    )
    assert horizon_rts(value).shape == value.shape
    smoke_depths = np.linspace(0.5, 1515.0, 38)
    smoke_t = np.full((2, 3, 38), 10.0)
    smoke_s = np.full((2, 3, 38), 35.0)
    assert sound_speed_np(smoke_t, smoke_s, smoke_depths).shape == smoke_t.shape
    assert physical_travel_time_np(smoke_t, smoke_s, smoke_depths, 1.45).shape == (2, 3)
    torch_t = torch.full((2, 3, 38), 10.0)
    torch_s = torch.full((2, 3, 38), 35.0)
    torch_d = torch.linspace(0.5, 1515.0, 38)
    assert sound_speed_torch(torch_t, torch_s, torch_d).shape == torch_t.shape
    assert physical_travel_time_torch(torch_t, torch_s, torch_d, 1.45).shape == (2, 3)
    safe_prepared = {
        "layers": 38,
        "depths": smoke_depths,
        "dz": np.diff(smoke_depths),
        "curvature_dz": 0.5 * (
            np.diff(smoke_depths)[:-1] + np.diff(smoke_depths)[1:]
        ),
        "curvature_q95": np.ones(36),
        "curvature_scale": np.ones(36),
    }
    current_profile = np.concatenate([smoke_t, smoke_s], axis=-1)
    fused_profile = current_profile.copy()
    fused_profile[..., :38] += 0.1
    depth_profile = depth_safe_fusion(
        current_profile, fused_profile, safe_prepared,
    )
    assert np.array_equal(
        depth_profile[..., :38][..., smoke_depths <= 150],
        current_profile[..., :38][..., smoke_depths <= 150],
    )
    router_current = np.repeat(current_profile[:, :1], HORIZON, axis=1)
    router_depth = np.repeat(depth_profile[:, :1], HORIZON, axis=1)
    router_target = np.repeat(fused_profile[:, :1], HORIZON, axis=1)
    router_prepared = {
        "layers": 38, "depths": smoke_depths, "upper": np.array([0, 1, 2]),
    }
    router_alpha, _ = fit_profile_reliability_router(
        router_current, router_depth, router_target, router_prepared, True,
    )
    routed = apply_profile_reliability_router(
        router_current, router_depth, router_alpha, router_prepared,
    )
    assert routed.shape == router_current.shape
    assert np.all((router_alpha >= 0) & (router_alpha <= 1))
    hard_mask = route_mask(smoke_depths, 300.0, 0.0, 38)
    smooth_mask = route_mask(smoke_depths, 300.0, 100.0, 38)
    assert hard_mask.shape == smooth_mask.shape == (76,)
    assert np.all(hard_mask[38:] == 0) and np.all(smooth_mask[38:] == 0)
    assert np.all((smooth_mask[:38] >= 0) & (smooth_mask[:38] <= 1))
    assert all(len(windows[name][0]) > 0 for name in windows)
    assert window_meta["test"]["valid_ns"].shape[1] == HORIZON
    unique = np.random.default_rng(1).normal(size=(400, NT + NS))
    hourly = np.arange(400, dtype=np.int64) * np.int64(3_600_000_000_000)
    assert all(x.shape == (NT + NS,) for x in fit_dynamics_unique(unique, hourly))
    origins = np.arange(4, dtype=np.int64) * np.int64(6 * 3_600_000_000_000)
    valid = origins[:, None] + (
        np.arange(1, HORIZON + 1, dtype=np.int64)[None]
        * np.int64(3_600_000_000_000)
    )
    latent = np.random.default_rng(2).normal(size=(4, HORIZON, NT + NS)).astype(np.float32)
    variance = np.ones((HORIZON, NT + NS), dtype=np.float32)
    no_fusion = causal_vintage_fusion(latent, origins, valid, variance, 0.9, 0.0)
    fused = causal_vintage_fusion(latent, origins, valid, variance, 0.9, 1.0)
    assert np.array_equal(no_fusion, latent)
    assert np.array_equal(fused[..., NT:], latent[..., NT:])

    print("Smoke test passed")


def experiment_spec(
    name,
    depth_balanced=False,
    sound_weight=0.0,
    stratification_weight=0.0,
    acoustic_weight=0.0,
    stability_weight=0.0,
    physics_mode="none",
    vsc_mode="legacy",
):
    return {
        "name": name,
        "temp_latent": 0.0,
        "salt_latent": 0.03,
        "loss_mode": "global_huber",
        "acoustic": float(acoustic_weight),
        "stability": float(stability_weight),
        "physics_mode": str(physics_mode),
        "depth_balanced": depth_balanced,
        "sound_weight": sound_weight,
        "stratification_weight": stratification_weight,
        "vsc_mode": vsc_mode,
    }

def physical_ablation_specs(suite="compact"):
    compact = (
        experiment_spec("P0-SupervisedOnly", vsc_mode="none"),
        experiment_spec("P1-LegacyVSC", vsc_mode="legacy"),
        experiment_spec("P2-MultiscaleVSC", vsc_mode="multiscale"),
    )
    if suite == "compact":
        return compact
    acoustic = experiment_spec(
        "P5-Physics+OASP",
        depth_balanced=True,
        sound_weight=0.03,
        stratification_weight=0.02,
        acoustic_weight=DEFAULT_OPT_ACOUSTIC_WEIGHT,
        stability_weight=DEFAULT_STABILITY_WEIGHT,
        physics_mode="optimized",
    )
    return compact + (
        experiment_spec("P1-DepthBalanced", depth_balanced=True),
        experiment_spec(
            "P2-Depth+Sound", depth_balanced=True, sound_weight=0.03,
        ),
        experiment_spec(
            "P3-Depth+Stratification", depth_balanced=True,
            stratification_weight=0.02,
        ),
        experiment_spec(
            "P4-PhysicsInformed", depth_balanced=True, sound_weight=0.03,
            stratification_weight=0.02,
        ),
        acoustic,
    )


def fit_physics_seed(seed, prepared, args, device, anchor_val):
    """Train one shared base, then isolate each physical fine-tuning loss."""
    base_spec = experiment_spec("Base")
    base_model, base_epoch, base_seconds = train_base(
        seed, prepared, args, device, anchor_val, base_spec,
    )
    raw = {
        name: collect_latent(base_model, prepared["loaders"][name], device)
        for name in ("train", "val", "test")
    }
    fixed_processor, processor_meta = calibrate_processor(
        DualRefiner(Refiner(), Refiner()),
        raw, prepared, device, "temp", 2,
    )

    rows, models, selection_scores = [], {}, {}
    for spec in physical_ablation_specs(args.physics_suite):
        model, processor, physical_epoch = fine_tune(
            base_model, fixed_processor, prepared, args, device, seed,
            args.vsc_weight, prepared["loaders"]["val"], spec,
        )
        val_prediction, val_target, val_anchor, _ = predict_processed(
            model, prepared["loaders"]["val"], processor, device,
        )
        test_prediction, test_target, test_anchor, future_tt = predict_processed(
            model, prepared["loaders"]["test"], processor, device,
        )
        val_score = metrics(
            val_prediction, val_target, prepared, val_anchor,
        )
        test_score = metrics(
            test_prediction, test_target, prepared, test_anchor,
        )
        selection_score = (
            val_score["Temp_RMSE"] / anchor_val["Temp_RMSE"]
            + 0.2 * val_score["Salt_RMSE"] / anchor_val["Salt_RMSE"]
        )
        selection_scores[spec["name"]] = selection_score
        models[spec["name"]] = copy.deepcopy(model).cpu()
        row = {
            "Model": spec["name"],
            "Stage": "PhysicalLoss",
            "Seed": seed,
            "Selected_Physics": False,
            "Physics_Config": spec["name"],
            "Information_Set": "single_origin",
            "Parameters": sum(p.numel() for p in model.parameters()),
            "Base_Best_Epoch": base_epoch,
            "Physical_Best_Epoch": physical_epoch,
            "Base_Seconds": base_seconds,
            "Validation_Selection_Score": selection_score,
            "Validation_Temp_RMSE": val_score["Temp_RMSE"],
            "Validation_Salt_RMSE": val_score["Salt_RMSE"],
            "Depth_Balanced_Loss": spec["depth_balanced"],
            "Sound_Loss_Weight": spec["sound_weight"],
            "Stratification_Loss_Weight": spec["stratification_weight"],
            "Acoustic_Loss_Weight": spec["acoustic"],
            "Stability_Loss_Weight": spec.get("stability", 0.0),
            "Physics_Mode": spec.get("physics_mode", "none"),
            "VSC_Weight": args.vsc_weight,
            **processor_meta,
            **test_score,
            **physics_metrics(test_prediction, test_target, test_anchor, prepared),
            **acoustic_metrics(test_prediction, future_tt, prepared),
        }
        rows.append(row)
        print(
            f"  {spec['name']:<30} "
            f"ValTemp={val_score['Temp_RMSE']:.6f} "
            f"TestTemp={test_score['Temp_RMSE']:.6f}"
        )
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows, models, selection_scores


def run_temporal_ablation(seed, model, selected_physics, prepared, device,
                          vsc_weight):
    """Freeze the selected physical model and optimize only module 3."""
    model = copy.deepcopy(model).to(device)
    latent = {
        name: collect_latent(model, prepared["loaders"][name], device)
        for name in ("train", "val", "test")
    }
    for split in ("val", "test"):
        assert_overlap_consistency(
            latent[split][1], prepared["window_meta"][split]["valid_ns"],
        )

    identity = DualRefiner(Refiner(), Refiner())
    temp_identity, temp_meta = calibrate_processor(
        identity, latent, prepared, device, "temp", 2,
    )
    ts_identity, ts_meta = calibrate_processor(
        identity, latent, prepared, device, "ts", 1,
    )
    repeated_refiner, repeated_meta = select_full_blts(
        latent["train"][1], latent["val"], temp_identity, prepared, device,
    )
    unique_refiner, unique_meta = select_unique_blts(
        prepared, latent["val"], temp_identity, device,
    )

    fixed_mask = route_mask(
        prepared["depths"], 150.0, 0.0, prepared["layers"],
    )
    hard_identity = DepthRouteProcessor(
        temp_identity, ts_identity, fixed_mask, 1.0,
    ).to(device)
    repeated_hard = DepthRouteProcessor(
        processor_with_refiner(temp_identity, repeated_refiner, device),
        processor_with_refiner(ts_identity, repeated_refiner, device),
        fixed_mask, 1.0,
    ).to(device)

    val_z, _, val_target, val_anchor, val_context = latent["val"]
    test_z, _, test_target, test_anchor, test_context = latent["test"]
    current_val = process_z(
        repeated_hard, val_z, val_anchor, val_context, device,
    )
    current_test, loader_target, loader_anchor, future_tt = predict_processed(
        model, prepared["loaders"]["test"], repeated_hard, device,
    )
    if not np.allclose(loader_target, test_target) or not np.allclose(
        loader_anchor, test_anchor,
    ):
        raise AssertionError("Test loader and latent arrays are not aligned")

    unique_val_z = refine_np(unique_refiner, val_z)
    unique_test_z = refine_np(unique_refiner, test_z)
    direct_test = process_z(
        hard_identity, test_z, test_anchor, test_context, device,
    )
    deduplicated_test = process_z(
        hard_identity, unique_test_z, test_anchor, test_context, device,
    )
    val_meta, test_meta = (
        prepared["window_meta"]["val"], prepared["window_meta"]["test"],
    )

    old_variance, old_gamma, old_beta, old_meta = select_causal_fusion(
        unique_val_z, latent["val"], val_meta, hard_identity, prepared, device,
    )
    old_test_z = causal_vintage_fusion(
        unique_test_z, test_meta["origin_ns"], test_meta["valid_ns"],
        old_variance, old_gamma, old_beta,
    )
    old_candidate = process_z(
        hard_identity, old_test_z, test_anchor, test_context, device,
    )
    hard150 = depth_safe_fusion(current_test, old_candidate, prepared)

    variance, gamma, fusion_meta = select_fusion_gamma_v2(
        unique_val_z, latent["val"], val_meta, hard_identity, prepared, device,
    )
    v3 = int(0.75 * len(unique_val_z))
    fused_val_z = causal_vintage_fusion(
        unique_val_z[v3:], val_meta["origin_ns"][v3:],
        val_meta["valid_ns"][v3:], variance, gamma, 1.0,
    )
    fused_test_z = causal_vintage_fusion(
        unique_test_z, test_meta["origin_ns"], test_meta["valid_ns"],
        variance, gamma, 1.0,
    )
    full_val = process_z(
        hard_identity, fused_val_z, val_anchor[v3:], val_context[v3:], device,
    )
    full_test = process_z(
        hard_identity, fused_test_z, test_anchor, test_context, device,
    )

    horizon_alpha, horizon_meta = fit_profile_reliability_router(
        current_val[v3:], full_val, val_target[v3:], prepared,
        depth_aware=False,
    )
    horizon_prediction = apply_profile_reliability_router(
        current_test, full_test, horizon_alpha, prepared,
    )

    depth_alpha, depth_meta = fit_profile_reliability_router(
        current_val[v3:], full_val, val_target[v3:], prepared,
        depth_aware=True,
    )
    depth_prediction = apply_profile_reliability_router(
        current_test, full_test, depth_alpha, prepared,
    )

    mode_weight = eof_deep_reliability(prepared)
    gated_val_z = apply_eof_fusion_gate(
        unique_val_z[v3:], fused_val_z, mode_weight,
    )
    gated_test_z = apply_eof_fusion_gate(
        unique_test_z, fused_test_z, mode_weight,
    )
    gated_val = process_z(
        hard_identity, gated_val_z, val_anchor[v3:], val_context[v3:], device,
    )
    gated_test = process_z(
        hard_identity, gated_test_z, test_anchor, test_context, device,
    )
    gated_alpha, gated_meta = fit_profile_reliability_router(
        current_val[v3:], gated_val, val_target[v3:], prepared,
        depth_aware=True,
    )
    gated_prediction = apply_profile_reliability_router(
        current_test, gated_test, gated_alpha, prepared,
    )

    common = {
        "Stage": "TemporalFusion",
        "Seed": seed,
        "Selected_Physics": True,
        "Physics_Config": selected_physics,
        "Parameters": sum(p.numel() for p in model.parameters()),
        "VSC_Weight": vsc_weight,
        **{f"TempSUA_{k}": v for k, v in temp_meta.items()},
        **{f"TSSUA_{k}": v for k, v in ts_meta.items()},
    }
    variants = (
        ("C2-Direct", direct_test, "single_origin", {}),
        (
            "S0-RepeatedBLTS", current_test, "single_origin", repeated_meta,
        ),
        (
            "S1-DeduplicatedBLTS", deduplicated_test, "single_origin",
            unique_meta,
        ),
        (
            "S2-Hard150CausalFusion", hard150,
            "causal_rolling_multi_origin", {**unique_meta, **old_meta},
        ),
        (
            "T1-HorizonReliability", horizon_prediction,
            "causal_rolling_multi_origin",
            {**unique_meta, **fusion_meta, **horizon_meta},
        ),
        (
            "T2-DepthHorizonReliability", depth_prediction,
            "causal_rolling_multi_origin",
            {**unique_meta, **fusion_meta, **depth_meta},
        ),
        (
            "T3-EOFDeep+DepthHorizon", gated_prediction,
            "causal_rolling_multi_origin",
            {
                **unique_meta, **fusion_meta, **gated_meta,
                **{
                    f"EOF_Deep_Weight_T{k + 1}": float(value)
                    for k, value in enumerate(mode_weight)
                },
            },
        ),
    )

    rows = []
    for variant, prediction, information_set, meta in variants:
        score = metrics(prediction, test_target, prepared, test_anchor)
        row = {
            "Model": f"{selected_physics}+{variant}",
            "Variant": variant,
            "Information_Set": information_set,
            **common,
            **meta,
            **score,
            **physics_metrics(
                prediction, test_target, test_anchor, prepared,
            ),
            **acoustic_metrics(prediction, future_tt, prepared),
        }
        rows.append(row)
        print(
            f"  {variant:<32} Temp={score['Temp_RMSE']:.6f} "
            f"Salt={score['Salt_RMSE']:.6f}"
        )
    return rows


def single_seed_predictions(args, prepared, device, anchor_val, seed):
    """Train/evaluate PAST-NLinear-EOF and NLinear-EOF for one random seed."""
    seed = int(seed)
    if args.opt_acoustic_weight < 0:
        raise ValueError("opt-acoustic-weight must be non-negative")
    if args.stability_weight < 0:
        raise ValueError("stability-weight must be non-negative")

    # Baseline remains the original supervised-only P0 configuration.
    p0 = experiment_spec("P0-SupervisedOnly", vsc_mode="none")

    # Final PAST model: dominant supervised objective + current weak OASP.
    # OASP = optimized acoustic consistency + N2-like static stability.
    final_spec = experiment_spec(
        "P0+OASP",
        acoustic_weight=args.opt_acoustic_weight,
        stability_weight=args.stability_weight,
        physics_mode="optimized",
        vsc_mode="none",
    )

    final_args = copy.deepcopy(args)
    final_args.model_variant = "causal-lowrank"
    final_args.model_rank = 24
    final_args.persistence_prior = True
    final_model, final_epoch, final_seconds = train_base(
        seed, prepared, final_args, device, anchor_val, final_spec,
    )
    latent = {
        name: collect_latent(final_model, prepared["loaders"][name], device)
        for name in ("train", "val", "test")
    }
    identity = DualRefiner(Refiner(), Refiner())
    temp_processor, temp_meta = calibrate_processor(
        identity, latent, prepared, device, "temp", 2,
    )
    ts_processor, ts_meta = calibrate_processor(
        identity, latent, prepared, device, "ts", 1,
    )
    fixed_mask = route_mask(
        prepared["depths"], 150.0, 0.0, prepared["layers"],
    )
    processor = DepthRouteProcessor(
        temp_processor, ts_processor, fixed_mask, 1.0,
    ).to(device)
    unique_refiner, blts_meta = select_unique_blts(
        prepared, latent["val"], temp_processor, device,
    )

    final_profiles = {}
    for split in ("val", "test"):
        predicted_z, _, target, anchor, context = latent[split]
        refined_z = refine_np(unique_refiner, predicted_z)
        final_profiles[split] = {
            "prediction": process_z(
                processor, refined_z, anchor, context, device,
            ),
            "target": target,
            "anchor": anchor,
        }

    baseline_args = copy.deepcopy(args)
    baseline_args.model_variant = "legacy"
    baseline_args.persistence_prior = False
    baseline_model, baseline_epoch, baseline_seconds = train_base(
        seed, prepared, baseline_args, device, anchor_val, p0,
    )
    baseline_profiles = {}
    for split in ("val", "test"):
        predicted_z, _, target, anchor, _ = collect_latent(
            baseline_model, prepared["loaders"][split], device,
        )
        baseline_profiles[split] = {
            "prediction": decode_np(predicted_z, anchor, prepared),
            "target": target,
            "anchor": anchor,
        }

    metadata = {
        "Final_Model": FINAL_MODEL_NAME,
        "Final_Model_Full_Name": FINAL_MODEL_FULL_NAME,
        "Baseline_Model": BASELINE_MODEL_NAME,
        "Seed": seed,
        "Rank": 24,
        "Persistence": True,
        "Loss": " + ".join(
            ["P0-SupervisedOnly"]
            + (["OptimizedAcoustic"] if args.opt_acoustic_weight > 0 else [])
            + (["N2Stability"] if args.stability_weight > 0 else [])
        ),
        "Physics_Constraint": (
            PHYSICS_CONSTRAINT_NAME
            if (args.opt_acoustic_weight > 0 or args.stability_weight > 0)
            else "None"
        ),
        "Optimized_Acoustic_Weight": float(args.opt_acoustic_weight),
        "Stability_Weight": float(args.stability_weight),
        "Train_Opt_Acoustic_Correlation": float(
            prepared["opt_acoustic_correlation"]
        ),
        "Train_Opt_Acoustic_Reliability": float(
            prepared["opt_acoustic_reliability"]
        ),
        "Train_Centered_Reliability": float(
            prepared["opt_center_reliability"]
        ),
        "Train_6hChange_Reliability": float(
            prepared["opt_change_reliability"]
        ),
        "Temporal_Module": "S1-DeduplicatedBLTS",
        "Final_Best_Epoch": final_epoch,
        "Baseline_Best_Epoch": baseline_epoch,
        "Final_Training_Seconds": final_seconds,
        "Baseline_Training_Seconds": baseline_seconds,
        **{f"TempSUA_{key}": value for key, value in temp_meta.items()},
        **{f"TSSUA_{key}": value for key, value in ts_meta.items()},
        **{f"BLTS_{key}": value for key, value in blts_meta.items()},
    }
    return (
        final_model, baseline_model, processor, unique_refiner,
        final_profiles, baseline_profiles, metadata,
    )


def publication_modules():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError as error:
        raise RuntimeError(
            "Figure export requires matplotlib. Install it with: "
            "python3 -m pip install matplotlib"
        ) from error
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    return matplotlib, plt, mdates


def panel_label(ax, label):
    ax.text(
        -0.16, 1.04, label, transform=ax.transAxes, ha="left", va="bottom",
        fontsize=9, fontweight="bold",
    )


def export_figure(fig, base_path, dpi):
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for extension, options in (
        ("svg", {}), ("pdf", {}), ("tiff", {"dpi": dpi}),
        ("png", {"dpi": 300}),
    ):
        fig.savefig(
            base_path.with_suffix(f".{extension}"), bbox_inches="tight",
            facecolor="white", **options,
        )


def smooth_density(value, edges):
    density, _ = np.histogram(value, bins=edges, density=True)
    kernel = np.asarray([1, 2, 3, 2, 1], dtype=np.float64)
    kernel /= kernel.sum()
    density = np.convolve(density, kernel, mode="same")
    return 0.5 * (edges[:-1] + edges[1:]), density


def horizon_rmse_table(final_prediction, baseline_prediction, target, output_dir):
    """Export 1/6/12/24/36/48/60/72-h temperature AND salinity RMSE.

    Temperature is reported in degC and salinity in PSU.  The same test samples
    and exact forecast horizon are used for NLinear-EOF and PAST-NLinear-EOF.
    """
    horizons = np.asarray([1, 6, 12, 24, 36, 48, 60, 72], dtype=int)
    layers = target.shape[-1] // 2
    rows = []

    for horizon in horizons:
        index = horizon - 1

        baseline_temp = baseline_prediction[:, index, :layers]
        final_temp = final_prediction[:, index, :layers]
        true_temp = target[:, index, :layers]

        baseline_salt = baseline_prediction[:, index, layers:]
        final_salt = final_prediction[:, index, layers:]
        true_salt = target[:, index, layers:]

        rows.append({
            "Forecast_Horizon": f"{horizon} h",

            # Temperature RMSE (degC)
            "NLinear_EOF_Temp_RMSE_C": rmse(
                baseline_temp,
                true_temp,
            ),
            "PAST_NLinear_EOF_Temp_RMSE_C": rmse(
                final_temp,
                true_temp,
            ),

            # Salinity RMSE (PSU)
            "NLinear_EOF_Salinity_RMSE_PSU": rmse(
                baseline_salt,
                true_salt,
            ),
            "PAST_NLinear_EOF_Salinity_RMSE_PSU": rmse(
                final_salt,
                true_salt,
            ),
        })

    table = pd.DataFrame(rows)

    # English CSV.
    table.to_csv(
        output_dir / "horizon_rmse_comparison.csv",
        index=False,
    )

    # Chinese CSV for direct use in the manuscript.
    cn_table = table.rename(columns={
        "Forecast_Horizon": "预测步长",
        "NLinear_EOF_Temp_RMSE_C": "NLinear-EOF 温度RMSE(℃)",
        "PAST_NLinear_EOF_Temp_RMSE_C": "PAST-NLinear-EOF 温度RMSE(℃)",
        "NLinear_EOF_Salinity_RMSE_PSU": "NLinear-EOF 盐度RMSE(PSU)",
        "PAST_NLinear_EOF_Salinity_RMSE_PSU": "PAST-NLinear-EOF 盐度RMSE(PSU)",
    })
    cn_table.to_csv(
        output_dir / "horizon_rmse_comparison_cn.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Markdown table.
    markdown_lines = [
        "| Forecast horizon | NLinear-EOF Temp RMSE (°C) | "
        "PAST-NLinear-EOF Temp RMSE (°C) | "
        "NLinear-EOF Salinity RMSE (PSU) | "
        "PAST-NLinear-EOF Salinity RMSE (PSU) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown_lines.append(
            f"| {row['Forecast_Horizon']} | "
            f"{row['NLinear_EOF_Temp_RMSE_C']:.4f} | "
            f"{row['PAST_NLinear_EOF_Temp_RMSE_C']:.4f} | "
            f"{row['NLinear_EOF_Salinity_RMSE_PSU']:.4f} | "
            f"{row['PAST_NLinear_EOF_Salinity_RMSE_PSU']:.4f} |"
        )

    markdown = "\n".join(markdown_lines)
    (output_dir / "horizon_rmse_comparison.md").write_text(
        markdown + "\n",
        encoding="utf-8",
    )

    # Excel: English and Chinese tables in separate sheets.
    try:
        with pd.ExcelWriter(
            output_dir / "horizon_rmse_comparison.xlsx",
        ) as writer:
            table.to_excel(
                writer,
                sheet_name="English",
                index=False,
            )
            cn_table.to_excel(
                writer,
                sheet_name="中文",
                index=False,
            )
    except Exception as error:
        print(f"XLSX export skipped: {error}")

    return table



def figure8(final_prediction, baseline_prediction, target, anchor, depths,
            valid_ns, output_dir, dpi):
    _, plt, _ = publication_modules()
    from matplotlib.patches import Patch

    layers = len(depths)
    final_t = final_prediction[..., :layers]
    final_s = final_prediction[..., layers:]
    base_t = baseline_prediction[..., :layers]
    base_s = baseline_prediction[..., layers:]
    true_t = target[..., :layers]
    true_s = target[..., layers:]
    baseline_color = "#42A5E8"
    final_color = "#D7301F"
    truth_color = "#23374D"
    region_colors = {
        "Mixed Layer": "#EAF4FB",
        "Thermocline": "#FBE9E7",
        "Deep Ocean": "#F2F2F2",
    }

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2))
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()
    for ax in axes.ravel():
        ax.grid(True, color="#AEB6BF", linestyle="--", linewidth=0.65, alpha=0.45)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#8A8A8A")
            spine.set_linewidth(0.8)

    temp_depth = np.sqrt(np.mean((final_t - true_t) ** 2, axis=(0, 1)))
    base_temp_depth = np.sqrt(np.mean((base_t - true_t) ** 2, axis=(0, 1)))
    salt_depth = np.sqrt(np.mean((final_s - true_s) ** 2, axis=(0, 1)))
    base_salt_depth = np.sqrt(np.mean((base_s - true_s) ** 2, axis=(0, 1)))

    for ax in (ax_a, ax_b):
        ax.axhspan(0, 50, color=region_colors["Mixed Layer"], alpha=0.70, zorder=0)
        ax.axhspan(50, 150, color=region_colors["Thermocline"], alpha=0.70, zorder=0)
        ax.axhspan(150, float(np.max(depths)), color=region_colors["Deep Ocean"],
                   alpha=0.52, zorder=0)
        ax.set_ylim(float(np.max(depths)) + 35.0, -35.0)
        ax.set_ylabel("Depth (m)")

    baseline_line_a, = ax_a.plot(
        base_temp_depth, depths, color=baseline_color, linestyle="--",
        marker="s", markersize=3.5, markevery=2, linewidth=1.35,
        label=f"{BASELINE_MODEL_NAME} (Baseline)",
    )
    final_line_a, = ax_a.plot(
        temp_depth, depths, color=final_color, linestyle="-",
        marker="o", markersize=3.2, markevery=2, linewidth=1.55,
        label="Final Model",
    )
    ax_a.set_xlabel("RMSE (°C)")
    ax_a.set_title("(a) Vertical Profile: Temperature RMSE", pad=6)
    region_handles = [
        Patch(facecolor=region_colors[name], edgecolor="none", label=name)
        for name in ("Mixed Layer", "Thermocline", "Deep Ocean")
    ]
    ax_a.legend(
        handles=[baseline_line_a, final_line_a, *region_handles],
        loc="lower right", frameon=True, framealpha=0.88, fontsize=7,
    )

    ax_b.plot(
        base_salt_depth, depths, color=baseline_color, linestyle="--",
        marker="s", markersize=3.5, markevery=2, linewidth=1.35,
        label=f"{BASELINE_MODEL_NAME} (Baseline)",
    )
    ax_b.plot(
        salt_depth, depths, color=final_color, linestyle="-",
        marker="o", markersize=3.2, markevery=2, linewidth=1.55,
        label="Final Model",
    )
    ax_b.set_xlabel("RMSE")
    ax_b.set_title("(b) Vertical Profile: Salinity RMSE", pad=6)

    hours = np.arange(1, HORIZON + 1)
    final_mse_h = np.mean((final_t - true_t) ** 2, axis=(0, 2))
    base_mse_h = np.mean((base_t - true_t) ** 2, axis=(0, 2))
    final_pointwise = np.sqrt(final_mse_h)
    base_pointwise = np.sqrt(base_mse_h)
    final_cumulative = np.sqrt(np.cumsum(final_mse_h) / hours)
    base_cumulative = np.sqrt(np.cumsum(base_mse_h) / hours)
    ax_c.plot(
        hours, base_cumulative, color=baseline_color, linestyle="--",
        linewidth=1.35, label=f"{BASELINE_MODEL_NAME} (Baseline)",
    )
    ax_c.plot(
        hours, final_cumulative, color=final_color, linestyle="-",
        linewidth=1.55, label="Final Model",
    )
    ax_c.fill_between(
        hours, 0.0, final_cumulative, color=final_color, alpha=0.075,
        linewidth=0,
    )
    ax_c.set_xlabel("Forecast Horizon (Hours)")
    ax_c.set_ylabel("Cumulative Temp. RMSE (°C)")
    ax_c.set_title("(c) Temporal Error Accumulation", pad=6)
    ax_c.set_xlim(1, 72)
    ax_c.set_xticks([1, 12, 24, 36, 48, 60, 72])
    ax_c.legend(loc="best", frameon=True, framealpha=0.88, fontsize=7)

    flat_t = true_t.reshape(-1)
    flat_s = true_s.reshape(-1)
    flat_bt = base_t.reshape(-1)
    flat_bs = base_s.reshape(-1)
    flat_ft = final_t.reshape(-1)
    flat_fs = final_s.reshape(-1)
    rng = np.random.default_rng(20250815)
    sample_count = min(5000, flat_t.size)
    sample = rng.choice(flat_t.size, size=sample_count, replace=False)
    ax_d.scatter(
        flat_bt[sample], flat_bs[sample], s=7, alpha=0.28,
        color=baseline_color, edgecolors="none", rasterized=True,
        label=BASELINE_MODEL_NAME,
    )
    ax_d.scatter(
        flat_t[sample], flat_s[sample], s=7, alpha=0.48,
        color=truth_color, edgecolors="none", rasterized=True,
        label="Ground Truth",
    )
    ax_d.scatter(
        flat_ft[sample], flat_fs[sample], s=7, alpha=0.40,
        color=final_color, edgecolors="none", rasterized=True,
        label="Final Model",
    )
    ax_d.set_xlabel("Temperature (°C)")
    ax_d.set_ylabel("Salinity")
    ax_d.set_title("(d) Global T–S Physical Consistency", pad=6)
    ax_d.legend(loc="best", markerscale=1.4, frameon=True,
                framealpha=0.88, fontsize=7)

    # (e) Prediction-error distribution at the thermocline-near depth used
    # in the requested paper figure.  This panel compares actual forecast
    # errors, not target/anchor variability.
    index_100 = int(np.argmin(np.abs(depths - 100.0)))
    baseline_error_e = (
        base_t[..., index_100] - true_t[..., index_100]
    ).reshape(-1)
    final_error_e = (
        final_t[..., index_100] - true_t[..., index_100]
    ).reshape(-1)

    combined_error_e = np.concatenate([
        baseline_error_e,
        final_error_e,
    ])
    finite_error_e = combined_error_e[np.isfinite(combined_error_e)]
    if len(finite_error_e):
        low, high = np.quantile(
            finite_error_e,
            [0.0025, 0.9975],
        )
        if high <= low:
            pad = max(float(np.std(finite_error_e)), 0.05)
            low, high = float(low - pad), float(high + pad)
    else:
        low, high = -1.0, 1.0

    bins = np.linspace(low, high, 55)

    ax_e.hist(
        baseline_error_e,
        bins=bins,
        density=True,
        color=baseline_color,
        alpha=0.58,
        edgecolor="none",
        label=f"{BASELINE_MODEL_NAME} Error",
    )
    ax_e.hist(
        final_error_e,
        bins=bins,
        density=True,
        color=final_color,
        alpha=0.58,
        edgecolor="none",
        label=f"{FINAL_MODEL_NAME} Error",
    )

    # Colored dashed lines mark the mean signed prediction error of each model.
    ax_e.axvline(
        float(np.mean(baseline_error_e)),
        color=baseline_color,
        linestyle="--",
        linewidth=1.0,
    )
    ax_e.axvline(
        float(np.mean(final_error_e)),
        color=final_color,
        linestyle="--",
        linewidth=1.0,
    )
    ax_e.axvline(
        0.0,
        color="#555555",
        linestyle=":",
        linewidth=0.85,
        alpha=0.8,
    )

    ax_e.set_xlabel("Prediction Error (°C)")
    ax_e.set_ylabel("Density")
    ax_e.set_title(
        f"(e) Prediction Error Distribution (~{depths[index_100]:.0f} m)",
        pad=6,
    )
    ax_e.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.88,
        fontsize=7,
    )

    index_150 = int(np.argmin(np.abs(depths - 150.0)))

    # Select a genuinely dynamic but non-cherry-picked case for visualization.
    # Selection uses ONLY the observed 72-h temperature variability, never model
    # error/skill.  The case closest to the 75th percentile of observed standard
    # deviation is used so Figure 8(f) shows meaningful temporal evolution while
    # remaining representative rather than choosing the easiest/best forecast.
    observed_variability = np.std(
        true_t[..., index_150],
        axis=1,
    )
    variability_target = float(np.quantile(
        observed_variability,
        0.75,
    ))
    median_index = int(np.argmin(np.abs(
        observed_variability - variability_target
    )))

    truth_curve_f = true_t[median_index, :, index_150]
    baseline_curve_f = base_t[median_index, :, index_150]
    final_curve_f = final_t[median_index, :, index_150]

    ax_f.plot(
        hours, truth_curve_f, color=truth_color,
        linewidth=1.65, label="Ground Truth",
    )
    ax_f.plot(
        hours, baseline_curve_f, color=baseline_color,
        linestyle="--", marker="s", markersize=3.0, markevery=4,
        linewidth=1.2, label=BASELINE_MODEL_NAME,
    )
    ax_f.plot(
        hours, final_curve_f, color=final_color,
        linestyle="-", marker="o", markersize=3.0, markevery=4,
        linewidth=1.45, label=FINAL_MODEL_NAME,
    )

    # Add a small robust vertical margin.  The limits still include all three
    # trajectories; no prediction is clipped to make the proposed model look
    # better.
    finite_f = np.concatenate([
        truth_curve_f,
        baseline_curve_f,
        final_curve_f,
    ])
    finite_f = finite_f[np.isfinite(finite_f)]
    if len(finite_f):
        y_low = float(np.min(finite_f))
        y_high = float(np.max(finite_f))
        y_pad = max(0.02, 0.04 * max(y_high - y_low, 0.1))
        ax_f.set_ylim(y_low - y_pad, y_high + y_pad)

    ax_f.set_xlabel("Forecast Horizon (Hours)")
    ax_f.set_ylabel("Temperature (°C)")
    ax_f.set_title(
        f"(f) 72-h Evolution (Dynamic Sample #{median_index}, "
        f"~{depths[index_150]:.0f} m)", pad=6,
    )
    ax_f.set_xlim(1, 72)
    ax_f.set_xticks([1, 12, 24, 36, 48, 60, 72])
    ax_f.legend(loc="best", frameon=True, framealpha=0.88, fontsize=7)

    fig.subplots_adjust(
        left=0.065, right=0.99, bottom=0.075, top=0.965,
        wspace=0.23, hspace=0.27,
    )
    export_figure(fig, output_dir / "Figure8_final_model_diagnostics", dpi)
    plt.close(fig)

    pd.DataFrame({
        "Depth_m": depths,
        "NLinear_Temp_RMSE_C": base_temp_depth,
        "Final_Temp_RMSE_C": temp_depth,
        "NLinear_Salt_RMSE": base_salt_depth,
        "Final_Salt_RMSE": salt_depth,
    }).to_csv(output_dir / "Figure8ab_source_data.csv", index=False)
    pd.DataFrame({
        "Horizon_h": hours,
        "NLinear_Pointwise_Temp_RMSE_C": base_pointwise,
        "Final_Pointwise_Temp_RMSE_C": final_pointwise,
        "NLinear_Cumulative_Temp_RMSE_C": base_cumulative,
        "Final_Cumulative_Temp_RMSE_C": final_cumulative,
    }).to_csv(output_dir / "Figure8c_source_data.csv", index=False)
    pd.DataFrame({
        "Observed_Temperature_C": flat_t[sample],
        "Observed_Salinity": flat_s[sample],
        "NLinear_Temperature_C": flat_bt[sample],
        "NLinear_Salinity": flat_bs[sample],
        "Final_Temperature_C": flat_ft[sample],
        "Final_Salinity": flat_fs[sample],
    }).to_csv(output_dir / "Figure8d_source_data.csv", index=False)
    pd.DataFrame({
        "NLinear_Prediction_Error_C": baseline_error_e,
        "PAST_NLinear_EOF_Prediction_Error_C": final_error_e,
    }).to_csv(output_dir / "Figure8e_source_data.csv", index=False)
    origin_text = (
        pd.to_datetime(valid_ns[median_index, 0], unit="ns")
        - pd.Timedelta(hours=1)
    ).strftime("%Y-%m-%d %H:%M:%S")
    pd.DataFrame({
        "Origin_Time": [origin_text] * HORIZON,
        "Horizon_h": hours,
        "Observed_Temperature_C": truth_curve_f,
        "NLinear_Temperature_C": baseline_curve_f,
        "PAST_NLinear_EOF_Temperature_C": final_curve_f,
        "Observed_72h_Std_C": [observed_variability[median_index]] * HORIZON,
        "Selection_75pct_Std_C": [variability_target] * HORIZON,
    }).to_csv(output_dir / "Figure8f_source_data.csv", index=False)
    return median_index

def figure9(final_prediction, target, depths, valid_ns, output_dir, dpi):
    _, plt, mdates = publication_modules()
    layers = len(depths)
    horizon_index = 23
    times = pd.to_datetime(valid_ns[:, horizon_index], unit="ns")
    truth = target[:, horizon_index, :layers]
    error = np.abs(final_prediction[:, horizon_index, :layers] - truth)
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(7.2, 5.3), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.24},
    )
    temp_mesh = ax_a.pcolormesh(
        times, depths, truth.T, shading="auto", cmap="magma",
        vmin=float(np.min(truth)), vmax=float(np.max(truth)),
    )
    error_cap = max(float(np.quantile(error, 0.99)), 1e-6)
    error_mesh = ax_b.pcolormesh(
        times, depths, error.T, shading="auto", cmap="YlOrRd",
        vmin=0.0, vmax=error_cap,
    )
    for ax in (ax_a, ax_b):
        ax.axhline(150, color="white", lw=0.9, ls="--")
        ax.text(0.995, 0.90, "150 m", transform=ax.transAxes, ha="right",
                color="white", fontsize=6, fontweight="bold")
        ax.set_ylim(float(np.max(depths)), 0.0)
        ax.set_ylabel("Depth (m)")
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax_a.set_title("Observed ocean temperature field")
    ax_b.set_title("24-h forecast absolute error")
    ax_b.set_xlabel("Valid time")
    ax_b.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for label in ax_b.get_xticklabels():
        label.set_rotation(25)
        label.set_ha("right")
    panel_label(ax_a, "a")
    panel_label(ax_b, "b")
    cbar_a = fig.colorbar(temp_mesh, ax=ax_a, pad=0.015, fraction=0.025)
    cbar_a.set_label("Temperature (°C)")
    cbar_b = fig.colorbar(error_mesh, ax=ax_b, pad=0.015, fraction=0.025)
    cbar_b.set_label("Absolute error (°C)")
    ax_b.text(
        0.01, 0.04, "Error color scale capped at the 99th percentile",
        transform=ax_b.transAxes, color="white", fontsize=5.5,
    )
    fig.subplots_adjust(left=0.09, right=0.93, bottom=0.13, top=0.95)
    export_figure(fig, output_dir / "Figure9_temperature_and_error_fields", dpi)
    plt.close(fig)

    time_text = times.strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for time_index, time_value in enumerate(time_text):
        for depth_index, depth in enumerate(depths):
            rows.append({
                "Valid_Time": time_value,
                "Depth_m": depth,
                "Observed_Temperature_C": truth[time_index, depth_index],
                "Final_Absolute_Error_C": error[time_index, depth_index],
            })
    pd.DataFrame(rows).to_csv(
        output_dir / "Figure9_source_data.csv", index=False,
    )


def write_figure_captions(output_dir, representative_seed=None):
    seed_note = (
        f" The displayed profiles use representative seed {representative_seed}, "
        "selected using validation performance only."
        if representative_seed is not None
        else ""
    )
    caption = f"""Figure 8 | Depth-, horizon- and physics-resolved evaluation of OASP-constrained PAST-NLinear-EOF.{seed_note} a, Depth-resolved temperature RMSE for {BASELINE_MODEL_NAME} and PAST-NLinear-EOF; background bands denote the mixed layer (0–50 m), thermocline (50–150 m) and deep ocean (>150 m). b, Corresponding salinity RMSE profiles. c, Cumulative temperature RMSE from forecast hour 1 through each displayed horizon. d, Joint temperature–salinity distributions for observations, {BASELINE_MODEL_NAME} and PAST-NLinear-EOF using a fixed random subsample for visualization. e, Prediction-error probability-density distributions for NLinear-EOF and PAST-NLinear-EOF near 100 m; colored dashed lines denote mean signed errors. f, Observed, baseline and PAST-NLinear-EOF temperature trajectories near 150 m for a dynamic test case selected from the 75th percentile of observed 72-h temperature variability; sample selection is independent of model error.

Figure 9 | Test-period temperature structure and 24-h forecast error.{seed_note} a, Observed temperature across the 38 depth levels at the valid times of the 24-h forecasts. b, Absolute error of PAST-NLinear-EOF at the same valid times. The dashed line marks 150 m. For readability, the error color scale is capped at its 99th percentile; uncapped numerical values are provided in the source-data file.
"""
    (output_dir / "figure_captions_en.txt").write_text(caption, encoding="utf-8")


def _mean_std_summary(frame, group_column, id_columns):
    """Return wide mean/std summary for every numeric metric column."""
    numeric_columns = [
        column
        for column in frame.columns
        if column not in set(id_columns)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    rows = []
    for group_value, group in frame.groupby(group_column, sort=False):
        row = {group_column: group_value, "N_Seeds": int(group["Seed"].nunique())}
        for column in numeric_columns:
            if column == "Seed":
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if len(values) == 0:
                continue
            row[f"{column}_Mean"] = float(values.mean())
            row[f"{column}_Std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_seed_metrics(per_seed_table, output_dir):
    """Save per-seed metrics and mean±std summaries."""
    per_seed_table = per_seed_table.copy()
    per_seed_table.to_csv(
        output_dir / "final_model_vs_nlinear_metrics_per_seed.csv",
        index=False,
    )
    summary = _mean_std_summary(
        per_seed_table,
        group_column="Model",
        id_columns=("Seed", "Model"),
    )
    summary.to_csv(
        output_dir / "final_model_vs_nlinear_metrics.csv",
        index=False,
    )
    try:
        with pd.ExcelWriter(
            output_dir / "final_model_vs_nlinear_metrics.xlsx",
        ) as writer:
            per_seed_table.to_excel(writer, sheet_name="PerSeed", index=False)
            summary.to_excel(writer, sheet_name="MeanStd", index=False)
    except Exception as error:
        print(f"Metrics XLSX export skipped: {error}")
    return summary


def summarize_horizon_tables(seed_tables, output_dir):
    """Aggregate per-seed horizon RMSE into mean±std publication tables."""
    combined = pd.concat(seed_tables, ignore_index=True)
    horizon_order = [f"{h} h" for h in (1, 6, 12, 24, 36, 48, 60, 72)]
    metric_columns = [
        "NLinear_EOF_Temp_RMSE_C",
        "PAST_NLinear_EOF_Temp_RMSE_C",
        "NLinear_EOF_Salinity_RMSE_PSU",
        "PAST_NLinear_EOF_Salinity_RMSE_PSU",
    ]

    rows = []
    for horizon in horizon_order:
        group = combined.loc[combined["Forecast_Horizon"] == horizon]
        row = {
            "Forecast_Horizon": horizon,
            "N_Seeds": int(group["Seed"].nunique()),
        }
        for column in metric_columns:
            values = group[column].astype(float)
            row[f"{column}_Mean"] = float(values.mean())
            row[f"{column}_Std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    combined.to_csv(
        output_dir / "horizon_rmse_comparison_per_seed.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "horizon_rmse_comparison.csv",
        index=False,
    )

    cn = summary.rename(columns={
        "Forecast_Horizon": "预测步长",
        "N_Seeds": "种子数",
        "NLinear_EOF_Temp_RMSE_C_Mean": "NLinear-EOF 温度RMSE均值(℃)",
        "NLinear_EOF_Temp_RMSE_C_Std": "NLinear-EOF 温度RMSE标准差(℃)",
        "PAST_NLinear_EOF_Temp_RMSE_C_Mean": "PAST-NLinear-EOF 温度RMSE均值(℃)",
        "PAST_NLinear_EOF_Temp_RMSE_C_Std": "PAST-NLinear-EOF 温度RMSE标准差(℃)",
        "NLinear_EOF_Salinity_RMSE_PSU_Mean": "NLinear-EOF 盐度RMSE均值(PSU)",
        "NLinear_EOF_Salinity_RMSE_PSU_Std": "NLinear-EOF 盐度RMSE标准差(PSU)",
        "PAST_NLinear_EOF_Salinity_RMSE_PSU_Mean": "PAST-NLinear-EOF 盐度RMSE均值(PSU)",
        "PAST_NLinear_EOF_Salinity_RMSE_PSU_Std": "PAST-NLinear-EOF 盐度RMSE标准差(PSU)",
    })
    cn.to_csv(
        output_dir / "horizon_rmse_comparison_cn.csv",
        index=False,
        encoding="utf-8-sig",
    )

    md = [
        "| Forecast horizon | NLinear Temp RMSE | PAST Temp RMSE | "
        "NLinear Salinity RMSE | PAST Salinity RMSE |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['Forecast_Horizon']} | "
            f"{row['NLinear_EOF_Temp_RMSE_C_Mean']:.4f} ± "
            f"{row['NLinear_EOF_Temp_RMSE_C_Std']:.4f} | "
            f"{row['PAST_NLinear_EOF_Temp_RMSE_C_Mean']:.4f} ± "
            f"{row['PAST_NLinear_EOF_Temp_RMSE_C_Std']:.4f} | "
            f"{row['NLinear_EOF_Salinity_RMSE_PSU_Mean']:.4f} ± "
            f"{row['NLinear_EOF_Salinity_RMSE_PSU_Std']:.4f} | "
            f"{row['PAST_NLinear_EOF_Salinity_RMSE_PSU_Mean']:.4f} ± "
            f"{row['PAST_NLinear_EOF_Salinity_RMSE_PSU_Std']:.4f} |"
        )
    (output_dir / "horizon_rmse_comparison.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )

    try:
        with pd.ExcelWriter(
            output_dir / "horizon_rmse_comparison.xlsx",
        ) as writer:
            combined.to_excel(writer, sheet_name="PerSeed", index=False)
            summary.to_excel(writer, sheet_name="MeanStd", index=False)
            cn.to_excel(writer, sheet_name="中文", index=False)
    except Exception as error:
        print(f"Horizon XLSX export skipped: {error}")

    return combined, summary


def choose_representative_seed(validation_rows, requested_seed=None):
    """Choose a figure seed using validation T/S RMSE only."""
    table = pd.DataFrame(validation_rows).copy()
    seeds = table["Seed"].astype(int).tolist()

    if requested_seed is not None:
        requested_seed = int(requested_seed)
        if requested_seed not in seeds:
            raise ValueError(
                f"--figure-seed={requested_seed} is not present in --seeds={seeds}"
            )
        table["Representative_Distance"] = np.nan
        table["Selected"] = table["Seed"].astype(int) == requested_seed
        return requested_seed, table

    temp_mean = max(float(table["Val_Temp_RMSE"].mean()), 1e-12)
    salt_mean = max(float(table["Val_Salt_RMSE"].mean()), 1e-12)
    table["Representative_Distance"] = np.sqrt(
        ((table["Val_Temp_RMSE"] - temp_mean) / temp_mean) ** 2
        + ((table["Val_Salt_RMSE"] - salt_mean) / salt_mean) ** 2
    )
    selected_index = int(table["Representative_Distance"].idxmin())
    selected_seed = int(table.loc[selected_index, "Seed"])
    table["Selected"] = table["Seed"].astype(int) == selected_seed
    return selected_seed, table


def main():
    args = arguments()
    if args.smoke_test:
        smoke_test()
        return

    publication_modules()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(seed) for seed in args.seeds]
    if len(seeds) == 0:
        raise ValueError("--seeds must contain at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds contains duplicates: {seeds}")
    if len(seeds) != 3:
        print(
            f"Warning: publication setting uses three seeds; received {len(seeds)}: "
            f"{seeds}"
        )

    # Plot-only uses the already saved representative-seed prediction file.
    # It never overwrites the root three-seed mean±std tables.
    if args.plot_only:
        prediction_file = args.output_dir / "final_test_predictions.npz"
        if not prediction_file.exists():
            raise FileNotFoundError(
                f"Plot-only input not found: {prediction_file}"
            )
        with np.load(prediction_file) as saved:
            final_prediction = saved["final_prediction"]
            baseline_prediction = saved["nlinear_prediction"]
            target = saved["target"]
            anchor = saved["anchor"]
            depths = saved["depths"]
            valid_ns = saved["valid_ns"]
            representative_seed = (
                int(saved["representative_seed"])
                if "representative_seed" in saved.files
                else None
            )

        if representative_seed is not None:
            seed_dir = args.output_dir / f"seed_{representative_seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            horizon_rmse_table(
                final_prediction,
                baseline_prediction,
                target,
                seed_dir,
            )

        representative_index = figure8(
            final_prediction, baseline_prediction, target, anchor, depths,
            valid_ns, args.output_dir, args.figure_dpi,
        )
        figure9(
            final_prediction, target, depths, valid_ns,
            args.output_dir, args.figure_dpi,
        )
        write_figure_captions(args.output_dir, representative_seed)
        print(
            f"Plot-only export complete; representative seed={representative_seed}; "
            f"Figure 8f window={representative_index}; outputs={args.output_dir}"
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocessing is deterministic; only the training DataLoader shuffle order
    # depends on the seed. train_base() reseeds the generator for every run.
    prepared = prepare(args, seeds[0])
    split_sizes = {
        name: len(prepared["loaders"][name].dataset)
        for name in ("train", "val", "test")
    }

    dummy = NLinearEOF().to(device)
    _, _, val_target, val_anchor, _ = collect_latent(
        dummy, prepared["loaders"]["val"], device,
    )
    anchor_val = metrics(val_anchor, val_target, prepared, val_anchor)
    del dummy

    future_tt = collect_future_tt(prepared["loaders"]["test"])
    valid_ns = prepared["window_meta"]["test"]["valid_ns"]
    depths = prepared["depths"]

    print(
        f"Final model: {FINAL_MODEL_NAME}\n"
        f"Full name: {FINAL_MODEL_FULL_NAME}\n"
        f"Physics constraint: {PHYSICS_CONSTRAINT_NAME}\n"
        f"Optimized acoustic weight: {args.opt_acoustic_weight:g}; "
        f"stability weight: {args.stability_weight:g}\n"
        f"Seeds={seeds}; device={device}; splits={split_sizes}"
    )

    per_seed_metric_rows = []
    per_seed_metadata = []
    validation_rows = []
    horizon_seed_tables = []
    seed_predictions = {}

    reference_target = None
    reference_anchor = None

    for run_index, seed in enumerate(seeds, start=1):
        print(
            f"\n{'=' * 78}\n"
            f"THREE-SEED RUN {run_index}/{len(seeds)} | seed={seed}\n"
            f"{'=' * 78}"
        )

        (
            final_model, baseline_model, processor, unique_refiner,
            final_profiles, baseline_profiles, metadata,
        ) = single_seed_predictions(
            args, prepared, device, anchor_val, seed,
        )

        final_test = final_profiles["test"]
        baseline_test = baseline_profiles["test"]
        final_val = final_profiles["val"]

        for key in ("target", "anchor"):
            if not np.allclose(final_test[key], baseline_test[key]):
                raise AssertionError(
                    f"Baseline/final arrays differ for seed={seed}, key={key}"
                )

        target = final_test["target"]
        anchor = final_test["anchor"]
        final_prediction = final_test["prediction"]
        baseline_prediction = baseline_test["prediction"]

        if reference_target is None:
            reference_target = target
            reference_anchor = anchor
        else:
            if not np.allclose(reference_target, target):
                raise AssertionError(
                    f"Test target changed across seeds at seed={seed}"
                )
            if not np.allclose(reference_anchor, anchor):
                raise AssertionError(
                    f"Test anchor changed across seeds at seed={seed}"
                )

        # Primary metrics.
        final_metric = metrics(
            final_prediction, target, prepared, anchor,
        )
        baseline_metric = metrics(
            baseline_prediction, target, prepared, anchor,
        )

        # Optimized-physics diagnostics.
        final_acoustic = acoustic_metrics(
            final_prediction, future_tt, prepared,
        )
        baseline_acoustic = acoustic_metrics(
            baseline_prediction, future_tt, prepared,
        )
        final_physics = physics_metrics(
            final_prediction, target, anchor, prepared,
        )
        baseline_physics = physics_metrics(
            baseline_prediction, target, anchor, prepared,
        )
        per_seed_metric_rows.extend([
            {
                "Seed": seed,
                "Model": BASELINE_MODEL_NAME,
                **baseline_metric,
                **baseline_acoustic,
                **baseline_physics,
            },
            {
                "Seed": seed,
                "Model": FINAL_MODEL_NAME,
                **final_metric,
                **final_acoustic,
                **final_physics,
            },
        ])

        # Validation-only seed selection information for figures.
        val_metric = metrics(
            final_val["prediction"],
            final_val["target"],
            prepared,
            final_val["anchor"],
        )
        validation_rows.append({
            "Seed": seed,
            "Val_Temp_RMSE": float(val_metric["Temp_RMSE"]),
            "Val_Salt_RMSE": float(val_metric["Salt_RMSE"]),
        })

        metadata = dict(metadata)
        metadata["Run_Index"] = run_index
        per_seed_metadata.append(metadata)

        # Per-seed outputs.
        seed_dir = args.output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        seed_horizon = horizon_rmse_table(
            final_prediction,
            baseline_prediction,
            target,
            seed_dir,
        )
        seed_horizon.insert(0, "Seed", seed)
        horizon_seed_tables.append(seed_horizon)

        pd.DataFrame([
            per_seed_metric_rows[-2],
            per_seed_metric_rows[-1],
        ]).to_csv(
            seed_dir / "final_model_vs_nlinear_metrics.csv",
            index=False,
        )
        pd.DataFrame([metadata]).to_csv(
            seed_dir / "model_configuration.csv",
            index=False,
        )

        torch.save({
            "seed": seed,
            "final_model_name": FINAL_MODEL_NAME,
            "final_model_full_name": FINAL_MODEL_FULL_NAME,
            "baseline_model_name": BASELINE_MODEL_NAME,
            "physics_constraint": PHYSICS_CONSTRAINT_NAME,
            "opt_acoustic_weight": float(args.opt_acoustic_weight),
            "stability_weight": float(args.stability_weight),
            "arguments": vars(args),
            "final_model_state": copy.deepcopy(final_model).cpu().state_dict(),
            "baseline_model_state": copy.deepcopy(baseline_model).cpu().state_dict(),
            "sparse_processor_state": copy.deepcopy(processor).cpu().state_dict(),
            "deduplicated_blts_state": copy.deepcopy(unique_refiner).cpu().state_dict(),
            "depths": depths,
            "temp_eof_basis": prepared["eof_t"].basis,
            "salt_eof_basis": prepared["eof_s"].basis,
            "coefficient_scale": prepared["coefficient_scale"],
        }, seed_dir / f"past_nlinear_eof_seed{seed}.pt")

        np.savez_compressed(
            seed_dir / f"test_predictions_seed{seed}.npz",
            final_prediction=final_prediction.astype(np.float32),
            nlinear_prediction=baseline_prediction.astype(np.float32),
            target=target.astype(np.float32),
            anchor=anchor.astype(np.float32),
            future_tt=future_tt.astype(np.float32),
            depths=np.asarray(depths, dtype=np.float32),
            origin_ns=prepared["window_meta"]["test"]["origin_ns"],
            valid_ns=valid_ns,
            seed=np.asarray(seed, dtype=np.int64),
        )

        # Keep only arrays required for later representative-seed figures.
        seed_predictions[seed] = {
            "final_prediction": final_prediction,
            "baseline_prediction": baseline_prediction,
            "target": target,
            "anchor": anchor,
        }

        print(
            f"seed={seed}: "
            f"PAST Temp_RMSE={final_metric['Temp_RMSE']:.6f}, "
            f"Salt_RMSE={final_metric['Salt_RMSE']:.6f}; "
            f"NLinear Temp_RMSE={baseline_metric['Temp_RMSE']:.6f}, "
            f"Salt_RMSE={baseline_metric['Salt_RMSE']:.6f}"
        )

        del final_model, baseline_model, processor, unique_refiner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Three-seed aggregation: primary paper numbers are mean ± std.
    # ------------------------------------------------------------------
    per_seed_table = pd.DataFrame(per_seed_metric_rows)
    metric_summary = summarize_seed_metrics(
        per_seed_table,
        args.output_dir,
    )
    _, horizon_summary = summarize_horizon_tables(
        horizon_seed_tables,
        args.output_dir,
    )

    pd.DataFrame(per_seed_metadata).to_csv(
        args.output_dir / "model_configuration_per_seed.csv",
        index=False,
    )

    representative_seed, representative_table = choose_representative_seed(
        validation_rows,
        requested_seed=args.figure_seed,
    )
    representative_table.to_csv(
        args.output_dir / "representative_seed_selection.csv",
        index=False,
    )

    # Root configuration summarizes the actual three-seed protocol.
    pd.DataFrame([{
        "Final_Model": FINAL_MODEL_NAME,
        "Final_Model_Full_Name": FINAL_MODEL_FULL_NAME,
        "Baseline_Model": BASELINE_MODEL_NAME,
        "Seeds": ",".join(str(seed) for seed in seeds),
        "N_Seeds": len(seeds),
        "Primary_Reporting": "mean ± sample std across independent seeds",
        "Representative_Seed_For_Figures": representative_seed,
        "Representative_Seed_Selection": (
            "validation-only distance to three-seed mean Temp/Salt RMSE"
            if args.figure_seed is None
            else "user-specified --figure-seed"
        ),
        "Optimized_Acoustic_Weight": float(args.opt_acoustic_weight),
        "Stability_Weight": float(args.stability_weight),
        "Physics_Module": PHYSICS_CONSTRAINT_NAME,
        "Temporal_Module": "S1-DeduplicatedBLTS",
    }]).to_csv(
        args.output_dir / "three_seed_configuration.csv",
        index=False,
    )

    # Optimized-physics diagnostic tables retain both raw seed values and mean±std.
    physics_columns = [
        "Seed",
        "Model",
        "Acoustic_TT_RMSE_s",
        "Acoustic_Centered_RMSE_s",
        "Acoustic_6hChange_RMSE_s",
        "N2Proxy_RMSE_s2",
        "N2Proxy_MAE_s2",
        "StaticInstability_Rate",
        "Reference_StaticInstability_Rate",
        "StaticInstability_Rate_Gap",
        "N2Envelope_Violation_Fraction",
        "Curvature_RMSE",
        "Curvature_Severity",
        "Density_Inversion_Fraction",
    ]
    physics_per_seed = per_seed_table[
        [column for column in physics_columns if column in per_seed_table.columns]
    ].copy()
    physics_per_seed.to_csv(
        args.output_dir / "optimized_physics_consistency_metrics_per_seed.csv",
        index=False,
    )
    physics_summary = _mean_std_summary(
        physics_per_seed,
        group_column="Model",
        id_columns=("Seed", "Model"),
    )
    physics_summary.to_csv(
        args.output_dir / "optimized_physics_consistency_metrics.csv",
        index=False,
    )

    # Save deterministic preprocessing once.
    np.savez_compressed(
        args.output_dir / "locked_preprocessing_eof.npz",
        depths=np.asarray(depths, dtype=np.float32),
        temp_eof_basis=prepared["eof_t"].basis,
        salt_eof_basis=prepared["eof_s"].basis,
        coefficient_scale=prepared["coefficient_scale"],
        mean_residual=prepared["mean_residual"],
        upper_indices=prepared["upper"],
        upper_t_mean=prepared["upper_t_mean"],
        upper_t_std=prepared["upper_t_std"],
        upper_s_mean=prepared["upper_s_mean"],
        upper_s_std=prepared["upper_s_std"],
        opt_tt_offset=np.asarray(prepared["opt_tt_offset"], dtype=np.float64),
        opt_tt_slope=np.asarray(prepared["opt_tt_slope"], dtype=np.float64),
        opt_tt_anomaly_scale=np.asarray(prepared["opt_tt_anomaly_scale"], dtype=np.float64),
        opt_center_scale=np.asarray(prepared["opt_center_scale"], dtype=np.float64),
        opt_change_scale=np.asarray(prepared["opt_change_scale"], dtype=np.float64),
        n2_proxy_floor=np.asarray(prepared["n2_proxy_floor"], dtype=np.float64),
        n2_proxy_ceiling=np.asarray(prepared["n2_proxy_ceiling"], dtype=np.float64),
        n2_proxy_scale=np.asarray(prepared["n2_proxy_scale"], dtype=np.float64),
    )

    # Figures use one real seed, chosen without test information.  Primary
    # numerical tables above remain the three-seed mean±std, not this seed.
    rep = seed_predictions[representative_seed]
    final_prediction = rep["final_prediction"]
    baseline_prediction = rep["baseline_prediction"]
    target = rep["target"]
    anchor = rep["anchor"]

    np.savez_compressed(
        args.output_dir / "final_test_predictions.npz",
        final_prediction=final_prediction.astype(np.float32),
        nlinear_prediction=baseline_prediction.astype(np.float32),
        target=target.astype(np.float32),
        anchor=anchor.astype(np.float32),
        future_tt=future_tt.astype(np.float32),
        depths=np.asarray(depths, dtype=np.float32),
        origin_ns=prepared["window_meta"]["test"]["origin_ns"],
        valid_ns=valid_ns,
        representative_seed=np.asarray(
            representative_seed,
            dtype=np.int64,
        ),
        all_seeds=np.asarray(seeds, dtype=np.int64),
    )

    representative_index = figure8(
        final_prediction, baseline_prediction, target, anchor, depths,
        valid_ns, args.output_dir, args.figure_dpi,
    )
    figure9(
        final_prediction, target, depths, valid_ns,
        args.output_dir, args.figure_dpi,
    )
    write_figure_captions(
        args.output_dir,
        representative_seed=representative_seed,
    )

    # Compact console summary.
    primary_metrics = [
        "Temp_RMSE",
        "Temp_MAE",
        "Salt_RMSE",
        "Salt_MAE",
        "Acoustic_TT_RMSE_s",
        "Acoustic_Centered_RMSE_s",
        "Acoustic_6hChange_RMSE_s",
        "N2Proxy_RMSE_s2",
        "StaticInstability_Rate_Gap",
        "N2Envelope_Violation_Fraction",
    ]
    print("\nTHREE-SEED PRIMARY METRICS (mean ± std)")
    for _, row in metric_summary.iterrows():
        values = []
        for metric in primary_metrics:
            mean_key = f"{metric}_Mean"
            std_key = f"{metric}_Std"
            if mean_key in row.index:
                values.append(
                    f"{metric}={row[mean_key]:.6f}±{row[std_key]:.6f}"
                )
        print(f"{row['Model']}: " + ", ".join(values))

    print("\nTHREE-SEED HORIZON TEMPERATURE + SALINITY RMSE (mean ± std)")
    for _, row in horizon_summary.iterrows():
        print(
            f"{row['Forecast_Horizon']:>4} | "
            f"NLinear T={row['NLinear_EOF_Temp_RMSE_C_Mean']:.4f}±"
            f"{row['NLinear_EOF_Temp_RMSE_C_Std']:.4f} | "
            f"PAST T={row['PAST_NLinear_EOF_Temp_RMSE_C_Mean']:.4f}±"
            f"{row['PAST_NLinear_EOF_Temp_RMSE_C_Std']:.4f} | "
            f"NLinear S={row['NLinear_EOF_Salinity_RMSE_PSU_Mean']:.4f}±"
            f"{row['NLinear_EOF_Salinity_RMSE_PSU_Std']:.4f} | "
            f"PAST S={row['PAST_NLinear_EOF_Salinity_RMSE_PSU_Mean']:.4f}±"
            f"{row['PAST_NLinear_EOF_Salinity_RMSE_PSU_Std']:.4f}"
        )

    print(
        f"\nRepresentative seed for figures (validation-only): "
        f"{representative_seed}"
    )
    print(
        f"Representative Figure 8f test-window index: "
        f"{representative_index}"
    )
    print(
        f"Figures, three-seed tables, checkpoints and source data: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
