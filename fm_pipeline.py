"""Small, deterministic characteristic-based FM prototype; training data only."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["size_rank", "momentum_rank"]
COEFFICIENTS = ["intercept", *FEATURES]


def build_features(panel: pd.DataFrame, momentum: dict) -> pd.DataFrame:
    if momentum.get("status") != "confirmed":
        raise ValueError("모멘텀 기간을 팀 확인 후 config.json에 확정하세요.")
    start = momentum.get("start_lag_from_target_month")
    end = momentum.get("end_lag_from_target_month")
    if not isinstance(start, int) or not isinstance(end, int) or not start >= end >= 1:
        raise ValueError("모멘텀 시차는 예측월 기준 start >= end >= 1이어야 합니다.")
    if panel.duplicated(["month", "stock_code"]).any():
        raise ValueError("중복 종목·월")
    returns = panel.pivot(index="month", columns="stock_code", values="adjusted_monthly_return")
    months = pd.period_range(returns.index.min(), returns.index.max(), freq="M").astype(str)
    returns = returns.reindex(months)
    # Invalid returns and missing history propagate to the entire window.
    returns = returns.where(returns >= -1)
    cumulative = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
    for lag in range(end, start + 1):
        # Input month t = target month m - 1; m-lag = t-(lag-1).
        cumulative = cumulative * (1 + returns.shift(lag - 1))
    cumulative = cumulative - 1
    cumulative.index.name = "month"
    cumulative.columns.name = "stock_code"
    mom = cumulative.stack(future_stack=True).rename("momentum_raw").reset_index()
    out = panel[["month", "stock_code", "raw_month_end_market_cap"]].merge(
        mom, on=["month", "stock_code"], validate="one_to_one"
    )
    out["eligible_at_input"] = out.raw_month_end_market_cap.gt(0)
    out["size_raw"] = np.log(out.raw_month_end_market_cap.where(out.eligible_at_input))
    out["momentum_missing"] = out.momentum_raw.isna()
    # Candidate universe depends only on input-month information, never future labels.
    for raw, ranked in [("size_raw", "size_rank"), ("momentum_raw", "momentum_rank")]:
        values = out[raw].where(out.eligible_at_input)
        out[ranked] = values.groupby(out.month).rank(method="average", pct=True) - 0.5
        out.loc[out.eligible_at_input, ranked] = out.loc[out.eligible_at_input, ranked].fillna(0)
    return out


def fit_monthly_fm(data: pd.DataFrame, train_start: str, train_end: str, min_stocks: int = 30):
    train = data[data.target_month.between(train_start, train_end)]
    rows, logs = [], []
    for month in pd.period_range(train_start, train_end, freq="M").astype(str):
        group = train[(train.target_month == month) & train.eligible_at_input].dropna(
            subset=["target_rank", *FEATURES]
        )
        log = {"target_month": month, "n_stocks": len(group), "status": "ok"}
        if len(group) < max(min_stocks, len(COEFFICIENTS) + 1):
            log["status"] = "insufficient_stocks"
        else:
            x = np.column_stack([np.ones(len(group)), group[FEATURES].to_numpy(float)])
            if np.linalg.matrix_rank(x) < x.shape[1]:
                log["status"] = "rank_deficient"
            else:
                beta, _, _, _ = np.linalg.lstsq(x, group.target_rank.to_numpy(float), rcond=None)
                rows.append({"target_month": month, **dict(zip(COEFFICIENTS, beta))})
        logs.append(log)
    if not rows:
        raise ValueError("적합 가능한 학습월이 없습니다.")
    monthly = pd.DataFrame(rows)
    mean = monthly[COEFFICIENTS].mean().to_dict()
    return monthly, mean, pd.DataFrame(logs)


def predict(features: pd.DataFrame, coefficients: dict) -> pd.Series:
    # Explicit whitelist prevents accidental inclusion of target columns.
    return coefficients["intercept"] + features[FEATURES].mul(
        pd.Series({name: coefficients[name] for name in FEATURES})
    ).sum(axis=1, min_count=len(FEATURES))


def monthly_ic(predictions: pd.DataFrame, start: str, end: str, min_stocks: int = 30):
    rows = []
    for month in pd.period_range(start, end, freq="M").astype(str):
        group = predictions[predictions.target_month == month].dropna(subset=["prediction", "target_return"])
        status, ic = "ok", np.nan
        if len(group) < min_stocks:
            status = "insufficient_stocks"
        elif group.prediction.nunique() < 2 or group.target_return.nunique() < 2:
            status = "constant_rank"
        else:
            # Spearman = Pearson correlation of average ranks; no scipy dependency.
            ic = group.prediction.rank(method="average").corr(group.target_return.rank(method="average"))
        rows.append({"target_month": month, "n_stocks": len(group), "ic": ic, "status": status})
    return pd.DataFrame(rows)


def evaluate_fold(data: pd.DataFrame, fold: dict, min_stocks: int = 30):
    train_start, train_end = fold["train"]
    val_start, val_end = fold["validation"]
    if not train_start <= train_end < val_start <= val_end <= "2021-12":
        raise ValueError("학습/검증 기간이 겹치거나 허용 범위를 벗어납니다.")
    monthly, mean, log = fit_monthly_fm(data, train_start, train_end, min_stocks)
    val = data[data.target_month.between(val_start, val_end) & data.eligible_at_input].copy()
    val["prediction"] = predict(val, mean)
    scores = monthly_ic(val, val_start, val_end, min_stocks)
    return monthly, mean, log, val, scores
