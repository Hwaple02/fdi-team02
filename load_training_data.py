"""Load only the supplied warmup/train CSVs. Never discover or read test files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PERIODS = {
    "warmup_2000": ("2000-01", "2000-12"),
    "train_2001_2021": ("2001-01", "2021-12"),
}
NUMERIC = (
    "adjusted_monthly_return", "raw_month_end_price",
    "raw_month_end_shares_outstanding", "raw_month_end_market_cap",
)
VARIABLES = NUMERIC + ("isin", "company_name")


@dataclass
class LoadResult:
    panel: pd.DataFrame
    audit: dict


def _read_wide(path: Path, period: str, variable: str, expected_count: int):
    # Exact allowlist filenames only; no glob and no test-period parameter.
    expected_name = f"{period}__{variable}.csv"
    if period not in PERIODS or variable not in VARIABLES or path.name != expected_name:
        raise ValueError("허용되지 않은 입력 파일")
    if path.is_symlink():
        raise ValueError(f"심볼릭 링크 입력은 허용하지 않습니다: {path.name}")
    raw = path.read_bytes()
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    if not rows or not rows[0] or rows[0][0] != "month":
        raise ValueError(f"month 헤더 없음: {path.name}")
    header = rows[0]
    codes = header[1:]
    if len(codes) != len(set(codes)) or not codes:
        raise ValueError(f"중복 또는 빈 종목코드: {path.name}")
    if len(codes) != expected_count:
        raise ValueError(f"종목 수 불일치: {path.name}: {len(codes)} != {expected_count}")
    if any(re.fullmatch(r"[0-9A-Z]{6}", code) is None for code in codes):
        raise ValueError(f"6자리 종목코드 형식 오류: {path.name}")
    if any(len(row) != len(header) for row in rows[1:]):
        raise ValueError(f"행 길이 불일치: {path.name}")
    frame = pd.DataFrame(rows[1:], columns=header, dtype="string")
    start, end = PERIODS[period]
    expected_months = pd.period_range(start, end, freq="M").astype(str).tolist()
    if frame["month"].tolist() != expected_months:
        raise ValueError(f"월 누락·중복·순서·기간 오류: {path.name}")
    wide = frame.set_index("month").replace("", pd.NA)
    wide.columns.name = "stock_code"
    if variable in NUMERIC:
        try:
            wide = wide.apply(pd.to_numeric, errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"비수치 데이터: {path.name}") from exc
        if np.isinf(wide.to_numpy()).any():
            raise ValueError(f"무한대 데이터: {path.name}")
    series = wide.stack(future_stack=True).rename(variable)
    record = {
        "filename": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw), "months": len(wide), "stocks": len(codes),
        "missing": int(series.isna().sum()),
    }
    return series, codes, record


def load_training_data(data_dir: str | Path, expected_stock_count: int = 1086) -> LoadResult:
    """Return every month/code cell, preserving missing values and string codes.

    month is an ISO YYYY-MM string. adjusted_monthly_return is contemporaneous
    raw input, NOT the prediction target. No feature imputation or scaling occurs.
    """
    data_dir = Path(data_dir)
    if expected_stock_count < 1:
        raise ValueError("expected_stock_count는 양수여야 합니다.")
    frames, files, reference_codes = [], [], None
    for period in PERIODS:
        columns = []
        for variable in VARIABLES:
            series, codes, record = _read_wide(
                data_dir / f"{period}__{variable}.csv", period, variable, expected_stock_count
            )
            if reference_codes is None:
                reference_codes = codes
            elif codes != reference_codes:
                raise ValueError(f"파일 간 종목코드 또는 순서 불일치: {record['filename']}")
            columns.append(series)
            files.append(record)
        frame = pd.concat(columns, axis=1).reset_index()
        frame["source_period"] = period
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True).sort_values(
        ["month", "stock_code"], kind="stable"
    ).reset_index(drop=True)
    panel["stock_code"] = panel["stock_code"].astype("string")
    if panel.duplicated(["month", "stock_code"]).any():
        raise ValueError("종목·월 중복 키")
    panel["has_market_observation"] = panel[list(NUMERIC)].notna().any(axis=1)
    observed = panel[panel["has_market_observation"]]
    warnings = []
    for variable in NUMERIC[1:]:
        count = int((panel[variable] <= 0).sum())
        if count:
            warnings.append({"kind": "nonpositive_raw_value", "column": variable, "rows": count})
    below_minus_one = int((panel["adjusted_monthly_return"] < -1).sum())
    if below_minus_one:
        warnings.append({"kind": "return_below_minus_one", "rows": below_minus_one})
    audit = {
        "schema_version": "0.1", "python": platform.python_version(),
        "pandas": pd.__version__, "numpy": np.__version__,
        "files_read": files, "allowed_periods": list(PERIODS),
        "rows": len(panel), "months": panel["month"].nunique(),
        "stocks": panel["stock_code"].nunique(),
        "month_min": panel["month"].min(), "month_max": panel["month"].max(),
        "rows_with_market_observation": len(observed),
        "missing_by_variable": {v: int(panel[v].isna().sum()) for v in VARIABLES},
        "observed_rows_by_month": observed.groupby("month").size().to_dict(),
        "warnings": warnings,
        "notes": [
            "전 기간 종목 집합의 모든 월·종목 조합을 보존하며 결측 행을 제거하지 않음",
            "has_market_observation은 실제 거래가능성·상장여부의 완전한 판정이 아님",
            "수익률·특성 결측을 0이나 이전 값으로 채우지 않음",
            "테스트 파일을 열지 않으며 재무정보·텍스트는 결합하지 않음",
        ],
    }
    return LoadResult(panel, audit)


def make_training_targets(panel: pd.DataFrame) -> pd.DataFrame:
    """Separate labels from features; join exact calendar t+1, never next observation.

    Labels only span 2001-01..2021-12. Rank denominator is all nonmissing
    supplied returns that month, before any later model-specific sample filter.
    """
    if panel.duplicated(["month", "stock_code"]).any():
        raise ValueError("종목·월 중복 키")
    if not panel["month"].between("2000-01", "2021-12").all():
        raise ValueError("학습/준비 기간 밖의 패널은 허용하지 않습니다.")
    left = panel[["month", "stock_code"]].copy()
    left["target_month"] = (pd.PeriodIndex(left["month"], freq="M") + 1).astype(str)
    left = left[left["target_month"].between("2001-01", "2021-12")]
    right = panel[["month", "stock_code", "adjusted_monthly_return"]].rename(columns={
        "month": "target_month", "adjusted_monthly_return": "target_return"
    })
    right = right[right["target_month"].between("2001-01", "2021-12")].copy()
    right["target_rank"] = right.groupby("target_month")["target_return"].rank(
        method="average", pct=True, na_option="keep"
    )
    targets = left.merge(right, how="left", on=["target_month", "stock_code"], validate="one_to_one")
    targets["has_target"] = targets["target_return"].notna()
    return targets.sort_values(["month", "stock_code"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="준비용·학습용 12개 CSV만 읽는 패널 로더")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--with-targets", action="store_true", help="다음 달 학습 목표를 별도 CSV로 저장")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = load_training_data(args.data_dir, config["expected_stock_count"])
    output_names = ["market_panel_train.csv.gz", "load_audit.json"]
    if args.with_targets:
        output_names.append("training_targets.csv.gz")
    if any((args.output_dir / name).exists() for name in output_names):
        parser.error("기존 산출물이 있습니다. 새로운 --output-dir을 사용하세요.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.panel.to_csv(args.output_dir / output_names[0], index=False, encoding="utf-8", compression="gzip")
    if args.with_targets:
        targets = make_training_targets(result.panel)
        targets.to_csv(args.output_dir / "training_targets.csv.gz", index=False, encoding="utf-8", compression="gzip")
        result.audit["targets"] = {
            "rows": len(targets), "nonmissing": int(targets["has_target"].sum()),
            "target_month_min": targets["target_month"].min(),
            "target_month_max": targets["target_month"].max(),
            "rank_scope": "각 목표월의 제공 수익률 비결측 종목 전체, 평균 동순위 / 관측수",
        }
    (args.output_dir / "load_audit.json").write_text(
        json.dumps(result.audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"완료: 허용 파일 {len(result.audit['files_read'])}개, "
          f"{result.audit['months']}개월, {result.audit['stocks']:,}종목, {len(result.panel):,}행")
    print(f"저장 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
