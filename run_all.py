"""Run warmup/train loading, features, expanding validation and final train fit."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from load_training_data import load_training_data, make_training_targets
from fm_pipeline import FEATURES, build_features, evaluate_fold, fit_monthly_fm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if config["momentum"]["status"] != "confirmed":
        p.error("모멘텀 기간 미확정. 팀 확인 전 학습하지 않습니다.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error("출력 폴더가 비어 있지 않습니다. 새 출력 폴더를 지정하세요.")
    loaded = load_training_data(args.data_dir, config["expected_stock_count"])
    features = build_features(loaded.panel, config["momentum"])
    targets = make_training_targets(loaded.panel)
    data = features.merge(targets, on=["month", "stock_code"], how="inner", validate="one_to_one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "load_audit.json").write_text(json.dumps(loaded.audit, ensure_ascii=False, indent=2))
    summaries, all_ic = [], []
    for i, fold in enumerate(config["validation_folds_by_target_month"], 1):
        directory = args.output_dir / f"fold_{i}"
        directory.mkdir()
        monthly, mean, logs, prediction, scores = evaluate_fold(data, fold)
        monthly.to_csv(directory / "monthly_coefficients.csv", index=False)
        logs.to_csv(directory / "fit_log.csv", index=False)
        prediction[["month", "target_month", "stock_code", "prediction", "target_return", "target_rank"]].to_csv(
            directory / "validation_predictions.csv.gz", index=False, compression="gzip"
        )
        scores.to_csv(directory / "monthly_ic.csv", index=False)
        (directory / "mean_coefficients.json").write_text(json.dumps(mean, indent=2))
        valid = scores.ic.dropna()
        summaries.append({"fold": i, "train": fold["train"], "validation": fold["validation"],
                          "fit_months": len(monthly), "skipped_fit_months": int(logs.status.ne("ok").sum()),
                          "valid_ic_months": len(valid), "mean_ic": float(valid.mean()) if len(valid) else None,
                          "positive_ic_fraction": float(valid.gt(0).mean()) if len(valid) else None})
        all_ic.append(scores.assign(fold=i))
    combined = pd.concat(all_ic, ignore_index=True)
    if combined.target_month.duplicated().any():
        raise ValueError("검증 목표월 중복: 전체 IC 집계 불가")
    combined.to_csv(args.output_dir / "monthly_ic.csv", index=False)
    valid = combined.ic.dropna()
    summary = {"model": "FM size + momentum prototype", "features": FEATURES,
               "momentum": config["momentum"], "folds": summaries,
               "overall_mean_ic": float(valid.mean()) if len(valid) else None,
               "valid_ic_months": len(valid), "test_evaluated": False,
               "note": "학습기간 내부 검증 결과. 최종 모델 또는 텍스트 추가 효과의 검증 결과가 아님."}
    (args.output_dir / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    monthly, mean, log = fit_monthly_fm(data, "2001-01", "2021-12")
    frozen = {"training_target_months": ["2001-01", "2021-12"], "features": FEATURES,
              "coefficients": mean, "config": config, "n_fit_months": len(monthly),
              "input_hashes": {f["filename"]: f["sha256"] for f in loaded.audit["files_read"]},
              "prototype_only": True}
    payload = json.dumps(frozen, ensure_ascii=False, sort_keys=True, indent=2).encode()
    (args.output_dir / "frozen_prototype.json").write_bytes(payload)
    (args.output_dir / "frozen_prototype.sha256").write_text(hashlib.sha256(payload).hexdigest() + "\n")
    monthly.to_csv(args.output_dir / "full_train_monthly_coefficients.csv", index=False)
    log.to_csv(args.output_dir / "full_train_fit_log.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
