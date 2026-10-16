# FDI Team 02 · Fama–MacBeth 프로토타입

기업 규모와 모멘텀으로 다음 달 KOSPI 종목별 수정수익률 순위를 예측하는 최소 실행본입니다. **재무·텍스트 특성이 포함된 최종 연구 모형은 아닙니다.**

## 코랩 시작

[코랩에서 실행](https://colab.research.google.com/github/Hwaple02/fdi-team02/blob/main/colab_prototype.ipynb)

비공개 저장소이므로 GitHub 저장소 접근 권한이 필요합니다. 링크로 열리지 않으면 GitHub에서 `colab_prototype.ipynb`를 다운로드하여 코랩의 파일 → 노트북 업로드로 여세요. 노트북은 실행에 필요한 소스를 내장하므로 별도의 GitHub 토큰이 필요하지 않습니다.

1. 코랩에서 셀을 위에서 아래로 실행합니다.
2. 데이터 다운로드 셀에 팀 내부에서 받은 Supabase 서명 다운로드 URL을 입력합니다. 입력은 숨겨집니다. Colab Secret의 `FDI_TRAINING_URL`에 저장해 사용할 수도 있습니다.
3. 학습자료 해시를 확인하고 12개 CSV만 내려받습니다.
4. FM 학습·3개 확장창 검증·월별 IC 계산이 실행됩니다.
5. 마지막 셀에서 구간별 결과를 확인합니다.

코랩 서비스에서 직접 실행한 결과는 아직 확인하지 않았습니다. 동일 코드를 로컬 Python 환경에서 실행·검증했습니다.

## 데이터와 접근

- Supabase 프로젝트: `FDI_Team02` / 서울 / `jiruebyoipsuzywedbhq`
- 비공개 버킷: `fdi-team02-private`
- 파일: `training/v1/fdi-team02-training-v1.zip`
- 구성: warmup_2000 및 train_2001_2021의 6개 변수씩 12개 CSV
- 테스트 자료, 재무 엑셀, 뉴스 자료는 업로드하지 않았습니다.
- 원자료·서명 URL·관리자 키는 GitHub에 포함하지 않습니다.
- 서명 URL은 발급 후 7일간 유효합니다. 만료 시 프로젝트 소유자가 Supabase Storage 화면에서 같은 파일의 새 서명 링크를 발급하여 팀 내부에 전달합니다. 링크를 가진 사람은 만료 전까지 다운로드할 수 있으므로 공개 게시하지 않습니다.
- 업로드 검증 이후 초기 업로드용 함수는 항상 HTTP 410을 반환하도록 닫았습니다.

## 로컬 실행

Python 3.11 이상. pandas/numpy 버전은 실제 검증한 버전으로 고정했습니다.

```bash
python -m pip install -r requirements.txt
python download_training_data.py --output-dir data/raw
python run_all.py --data-dir data/raw --output-dir artifacts/run_v02
```

이미 원본 CSV가 있다면 다운로드를 생략하고 `--data-dir`에 그 폴더를 지정하세요. 기존 산출물이 있는 폴더에는 덮어쓰지 않습니다.

## 모형과 시간 기준

- 입력월 `t`, 예측·평가월 `m=t+1`.
- 사용자 확정 모멘텀: **입력월 t−12~t−2, 11개월**, 곧 예측월 m−13~m−3. 예: 2001년 1월 입력은 2000년 1~11월 수익률을 사용해 2001년 2월을 예측합니다.
- 규모는 월말 시가총액 로그. 양수 시가총액이 있는 종목을 최소 입력 후보로 사용합니다.
- 각 입력월 후보 종목 내 평균동순위/관측수−0.5로 변환합니다. 특성 결측만 0으로 대체합니다. 원자료·목표 결측은 보존합니다.
- 다음 달 수정수익률의 백분위 순위를 학습합니다. 목표 순위는 해당 월 제공 수익률 비결측 종목 전체에서 산출합니다.
- 매월 절편 포함 OLS를 적합하고 월별 계수를 동일 가중 평균합니다. 최소 30종목, 설계행렬의 완전한 열 순위를 요구합니다. 실패월은 이유를 기록하고 평균에서 제외합니다.
- 검증 중 계수는 고정됩니다. Spearman IC는 예측값 순위와 실제 수익률 순위의 Pearson 상관으로 계산합니다.
- 분할은 목표월 기준: 2001–2012→2013–2015, 2001–2015→2016–2018, 2001–2018→2019–2021.
- 2000년 수익률을 학습 정답으로 사용하지 않으며 2022년 이후 파일은 읽지 않습니다.
- 전체 학습기간으로 재적합한 계수는 `frozen_prototype.json`에 저장합니다. 이 파일은 최소 프로토타입용이며 최종 연구 모형의 확정을 뜻하지 않습니다.

## 코드 구조

| 파일 | 역할 |
|---|---|
| `PROTOTYPE_SPEC.md` | 프로토타입 명세 |
| `config.json` | 확정 모멘텀·기간 설정 |
| `load_training_data.py` | 12개 허용 CSV 검사·긴 패널 생성·정확한 다음 달 목표 연결 |
| `fm_pipeline.py` | 특성 생성·월별 회귀·평균 계수·IC |
| `run_all.py` | 전체 실행과 결과 저장 |
| `download_training_data.py` | 비공개 링크 다운로드·해시 검증·허용 파일 추출 |
| `build_colab.py` | 최신 소스로 독립 실행 노트북 재생성 |
| `colab_prototype.ipynb` | 팀 공유용 코랩 실행본 |
| `tests/` | 가상 자료로 시점·계수·IC·입력 검증 |
| `validation_results/` | 실제 내부 검증 요약·계수. 종목별 예측 원본은 로컬 실행 출력에 저장 |

## 출력

`fold_1`~`fold_3`: 월별 계수, 평균 계수, 적합 로그, 검증 예측, 월별 IC.
루트: 합친 월별 IC, 구간별 요약, 전체 학습계수와 동결 파일·해시, 입력 파일 해시·결측 점검 기록.

로더만 사용하려면:

```bash
python load_training_data.py --data-dir data/raw --output-dir artifacts/load_v1 --with-targets
```

패널은 `month`, 문자열 `stock_code`, 수정수익률(소수), 원주가(원), 주식수(주), 시가총액(원), ISIN, 회사명, source_period, has_market_observation으로 구성됩니다. 마지막 플래그는 실제 거래가능성의 완전한 판정이 아닙니다. 압축 CSV 재로딩 시 `stock_code`는 문자열로 지정하세요.

## 검증과 해석 범위

```bash
python -m unittest discover -s tests -v
```

13개 자동 검사 통과. 자세한 실제 결과는 `VALIDATION.md`를 참고하세요. 두 변수 최소 모형의 내부 검증이며, IC가 좋다는 주장이나 텍스트 추가 효과를 입증한 결과가 아닙니다. Newey–West 유의성 검정, 재무·텍스트 결합, 거래비용·포트폴리오 검증은 후속 구현입니다. 중간 계획보고서에 실증 결과를 넣지 않는 과제 규칙과 내부 개발용 결과를 구분하세요.
