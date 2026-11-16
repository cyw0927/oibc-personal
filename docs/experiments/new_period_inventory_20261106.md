# #1314 새 과거 평가 기간 자료 조사 (2026-11-06)

`Documents/Codex`, `Downloads`, 프로젝트의 OOF inbox를 파일명으로 조사하고 기존 OOF parquet의 row group 시각 통계를 확인했다. 각 row group이 잠금 시작 전임을 확인한 OOF만 fold·시각 열을 읽었다. 기존 인계 ZIP은 파일명 목록만 확인했다. 대회 원본 ZIP과 잠금 데이터는 열지 않았다.

- FS2+L1 3-seed 평균과 FS2 각 seed 3개, 기후평균·공식 baseline 재현·선형회귀·태양 위치 회귀의 기존 OOF는 전부 JUL/OCT/JAN/FEB 4개 기존 fold, 각 700,248행이었다. 신규 기간의 FS2+L1 OOF는 이번 검색 범위에서 찾지 못했다.
- 안전 분리본 `C:/Users/dkrak/Documents/Codex/OIBC_1314_review/unlocked_data/train/`의 generation, forecast_wa, forecast_wb는 메타데이터상 2023-06-01 01:00~2024-11-01 00:00 KST, 모든 row group 잠금 전이다. `safe_features/group_features_train.parquet`도 같은 구간이다. `variant_features/group_features_train.parquet`의 최대는 2024-12-05 00:00 KST다. 실제 예보 수신 시점은 미확인이다.
- 2024년 4월·8월은 과거 대리 모델 실험에 이미 사용됐고, 실제 #1314의 해당 기간 FS2+L1 바탕 OOF가 없어 직접 비교가 아니다. 4월은 코호트가 두 VPP 20MW 요건을 만족하지 못했다. 7·10월 v2 그룹 예측은 재현됐으나 1·2월 v2 원본은 확인되지 않았다.
- `C:/dev/icn-solar-oibc/oibc/src/hf_forecast/adapters/oibc2026/fs2.py`가 존재한다. 이것만으로 FS2+L1 3-seed 전체 학습·정확한 재현과 새로운 기간 OOF를 생성할 수 있다고 보장할 수는 없다. cutoff·버전·환경·입력 provenance와 종속 코드부터 점검해야 한다.

결론: 새 과거 기간의 안전한 학습자료는 일부 있으나, 같은 기간에 동일 FS2+L1 OOF·#1314 기준 예측·적합한 코호트까지 갖춰진 독립 비교 세트는 이번 검색에서 발견하지 못했다. 조건을 충족하지 않아 이번에는 새 성능 수치를 계산하지 않았다. 새 fold를 만들려면 과거 cutoff 이전 학습의 FS2+L1 3-seed OOF(모델·데이터·split·feature 버전, 코드 commit, 키, SHA, 학습 범위 포함)와 해당 fold의 안전한 #1314 그룹 예측/재현 자료, 예보 수신 근거가 필요하다. 반복 사용한 기간을 독립 검증이라 부르지 않는다.
