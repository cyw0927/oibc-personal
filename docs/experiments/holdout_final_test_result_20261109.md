# HOLDOUT 최종 1회 시험 결과 + v3 후보 (2026-11-09, Claude)

사전 등록 문서는 `docs/holdout_final_test_protocol_20261109.md`(SHA 28a9f51e)이고, 시험은 한 번만 실행했다.
수치는 README 잠정 정산 기준이다(D05 HOLD, official_S=null).

## HOLDOUT 결과 (2025-03 ~ 05, 253곳 전체, 그룹 직접 예측 부분)

| 후보 | 3월 | 4월 | 5월 | 합계 | 편향(SMP·k) | 그룹 wMAE |
|---|---|---|---|---|---|---|
| H0 (v0.2, 5차 제출 모델) | 15.699M | 17.614M | 24.089M | **57.402M** | +0.51% | 0.1160 |
| H1 (+ 하루 맥락 피처) | 15.357M | 17.720M | 20.702M | **53.779M (−6.3%)** | +0.36% | 0.1087 |

- 3월과 5월은 개선, 4월은 +0.6% 악화.
- 사전 규칙(합계 개선 + 3개월 중 2개월 이상 개선)에 따라 **H1을 채택**했다.
- 결과 파일: `OIBC_1314_review/personal_safe/holdout_20261109/results.json`(Git 금지).
- 결과 JSON의 "2025-06" 항목은 2025-06-01 00:00 한 시간(야간)뿐이다. 판정은 3·4·5월로 했다.

## 한계

- FS2+L1 발전소 모델은 HOLDOUT 예측이 없어서 시험하지 못했다. 최종 결합(0.4/0.6)의 HOLDOUT 성능도 직접 확인하지 않았다.
- 시험한 기간은 봄 3개월뿐이다. 12개월 롤링 결과(−1.6%, 12개월 중 7개월 개선)와 함께 봐야 한다.

## v3 후보 (미제출)

위치: `C:\Users\dkrak\Downloads\인천공항 작업 파일\_ai_analysis\OIBC\candidates\Y1314_h1_ctx_full_v3\`

- **구성:** 0.4 × FS2+L1(SHA 226b2bc9, 변경 없음) + 0.6 × H1 그룹 직접. H1은 2023-06-01 01:00 ~ 2025-06-01 00:00 전체로 학습했고 14,956행, 피처 110개다. 배정은 #990 reference다.
- **파일:**
  - composition.csv `933b6709` (#990과 동일)
  - forecast.csv `8da0884d`
  - 팀 validator PASS, 공개 validator PASS
- **재현 점검:** H0를 옛 학습구간으로 다시 돌려 5차 제출본 v2(`38f1bc37`)를 재현했다. 최대 차이는 5e-7 kWh다.
- **v2 대비 월별 총량:** −1.6% ~ +1.0%.
- **코드:** `scripts/oibc_1314/claude_1109/`
  - `prep_group_features_full.py`
  - `ctx_features.py` (5ccea378)
  - `holdout_test.py` (ac87d0d2)
  - `final_predict.py`
