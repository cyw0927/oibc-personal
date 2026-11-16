# OIBC 2026 태양광 VPP 발전량 예측 — 개인 연구 저장소

2026 OIBC Challenge(#1314, 팀 WATT NEXT)에서 진행한 개인 실험 코드와 최종 결과물이다.
발전소 253곳을 두 그룹(VPP)으로 나누고, 그룹별 시간당 발전량을 전날 17시 기준으로 예측하는 대회였다.

## 최종 결과

| 항목 | 값 |
|---|---|
| 최종 제출 | **v14s** (2026-11-13, 17번째 제출) |
| 정산금 | **6,462,653,869원** (학부생 부문 **167위**) |
| 첫 제출 대비 | +44.3M (첫 제출 6,418,339,340원) |
| 구조 | v10s 90% + 유사 기상일(AnEn) 5% + ECMWF 물리 모델 5% |

## 바로가기

- 📘 **[프로젝트 회고록](docs/oibc_2026_retrospective.md)** — 한 파일에 두 부분
  - 1부(1~13장): 사용자 시점 회고, Codex 작성
  - 2부(14~20장): Claude 기술 회고 — 버전별 수치, 외부 예보 파이프라인, 실패 분석, 실수
- 📦 **[최종 제출물 v14s](final/v14s/)** — `composition.csv`, `forecast.csv`, `MANIFEST.json`, 형식 검사 결과, [모델 설명서](final/v14s/v14s_모델설명.md), [외부 데이터 규정 점검](final/v14s/규정점검_v14s.md)
- 📝 이전 버전 설명서 — [v9](final/model_notes/v9_모델설명.md) (외부 예보 박스 → 하루 총량 모델), [v10](final/model_notes/v10_모델설명.md) (시간별 GFS → 발전소 모델)
- 🧪 [실험 문서 모음](docs/experiments/) — 11/05~11/09 실험 프로토콜·결과 (기각된 실험 포함)

## 폴더 구조

```
README.md
docs/
  oibc_2026_retrospective.md     회고록 (1부 Codex + 2부 Claude, 한 파일)
  experiments/                   실험 프로토콜·결과 문서 (파일명 끝이 날짜)
final/
  v14s/                          최종 제출 파일과 근거
  model_notes/                   v9, v10 모델 설명서
scripts/oibc_1314/
  *.py                           팀 v0.2(#1314) 그룹 직접 예측에서 이어받은 스크립트 + 초기 실험
  claude_1109/                   Claude 실험·최종 생성 코드 (v4b ~ v15)
  codex/                         Codex 독립 검증·감사 스크립트
```

`scripts/oibc_1314/` 루트 스크립트는 팀 #1314 v0.2 작업 트리(commit c955267)에서 가져왔다. 개별 발전소 기준 모델과 대회 데이터는 포함하지 않는다.

## 버전 흐름 (대시보드 기준)

| 버전 | 정산금 | 핵심 변화 |
|---|---|---|
| v2 | 6,436,427,653 | 발전소별 합산 40% + 그룹 직접 예측 60% |
| v7 | 6,450,725,175 | 오차 상쇄형 그룹 배정 + 하루 총량 보정 |
| v9 | 6,455,442,984 | ECMWF·GFS 고정 박스 → 하루 총량 모델 |
| v10 | 6,461,252,004 | 시간별 GFS → 발전소 모델 |
| v11 | 6,458,729,356 | 혼합비 재조정 (과적합으로 하락) |
| **v14s** | **6,462,653,869** | v10s + 유사 기상일 5% + ECMWF 물리 5% |

자세한 수치와 해석은 회고록 2부 14~15장.

## 주요 스크립트 (`scripts/oibc_1314/claude_1109/`)

| 역할 | 파일 |
|---|---|
| 외부 예보 수집 (고정 박스, 발전소 좌표 미사용) | `gfs_box_fetch.py`, `gfs_rich_fetch.py` (GFS 시간별·시간차·추가 변수·4구역), `ecmwf_box_fetch.py`, `ecmwf_hourly_gcp_fetch.py` (ECMWF 12Z D−2, GCP 미러), `ecmwf_00z_fetch.py` (00Z D−1 + 공개 시각 기록) |
| 발전소 모델 E·B21 | `cv_site.py` (12개월 검증), `final_site.py` (최종 학습), `site_features.py` |
| 그룹 직접 모델 H1 | `cv_group.py`, `final_h1.py` |
| 하루 총량 보정 | `cv_day3.py`, `final_day.py`, `combine_day.py` |
| 그룹 배정 | `comp_search.py` |
| 유사 기상일 / 물리 모델 | `anen.py`, `phys_blend_exp.py` |
| 최종 v14 생성 | `final_v14.py` (기반 예측 경로를 v10s로 주면 v14s) |
| 평가 | `eval_v7.py` (5개월), `eval_holdout7.py` (미사용 7개월), `eval_12m.py` (12개월·LOMO), `blend_var.py`·`blend3e.py` (v14 비교) |
| 기각 실험 | `stack_total.py`, `mos_ridge.py`, `site_anen.py`, `last_two.py`, `diag_spread.py`, `v15_eval.py` 등 — 결과는 회고록 2부 17장 |

스크립트는 저장소 밖의 대회 데이터·중간 산출물 경로(`personal_safe` 등)를 직접 참조한다. 이 저장소만 받아서는 실행되지 않고, 저장소를 받는다고 모델이 학습되지도 않는다.

## 데이터와 평가 원칙

- 대회 데이터, OOF, 중간 산출물은 Git에 넣지 않는다(`.gitignore`). **예외:** 대회 종료 후 최종 제출 파일 `final/v14s/*.csv`만 기록용으로 추가했다.
- 잠긴 HOLDOUT 자료로 모델을 고르지 않고, 리더보드 결과로 튜닝하지 않는다. 내부 정산 계산은 잠정값이며 공식 점수가 아니다. 보관된 v1 4-fold 결과와 v2 2-fold 재현 근거는 서로 다르니 섞어 해석하지 않는다.
- 외부 데이터는 DataEn 공지(2026-10-28 "외부 소스 이용 안내") 조건을 따른다: 공개·상업 이용·재취득 가능, 전날 17:00 KST 이전 공개, 발전소 좌표 매칭 없이 넓은 영역 요약만 사용.
- Claude와 Codex는 로컬 상태판 `C:\dev\icn-solar-oibc\STATUS.md`와 `status/claude.md`·`status/codex.md`를 공유했다(로컬 전용, 커밋하지 않음).

## 이어받은 스크립트 사용 시 주의

`scripts/oibc_1314/` 루트의 팀 시절 스크립트는 출처 보존용이며 안전한 기본값이 아니다. `prep_group_features.py`는 기본으로 테스트 예보를 읽고, `dev_ens.py`는 혼합 가중치 격자를 평가한다. 현재 규칙과 대조하지 않고 실행하지 말 것. `safe_reproduce.py`는 별도의 JUL/OCT 전용 경로로, 읽기 전에 parquet row-group 시각 통계를 확인하고, 미리 자른 시장가격 CSV를 기대하며, 고정 0.4/0.6 혼합만 평가한다. 출력은 저장소 밖에 둔다. JAN/FEB 전체 재학습은 안전하게 분리된 입력이 필요하며 JUL/OCT 결과로 확립되지 않는다.
