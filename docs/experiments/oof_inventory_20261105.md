# #1314 다른 기간 직접 비교에 필요한 자료 확인 (2026-11-05)

## 확인한 자료

- FS2+L1 3-seed 평균 사이트 OOF: `C:\Users\dkrak\Documents\Codex\OIBC_shared\1314_dev_20261104\oof_dev.parquet`, SHA-256 `9d5a75f49ff56a891500461517b2a317da829bc6a0946447b6a5f47f4212fe80`. ZIP `C:\Users\dkrak\Downloads\oibc_l1_avg3_1452_v1.zip` README에도 같은 SHA, FS2+L1 3-seed 평균, DEV 4 fold가 명시돼 있다. 메타데이터에서 모든 row group이 잠금 이전임을 확인한 뒤 fold·시간·ID 컬럼만 읽었다.
- 내용: 2024 JUL 224곳·166,656행, OCT 252곳·187,488행, 2025 JAN 253곳·182,160행, FEB 253곳·163,944행. APR/AUG 등 다른 달 예측은 이 파일에 없다. 4개 fold는 기존에 사용된 개발 구간이므로 독립 평가 구간이 아니다.
- 이전 v1 그룹 OOF: `C:\Users\dkrak\Documents\Codex\OIBC_shared\1314_dev_20261104\dev_group_preds_ens.parquet`, SHA-256 `b291b5d98dba4b861a19e4493bc3a62a7c7482a13f7a208cb3dd63d623478ccd`. 동일한 4개 구간·2개 그룹이 있으나 v1이다. 과거 전처리는 원본 전체를 읽고 행을 필터했으므로 당시 잠금 기간 원본을 읽지 않았다고 보장할 수 없다. v2와 동일하다고 가정하면 안 된다.
- v2 그룹 OOF: `C:\Users\dkrak\Documents\Codex\OIBC_1314_review\personal_safe\handoff_5_20261105\v2_jul_oct_group_oof.parquet`, SHA `9b67dd50…`. 안전 분리 재현 7·10월만 확인됨. 1·2월 v2 완전한 재현 자료는 이번 검색 범위에서 확인하지 못했다.
- 새 모델용 안전 분리 그룹 특성: `C:\Users\dkrak\Documents\Codex\OIBC_1314_review\variant_features\group_features_train.parquet`: 메타데이터상 2024-12-05 00:00 KST까지만. 이를 포함해 찾은 안전 분리 예보 파일에는 JAN/FEB 전체 예보가 없다. 따라서 FS2+L1 겨울 OOF가 있다는 것만으로 새 그룹 예측을 재현할 수 없다.

## 추가로 필요한 자료

1. 2025 JAN/FEB: 해당 fold 이전에만 학습한 #1314 v2 그룹별 OOF 또는 동일 코드·학습/예보/SMP 자료의 안전 분리본. 이미 있는 253개소 FS2+L1 사이트 OOF와 키·실적·버전·SHA 대조 필요. 새 태양 높이 모델도 돌리려면 각 fold 학습 및 검증에 필요한 해당 기간 이전의 발전량, WA/WB 예보, 사이트·SMP 자료가 잠금 기간과 물리적으로 분리돼야 한다. 데이터 및 예보가 실제 예측 시점까지 사용 가능했음을 입증하는 발행·수신 시각도 확인한다.
2. APR/AUG 등 추가 구간: #1314 바탕 FS2+L1을 각 검증 기간 시작 이전 자료로 학습한 발전소·시간별 OOF, 같은 fold의 v2 그룹 OOF 또는 그 안전 재현 자료가 필요하다. 사이트·시간·actual/prediction, fold, 학습 cutoff, 모델·데이터·split·feature 버전, 코드 commit 및 SHA-256을 함께 확인한다. 모델 개발에 이미 쓰인 달은 독립 검증이 아님을 별도 표시한다.
3. 기존 FS2+L1 바탕 모델과 예측/학습 코드는 과거 Forecast 담당 팀원 A에게, 데이터 계약·안전 분리 및 예보 시점 근거는 팀원 B에게, #1314 v2 산출물 원본 및 재현 경로는 생성 주체(Claude/해당 실행 환경)부터 확인할 수 있다. 별도 전송은 사용자가 지시할 때만 한다.

대회 데이터 및 파생물은 Git에 올리지 않는다. 이번 조사에서 잠금 기간 데이터를 읽지 않았다.
