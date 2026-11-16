# #1314 논문 검토와 일별 조정 탐색 (2026-11-05)

## 근거와 적용 가능성
- Yang et al. 2017, *Reconciling solar forecasts: Geographical hierarchy*: 발전소별 합계와 지역 전체 예측을 일관되게 조정. 평균제곱오차 중심이며 대회 수익 개선을 보장하지 않는다.
- Yagli et al. 2019, *Reconciling solar forecasts: Sequential reconciliation*: 공간과 시간의 서로 다른 단위 예측을 순차 조정. 아래 실험은 그 아이디어를 단순화한 변형이며 논문 알고리즘 재현이 아니다.
- Di Fonzo & Girolimetto 2023, *Spatio-temporal reconciliation of solar forecasts*: 공간·시간 일관성과 음수 방지를 함께 고려. 앞 논문의 순차 방식에도 한계가 있을 수 있음을 제시한다.
- Visser et al. 2024, *Probabilistic solar power forecasting*: 오차와 수익이 같은 개념은 아니며 확률 예측을 시장 입찰에 활용한다. 이 대회 정산 규칙과 달라 해당 수익 결과를 그대로 적용할 수 없다.
- Phan et al., *Short-term Solar Power Forecasting Using XGBoost with Numerical Weather Prediction* 포스터: KPCA-XGBoost를 5개 PV 현장에 적용. 한 시간 앞 예측 및 과거 관측을 포함해 이 대회에 그대로 쓰면 시점 위반 가능성이 있다.

## 사전 정의한 실험
#1314 v2 시간별 그룹 예측과 별도 학습한 일별 그룹 합계 모델을 사용한다. 매일 각 그룹의 기존 시간별 패턴을 유지하면서 합계만 일별 모델 예측에 맞춘다. 후처리 비율 탐색 없음. 학습은 각 검증일 이전, 학습행은 관측된 용량이 전체의 30% 이상인 완전한 24시간 일자만 사용한다. 태양광 입력 파일은 잠금 기간 row group을 읽기 전에 차단한다.

7월은 224개소, 10월은 252개소이며 검증 기간의 기존 #1314 v2 OOF와 같은 키·실적·SMP를 사용한다. 예보의 실제 발행·수신 시점 검증은 미완이다. 일별 예측에 사용한 입력의 대상일별 가용성을 별도 검증하기 전에는 제출에 쓸 수 없다. JUL/OCT는 이미 반복 사용된 DEV이므로 독립 테스트가 아니다.

## 결과
README 잠정, D05 HOLD, official_S=null. 벌점은 낮을수록 좋다.

| 기간 | #1314 | 일별 조정 | 차이 |
|---|---:|---:|---:|
| 7월 | 19,120,713원 | 21,845,007원 | +14.25% |
| 10월 | 19,073,121원 | 20,202,226원 | +5.92% |

결과적으로 이 단순 방법은 채택하지 않는다. 자료는 대회 파생물이므로 Git에 넣지 않는다. 완전한 시공간 동시 조정, 별도 모델 개선 및 기존 데이터와 다른 시점의 검증은 아직 하지 않았다.
