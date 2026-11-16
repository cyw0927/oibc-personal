# #1314 랜덤 포레스트 그룹 모델 계획

부스팅이나 신경망 대신 서로 다른 표본과 변수로 학습한 나무 80개의 평균을 사용한다. RandomForestRegressor(n_estimators=80, min_samples_leaf=25, max_features=0.8, max_depth=16, random_state=42). 결측은 각 학습 구간의 중앙값으로 대체한다. 기존 coverage×과거 SMP×k 가중치 및 바탕 40%·그룹 60%를 유지한다. 야간 0 및 출력 [0,1] 제한을 적용한다. 비율이나 설정 탐색을 하지 않는다.

동일 코호트·실적·키·가중치의 JUL/OCT/JAN/FEB walk-forward 비교다. parquet row group의 최대 시각을 읽기 전 검사해 잠금 구간과 교차하면 중단한다. 4개 모두 반복 사용 DEV, 7·10월은 부분 코호트, 1·2월 기준은 원본 v2 OOF 미대조 재학습본이다. 겨울 특성의 과거 생성 당시 잠금 혼합 row group 접근 가능성 및 예보 실제 수신시각 미확인. README 잠정(D05 HOLD), official_S=null이며 데이터·파생물 Git 금지.
