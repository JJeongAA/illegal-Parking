# illegal-Parking
https://colab.research.google.com/drive/1efzSm4qmqN5Ti2KvKgUbbncIpIndi4nm?usp=drive_link
<br />
<br />
원본 데이터 행 수 : 5178626  
증강 데이터 행 수 : 20714476  
  
원본 window 개수 : 897  
증강 window 개수 : 3588  
  
aug_type  
original        5178619  
time_shift      5178619  
jitter          5178619  
scaled_slice    5178619  
Name: count, dtype: int64  
  
✅ 1. KS-test  
KS-test 통계량: 0.0    
KS-test p-value: 1.0  
- 결론: 분포가 완벽하게 동일하다.
  
✅ 2. Shapiro-Wilk 정규성 검정  
원본 Shapiro p: 3.3e-07  
증강 Shapiro p: 9.7e-18  
- p-value가 모두 0.05보다 훨씬 작음 ⇒ 두 데이터 모두 정규분포는 아님.  
- 하지만 둘 다 “비정규”이므로 서로 비교하는 데엔 문제 없음.  
  
✅ 3. Levene Test (등분산 검정)  
p-value: 1.0  
- 원본 vs 증강 분산이 100% 동일하다.  
  
✅ 4. t-test (평균 비교)  
p-value: 1.0  
- 평균 차이가 전혀 없음.  
- 원본 데이터의 “중심값(평균)”과 증강 데이터의 “중심값”이 완벽하게 동일하게 유지됨.  
  
✅ 5. Cohen’s d  
Cohen's d: 0.0  
- 효과 크기 0 ⇒ 차이가 전혀 없음.  
