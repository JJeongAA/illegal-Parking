서울시 불법주정차 단속 데이터(2021.9.29.부터 2024.3.13까지) + 불법주정차 단속카메라 위치 데이터

<img width="695" height="216" alt="image" src="https://github.com/user-attachments/assets/31fe4e4c-4b68-4c86-8f03-e0e96a6f64a0" />



현장구분: 불법주정차구역인 데이터만 사용하기 (전처리된 파일 올려드리겠습니다.->불법주정차카메라위치파일_데이터처리본)


드라이브 링크: https://drive.google.com/drive/folders/1T_IUpbJ_JP0ElSQRkEOVAZGiZrOX0QGZ?usp=drive_link
-> 여기서 다운받으시면 됩니다.

(1) baseline_convlstm.py 
(2) baseline_lstm.py
(3) convlstm_batch.py
==> 서울시 관련 학습 모델 (3레이어: 불법주정차단속현황, 불법주정차단속카메라위치, 서울시POI)

(1) h_convlstm.py
(2) h_lstm.py
(3) h_convlstm_Batch.py
==> 서울-해운대 전이학습 모델

실험결과폴더(서울시)
(1) convlstm_improved_no_bn_results: baseline_convlstm 결과
(2) lstm_3channel_results: baseline_lstm 결과
(3) convlstm_improved_results: convlstm 결과

실험결과폴더(해운대)
(1) haeundae_transfer_no_bn_3ch_results: h_convlstm 결과
(2) haeundae_lstm_transfer_poi_results: h_lstm 결과
(3) haeundae_transfer_bn_3ch_results:h_convlstm_Batch 결과

