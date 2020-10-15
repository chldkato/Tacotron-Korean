# Tacotron Korean TTS

### Training

1. **한국어 음성 데이터 다운로드**

    * [KSS](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset)

2. **`~/Tacotron-Korean`에 학습 데이터 준비**

   ```
   Tacotron-Korean
     |- kss
         |- 1
         |- 2
         |- 3
         |- 4
         |- transcript.v.1.x.txt
   ```

3. **Preprocess**
   ```
   python preprocess.py
   ```
     * data 폴더에 학습에 필요한 파일들이 생성됩니다

4. **Train**
   ```
   python train1.py
   python train2.py
   ```
     * train1.py - train2.py 순으로 실행합니다

   재학습 시
   ```
   python train1.py --step 100000
   ```
     * 불러올 step에 해당하는 숫자를 변경합니다 (train2.py도 동일)

5. **Synthesize**
   ```
   python test1.py --step 100000
   python test2.py --step 100000
   ```
     * test1.py - test2.py 순으로 실행하면 output 폴더에 wav 파일이 생성됩니다



윈도우에서 Tacotron 한국어 TTS 학습하기
  * https://chldkato.tistory.com/141
  
Tacotron 정리
  * https://chldkato.tistory.com/143
