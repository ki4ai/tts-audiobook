# tts-audiobook

버전: 0.95
작성자: 박세직
히스토리:
2020/11/23, 초안작성

***

#### Note

* (2020/11/23) 11월 마스터 버전이 업데이트 되었습니다.

***

#### System/SW Overview

* 개발목표: 감정이 들어간 자연스러운 발화가 가능한 음성합성모델
* 최종 결과물: Tacotron2를 기초로 하는 TTS base mel spectrogram 합성 후 VocGAN을 통한 음성합성

***

#### How to Install

* pip install -r requirements.txt -r requirements-local.txt

***

#### (필수) Main requirement

* OS: Ubuntu 18.04 or 16.04
* Container : Docker community 19.03, Nvidia-docker v2.*
* GPU driver : Nvidia CUDA 10.2
* 파이썬 : Python3.7.5
* 프레임워크: Pytorch 1.5
* 이 외 requirements.txt 참조

***

#### (필수) Network Architecture and features

* **Model**
1. Tacotron2를 통한 txt2spectrogram 합성
2. VocGAN을 통한 spectrogram2wav 합성
* **Tacotron2 구조**
1. CNN과 biLSTM으로 이루어진 Encoder
2. Location Sensitive Attention
3. 2개의 LSTM으로 이루어진 Decoder 
4. 감정, 화자, 언어를 control 할 수 있는 embedding
5. prosody를 control 할 수 있는 GST
* **VocGAN 구조**

***

#### (필수) Quick start

* Step1. GPU Version - 호스트 머신에서 아래 명령을 실행한다. 
```
export CUDA_VISIBLE_DEVICES=<GPUID>
python flask_server.py [--port <PORT>]: 통합버전용
python flask_server_test.py [--port <PORT>]: 개별 모델 테스트용
```

* Step2. (POST 방법 참조) GPU Version - 클라이언트 머신(예제는 호스트와 동일)에서 아래 명령을 실행한다. 
```
통합버전용: client.py
M1.4 의 TTS_INPUT 데이터셋을 활용하여 TTS 를 사용합니다.
개별 모델 테스트용:
python client_test_emotion.py [--port <PORT>]
또는 python client_test_speaker.py [--port <PORT>]
해당 실행은 sentences 내부의 있는 문장을 random sampling 하여 문장을 생성합니다.
합성된 음성은 M2.4 아래에 생성이 됩니다.
```
```
주요 라인
r = s.post(ip_address, data={'sentence': cur[0], 'speaker':'{}'.format(cur[1]), 'emotion': '{}'.format(cur[2]), 'intensity': 1, 'result_folder': 'emotion/'})
설명
sentence: 문장 정보, speaker: 화자 정보, emotion: 감정 정보, result_folder: 저장 위치
```
