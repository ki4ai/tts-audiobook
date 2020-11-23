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
* 최종 결과물: Tacotron2 base mel spectrogram 합성 후 VocGAN을 통한 음성합성

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
python flask_server.py [--port <PORT>]
```
