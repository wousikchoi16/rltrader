# 파이썬과 케라스를 이용한 딥러닝/강화학습 주식투자 - 퀀트 투자, 알고리즘 트레이딩을 위한 최첨단 해법 입문 (개정판)

[퀀티랩 블로그 관련 포스트](http://blog.quantylab.com/pages/rltrader.html)

강화학습은 스스로 학습하는 머신러닝 기법으로서 주식 데이터 학습에 잘 적용되는 기법입니다. 이 책은 파이썬을 이용한 강화학습 기반의 주식투자 시뮬레이션 프로그램을 개발하는 방법을 설명합니다. 이를 위해 이론과 코드 수준에서 상세한 설명을 덧붙였습니다. 이 책을 통해 딥러닝과 강화학습을 이해하고 이를 주식투자를 비롯한 다양한 도메인에서 활용할 수 있을 것입니다.

## 구매 링크
- [교보문고](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791158392031&orderClick=LEa&Kc=)
- [Yes24](http://www.yes24.com/Product/Goods/89999978?scode=032&OzSrank=1)

![표지](/img/e2.1_introduction.jpg)

## 이 책에서 다루는 내용
- 딥러닝과 강화학습 이론
- 주식투자에 강화학습을 적용하는 법
- 강화학습 기반의 주식투자 시스템 개발
- 강화학습을 위한 실제 주식 데이터 획득 및 처리
- 강화학습으로 주식 데이터를 학습하는 법
- 학습한 강화학습 모델을 활용하는 법
- 강화학습 기반의 주식투자 시스템을 커스터마이징하는 법

## 환경설정
- [Anaconda 3.7+](https://www.anaconda.com/distribution/)
- [TensorFlow 1.15.2](https://www.tensorflow.org/)
  - `pip install tensorflow==1.15.2`
  - `pip install tensorflow-gpu==1.15.2`
- [plaidML](https://plaidml.github.io/plaidml/)
  - `pip install plaidml-keras==0.6.2`
  - `pip install mplfinance`

# conda 환경

## TF 1.15

> TF 1.15 사용을 위해서 Python 3.6을 설치한다.
> TF 1.15 사용할 경우 cuda 10.0, cudnn 7.4.2 (7.3.1) 설치해야 한다.
> https://www.tensorflow.org/install/source#tested_build_configurations
> https://github.com/tensorflow/models/issues/9706

conda create -n rltrader python=3.6
conda activate rltrader
pip install tensorflow-gpu==1.15
conda install cudatoolkit=10.0
conda install cudnn=7.3.1
pip install numpy
pip install pandas

### TF 2.5
https://www.tensorflow.org/install/source_windows?hl=en#gpu

conda create -n rltrader2 python=3.6
conda activate rltrader2
pip install tensorflow==2.5

CUDA 11.2
cuDNN 8.1

PATH
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp

# 프로파일링
- `python -m cProfile -o profile.pstats main.py ...`
- `python profile.py`

## Troubleshooting

### TF 1.15에서 다음 에러가 나면 Python 3.6으로 맞춰준다.
```
NotImplementedError: Cannot convert a symbolic Tensor (lstm/strided_slice:0) to a numpy array.
```

### original_keras_version = f.attrs['keras_version'].decode('utf8') AttributeError: 'str' object has no attribute 'decode'
```
https://github.com/keras-team/keras/issues/14265
https://pypi.org/project/h5py/#history
pip install h5py==2.10.0
```
