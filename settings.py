import locale
import os
import platform


# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print(BASE_DIR) # c:\projecth\example\rltrader-master

# 로케일 설정
if 'Linux' in platform.system() or 'Darwin' in platform.system():
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
elif 'Windows' in platform.system():
    locale.setlocale(locale.LC_ALL, '')
    # print(locale.setlocale(locale.LC_ALL, '')) # Korean_Korea.949