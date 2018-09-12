
# Reference Codes
# 김태훈 님의 코드를 참조하였습니다.  https://github.com/carpedm20/a3c-tensorflow/
# https://github.com/hongzimao/a3c
# https://github.com/hiwonjoon/tf-a3c-gpu
# https://github.com/carpedm20/a3c-tensorflow
# https://github.com/stefanbo92/A3C-Continuous
# 감사합니다.



import argparse

parser = argparse.ArgumentParser(description="Argument Setting")
parser.add_argument("--LEARNING_RATE", default = 0.001)
parser.add_argument("--DISCOUNT_FACTOR", default = 0.99)
parser.add_argument("--FORWARD_MAX", default = 20)
parser.add_argument("--NUM_WORKERS", default = 1)
args = parser.parse_args()
