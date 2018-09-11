import argparse

parser = argparse.ArgumentParser(description="Argument Setting")
parser.add_argument("--LEARNING_RATE", default = 0.001)
parser.add_argument("--DISCOUNT_FACTOR", default = 0.99)
parser.add_argument("--FORWARD_MAX", default = 20)
parser.add_argument("--NUM_WORKERS", default = 1)
args = parser.parse_args()
