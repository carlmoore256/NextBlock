# test performance of NextBlock model

import tensorflow as tf
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-model", type=str, default='', help="path to existing model to load")
parser.add_argument("-dataset", type=str, default='E:/Datasets/VoxVerified', help="path to dataset")


