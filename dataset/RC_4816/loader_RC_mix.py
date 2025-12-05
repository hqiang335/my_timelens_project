import torch
import numpy as np
import os
from tools.registery import DATASET_REGISTRY
from dataset.RC_4816.mixloader import MixLoader
# from dataset.RC_4816.x2loader import x2Loader





@DATASET_REGISTRY.register()
class mix_loader_smallRC(MixLoader):
    def __init__(self, para, training=True):
        super().__init__(para, training)
# class mix_loader_smallRC(x2Loader):
#     def __init__(self, para, training=True):
#         super().__init__(para, training)