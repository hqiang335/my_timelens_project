import os
from os.path import join, split, splitext
from tools import parse_path
import socket
from easydict import EasyDict as ED
import datetime


mkdir = lambda x:os.makedirs(x, exist_ok=True)


# hostname = 'server' if 'PC' not in socket.gethostname() else 'local'

# mydata = ED()
# mydata.train = ED()
# # GOPRO.train.dense_rgb = r'E:\Research\EVS\Dataset\RGB_dense\test_x8interpolated'
# mydata.train.dense_rgb = r'TimeLens-XL\mydata\test\scene1\aps_png' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/GOPRO/RGB_dense/train_x8interpolated/'
# mydata.train.rgb = 'TimeLens-XL\mydata\test\scene1\aps_png' if hostname == 'local' else '/mnt/data/oss_beijing/mayongrui/dataset/GOPRO_Large_all/RGB/train'
# # Events in IMX 636 params, ct=0.25, ct_variance = 0.06
# mydata.train.evs = r'TimeLens-XL\mydata\test\scene1\events' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/GOPRO/Events_v2eIMX636v230809/train_x8interpolated/'
# # Events in REFID params, ct=0.2, ct_variance=0.03
# mydata.train.evs_refid = r'TimeLens-XL\mydata\test\scene1\events' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/GOPRO/EVS/train_x8interpolatedct25v006/EVS'

# mydata.test = ED()
# mydata.test.dense_rgb = r'TimeLens-XL\mydata\test\scene1\aps_png' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/GOPRO/RGB_dense/test_x8interpolated/'
# mydata.test.rgb = 'TimeLens-XL\mydata\test\scene1\aps_png' if hostname == 'local' else '/mnt/data/oss_beijing/mayongrui/dataset/GOPRO_Large_all/RGB/test'
# # Events in IMX 636 params, ct=0.25, ct_variance=0.06
# mydata.test.evs = r'TimeLens-XL\mydata\test\scene1\events' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/GOPRO/Events_v2eIMX636v230809/test_x8interpolated/'
# # Events in REFID params, ct=0.2, ct_variance = 0.03
# mydata.test.evs_refid = r'TimeLens-XL\mydata\test\scene1\events' if hostname == 'local' else '/mnt/workspace/mayongrui/dataset/GOPRO/EVS/test_x8interpolatedct25v006/EVS'

mydata = ED()
mydata.train = ED()
# GOPRO.train.dense_rgb = r'E:\Research\EVS\Dataset\RGB_dense\test_x8interpolated'
mydata.train.dense_rgb = r'TimeLens-XL\mydata\test\scene1\aps_png' 
mydata.train.rgb = 'TimeLens-XL\mydata\test\scene1\aps_png' 
# Events in IMX 636 params, ct=0.25, ct_variance = 0.06
mydata.train.evs = r'TimeLens-XL\mydata\test\scene1\events' 
# Events in REFID params, ct=0.2, ct_variance=0.03
mydata.train.evs_refid = r'TimeLens-XL\mydata\test\scene1\events' 

mydata.test = ED()
mydata.test.dense_rgb = r'TimeLens-XL\mydata\test\scene1\aps_png' 
mydata.test.rgb = 'TimeLens-XL\mydata\test\scene1\aps_png' 
# Events in IMX 636 params, ct=0.25, ct_variance=0.06
mydata.test.evs = r'TimeLens-XL\mydata\test\scene1\events' 
# Events in REFID params, ct=0.2, ct_variance = 0.03
mydata.test.evs_refid = r'TimeLens-XL\mydata\test\scene1\events' 
