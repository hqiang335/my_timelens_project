import os
from os.path import join, split, splitext
from tools import parse_path
import socket
from easydict import EasyDict as ED
import datetime


mkdir = lambda x:os.makedirs(x, exist_ok=True)

mydata = ED()
mydata.train = ED()
mydata.train.dense_rgb = 'mydata/scene1/aps_png' 
# mydata.train.rgb = 'TimeLens-XL\mydata\test\scene1\aps_png' 
# mydata.train.evs = r'TimeLens-XL\mydata\test\scene1\events' 
mydata.train.evs_refid = 'mydata/scene1/events' 

mydata.test = ED()
mydata.test.dense_rgb = 'mydata/scene1/aps_png' 
# mydata.test.rgb = 'TimeLens-XL\mydata\test\scene1\aps_png' 
# mydata.test.evs = 'mydata/test/scene1/events' 
mydata.test.evs_refid = 'mydata/scene1/events' 
