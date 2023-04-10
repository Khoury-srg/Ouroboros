import importlib
from numpy.core.numeric import Inf
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from NNet.utils.writeNNet import writeNNet
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
import time
import importlib
import models
import datasets
import math
import utils
from utils import *
import tasks


importlib.reload(datasets)
importlib.reload(tasks)
def RealSys_DB(time_out):
    N = 10000000
    v2_num = 10
    # tasks.FBDBIndexTask(task_index=0, v2_num=v2_num, batch_size=N//10, time_out = time_out).train_and_verify() # normal
    # tasks.FBDBIndexTask(task_index=0, v2_num=v2_num, batch_size=N//10, add_counterexample = True, incremental_training = True, batch_counterexample = True, time_out=time_out).train_and_verify() # spec-data        
    
    for task_index in range(1, v2_num+1):
        tasks.FBDBIndexTask(task_index=task_index, v2_num=v2_num, batch_size=N//(v2_num*10), time_out = time_out).train_and_verify() # normal
        tasks.FBDBIndexTask(task_index=task_index, v2_num=v2_num, batch_size=N//(v2_num*10), add_counterexample = True, incremental_training = True, batch_counterexample = True, time_out=time_out).train_and_verify() # spec-data        
    merge_db_results(time_out)

RealSys_DB(1800)