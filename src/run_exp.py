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

def training_DB(time_out):
    v2_num=10
    for task_index in range(v2_num+1):
        tasks.DBIndexTask(task_index=task_index, v2_num=v2_num, time_out = time_out).train_and_verify() # normal
        tasks.DBIndexTask(task_index=task_index, v2_num=v2_num, add_counterexample = False, incremental_training = True, time_out = time_out).train_and_verify() # normal
        tasks.DBIndexTask(task_index=task_index, v2_num=v2_num, add_counterexample = True, incremental_training = True, time_out=time_out).train_and_verify() # vanilla
        tasks.DBIndexTask(task_index=task_index, v2_num=v2_num, add_counterexample = True, incremental_training = True, batch_counterexample = True, time_out=time_out).train_and_verify() # spec-data
        tasks.DBIndexTask(task_index=task_index, v2_num=v2_num, add_counterexample = True, incremental_training = True, batch_counterexample = True, early_rejection = True, time_out=time_out).train_and_verify() # early rejection
    merge_db_results(time_out)

def training_all():
    task_classes = [
                    (tasks.CardWikiTask, 300),   
                    (tasks.RedisTask, 300), 
                    (tasks.BloomCrimeTask, 500), 
                    (tasks.LinnosTask, 300), 
                    ]
    for task_class, time_out in task_classes:
        print("task_class: ", task_class, "normal")
        task_class(time_out = time_out).train_and_verify() # normal
        print("task_class: ", task_class, "vanilla")
        task_class(add_counterexample = True, incremental_training = True, time_out=time_out).train_and_verify() # vanilla
        print("task_class: ", task_class, "spec data")
        task_class(add_counterexample = True, incremental_training = True, batch_counterexample = True, time_out=time_out).train_and_verify() # spec data
        print("task_class: ", task_class, "early reject")
        task_class(add_counterexample = True, incremental_training = True, batch_counterexample = True, early_rejection = True, time_out=time_out).train_and_verify() # w/ early rejection
        
training_DB(300)
training_all()
