import logging
import os
import time
from collections import defaultdict
from typing import List, Union, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import psutil
from pathlib import Path
import torchvision
from torchvision import datasets, models, transforms
import torch
from ..utils import s3_utils

logger = logging.getLogger(__name__)


def load_train(path: str) -> str:
    
    return  torch.load(os.path.join(str(path),"train")) 

def load_val(path: str) -> str:
    
    return  torch.load(os.path.join(str(path),"val")) 

def load_test(path: str) -> str:
    
    return  torch.load(os.path.join(str(path),"test")) 