import torch
import time
import random
import numpy as np
import os

from params_load import *

# os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
from run_pre import train_on_data

def set_seed():
    fix_seed = int(time.time()*1000)%1000
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return fix_seed

args = parse_args(dataset_name='PEMS08', config_path='scripts/PEMS.conf')
device = args.device

def model_eval():
    seed = set_seed()

    train_on_data(args=args, dataset_name='PEMS04', seed=seed, model_para_path='gpt2.pth', \
        mode='llm=llm plugin=1 mode_att=1 @cluster=4', save=False, config_path='scripts/PEMS.conf')
    train_on_data(args=args, dataset_name='PEMS03', seed=seed, model_para_path='gpt2.pth', \
        mode='llm=llm plugin=1 mode_att=1 @cluster=6', save=False, config_path='scripts/PEMS.conf')
    train_on_data(args=args, dataset_name='PEMS07', seed=seed, model_para_path='gpt2.pth', \
        mode='llm=llm plugin=1 mode_att=1 @cluster=6', save=False, config_path='scripts/PEMS.conf')


model_eval()

