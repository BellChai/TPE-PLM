import argparse
# import numpy as np
import configparser
# import pandas as pd
import copy
import torch

def load_data_args(args_, dataset_name, config_path='scripts/PEMS.conf'):
    config_file = config_path
    config = configparser.ConfigParser()
    config.read(config_file)

    print('type(args_):', type(args_))
    args = argparse.Namespace(**{k: copy.deepcopy(v) for k,v in vars(args_).items()})

    args.dataset = dataset_name
    args.node_num = int(config[args.dataset]['node_num'])
    # args.time_num = int(config[args.dataset]['time_num'])
    args.filename = config[args.dataset]['filename']
    args.adj_path = config[args.dataset]['adj_path']
    args.model_id = config[args.dataset]['model_id']
    args.model_comment = config[args.dataset]['model_comment']
    args.data_path = config[args.dataset]['data_path']
    args.batch_size = int(config[args.dataset]['batch_size'])

    args.week_opt = int(config[args.dataset]['week_opt'])
    args.Q_bias = float(config[args.dataset]['Q_bias'])
    args.b1_ = float(config[args.dataset]['b1_'])
    args.b2_ = float(config[args.dataset]['b2_'])
    args.scheduler_threshold = float(config[args.dataset]['scheduler_threshold'])
    # args.num_hidden_layers = int(config['model']['num_hidden_layers'])

    return args

def parse_args(dataset_name='PEMS08', config_path='scripts/PEMS.conf'):
    config = configparser.ConfigParser()
    config.read(config_path)
    # s = ''
    # for i in config:
    #     s += f'{i}:'+'{'
    #     for j in config[i]:
    #         s += f'{j}:{config[i][j]},'
    #     s += '}, '
    # print(s)

    args = argparse.ArgumentParser(prefix_chars='--', description='arguments')
    
    # cyh set
    args.add_argument('--dataset', type=str, default=dataset_name)
    args.add_argument('--scaler_loss', type=int, default=config['model']['scaler_loss'], help="1: time series; 0: st data")
    args.add_argument('--data_type', type=str, default=config['model']['data_type'], help="use graph type?")
    args.add_argument('--dataset_graph', default=config['model']['dataset_graph'].split(','))
    args.add_argument('--gpu', type=int, default='0')

    args.add_argument('--filter_mode', type=str, default=config['model']['filter_mode'])
    args.add_argument('--few_shot', type=float, default=config['model']['few_shot'])


    # basic config
    args.add_argument('--task_name', type=str, default=config['model']['task_name'],
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    args.add_argument('--is_training', type=int, default=config['model']['is_training'], help='status')
    args.add_argument('--model', type=str, default=config['model']['model'],
                        help='model name, options: [Autoformer, DLinear]')
    # data loader
    args.add_argument('--data', type=str, default=config['model']['data'], help='dataset type')
    args.add_argument('--root_path', type=str, default=config['model']['root_path'], help='root path of the data file')
    args.add_argument('--features', type=str, default=config['model']['features'],
                        help='forecasting task, options:[M, S, MS]; '
                            'M:multivariate predict multivariate, S: univariate predict univariate, '
                            'MS:multivariate predict univariate')
    args.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    args.add_argument('--loader', type=str, default='modal', help='dataset type')
    args.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, '
                            'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                            'you can also use more detailed freq like 15min or 3h')
    args.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    args.add_argument('--seq_len', type=int, default=config['model']['seq_len'], help='input sequence length')
    args.add_argument('--label_len', type=int, default=config['model']['label_len'], help='start token length')
    args.add_argument('--pred_len', type=int, default=config['model']['pred_len'], help='prediction sequence length')
    args.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    args.add_argument('--c_out', type=int, default=config['model']['c_out'], help='output size')
    args.add_argument('--d_model', type=int, default=config['model']['d_model'], help='dimension of model')
    args.add_argument('--n_heads', type=int, default=8, help='num of heads')
    args.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    args.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    args.add_argument('--d_ff', type=int, default=config['model']['llm_dim'], help='dimension of fcn')
    args.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    args.add_argument('--factor', type=int, default=config['model']['factor'], help='attn factor')
    # args.add_argument('--dropout', type=float, default=0.1, help='dropout')
    args.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    args.add_argument('--activation', type=str, default='gelu', help='activation')
    args.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    args.add_argument('--patch_len', type=int, default=config['model']['patch_len'], help='patch length')
    args.add_argument('--stride', type=int, default=config['model']['stride'], help='stride')
    args.add_argument('--prompt_domain', type=int, default=0, help='')
    args.add_argument('--llm_model', type=str, default=config['model']['llm_model'], help='LLM model') # LLAMA, GPT2, BERT
    args.add_argument('--llm_dim', type=int, default=config['model']['llm_dim'], help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
    args.add_argument('--num_hidden_layers', type=int, default=config['model']['num_hidden_layers'])
    

    # optimization
    args.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    args.add_argument('--itr', type=int, default=config['model']['itr'], help='experiments times')
    args.add_argument('--train_epochs', type=int, default=config['model']['train_epochs'], help='train epochs')
    args.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
    args.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    args.add_argument('--patience', type=int, default=config['model']['patience'], help='early stopping patience')
    args.add_argument('--learning_rate', type=float, default=config['model']['learning_rate'], help='optimizer learning rate')
    args.add_argument('--des', type=str, default=config['model']['des'], help='exp description')
    args.add_argument('--loss', type=str, default='MSE', help='loss function')
    args.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    args.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    args.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    args.add_argument('--llm_layers', type=int, default=6)
    args.add_argument('--percent', type=int, default=100)
    
    args, _ = args.parse_known_args()

    args.node_num = config[args.dataset]['node_num']
    # args.time_num = config[args.dataset]['time_num']
    args.filename = config[args.dataset]['filename']
    args.adj_path = config[args.dataset]['adj_path']
    args.model_id = config[args.dataset]['model_id']
    args.model_comment = config[args.dataset]['model_comment']
    args.data_path = config[args.dataset]['data_path']
    print(f'\n\n****\nargs.data_path:{args.data_path}\n****\n\n\n')
    args.batch_size = config[args.dataset]['batch_size']

    args.device = torch.device(f'cuda:{args.gpu}') if args.gpu>=0 else torch.device('cpu')

    return args
