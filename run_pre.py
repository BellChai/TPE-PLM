import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from data_provider.data_factory import data_provider
import time
import numpy as np
import os

from params_load import *
from utils.tools import EarlyStopping, load_content, vali_graph
from models.STGLLM import PredictModel

def pre_setting(args):
    setting = '{}-{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.filename,
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, 1)
    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    return path


def check_nan(module, input, output):
    # print(type(output),output)
    if torch.isnan(output).any() or torch.isinf(output).any():
        print(f"{module.__class__.__name__} output contains NaN!")

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    else:
        print(f"init_weights: {m.__class__.__name__} not init!")
        

def test_epoch(args, model, test_data, test_loader, device, show_step=True, epoch=0, time_start=0):
    test_loss, test_mae_loss, test_mape_loss = vali_graph(args, model, test_data, test_loader, args.scaler_loss, device=device, 
                        show_step=False, epoch=epoch, time_start=time_start)
    save_txt = f' Test RMSE Loss: {test_loss:.7f} MAE Loss: {test_mae_loss:.7f} MAPE Loss: {test_mape_loss:.7f}'
    print(save_txt)
    return save_txt, test_loss, test_mae_loss, test_mape_loss

def train_on_data(args, dataset_name, seed, model_para_path, mode='pretrain', save=False, current_file_name='', config_path='scripts/PEMS.conf'):
    model_para_path = f'save/{model_para_path}'
    
    # pre setting
    print(f'train_on_data config_path: {config_path}')
    args = load_data_args(args_=args, dataset_name=dataset_name, config_path=config_path)
    device = args.device

    args.plugin_mode = 0 if 'plugin=0' in mode else 1
    args.filter_mode = 0 if 'filter=0' in mode else 'one'
    args.mode_att_mode = 0 if 'mode_att=0' in mode else 1
    args.Q_bias = 0.0
    args.load = True if 'load' in mode else False
    # if 'Q_bias=' in mode:
    #     tmp = mode.split(' ')
    #     for t in tmp:
    #         if 'Q_bias=' in t:
    #             args.Q_bias = float(t.split('=')[-1])

    args.cluster = int(mode.split('@')[-1].split('=')[-1])
    if 'llm=llm' in mode:
        args.llm_mode = 'llm'
    elif 'llm=mlp' in mode:
        args.llm_mode = 'mlp'
    elif 'llm=gcn' in mode:
        args.llm_mode = 'gcn'
    elif 'llm=avwgcn' in mode:
        args.llm_mode = 'avwgcn'
    elif 'llm=none' in mode:
        args.llm_mode = 'none'
    elif 'llm=transformer' in mode:
        args.llm_mode = 'transformer'
    else:
        raise ValueError('llm_mode error!')

    for k,v in sorted(vars(args).items()):
        print(k,'=',v)

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    
    # save result
    time_start = time.time()
    if not os.path.exists('results/'): os.makedirs('results')
    if not os.path.exists('save/'): os.makedirs('save')
    fname = f'results/{time_start//1}_{args.model_comment} llm={args.llm_mode} plugin={args.plugin_mode} Mode_att={args.mode_att_mode} ' + \
           f'@cluster={args.cluster}' +  f'_{args.llm_model}.txt'
    if args.load:
        fname = f'results/{time_start//1}_{model_para_path[10:12]}->{args.model_comment[-2:]} llm={args.llm_mode} plugin={args.plugin_mode} Mode_att={args.mode_att_mode} ' + \
                f'@cluster={args.cluster}' +  f'_{args.llm_model}.txt'
    f = open(fname, 'wb', buffering=0)
    fname_current = f'results/current' + current_file_name + '.txt'
    f_current = open(fname_current, 'wb', buffering=0)
    print(f'save result in {fname}\n')
    save_txt = f'dataset = {dataset_name}\nseed = {seed}\n'
    save_txt = bytes(save_txt, encoding='utf-8')
    f.write(save_txt)
    f_current.write(save_txt)

    # pretrain supports
    if 'PEMS0' in dataset_name and 'miss' not in args.data:
        args.supports = train_data.supports.to(device).to(torch.float)
    else:
        args.supports = None
    
    model = PredictModel(args)
    model.to(device)

    print(f'\n<{mode}> on data {dataset_name}\n')
    path = pre_setting(args)
    # args.content = load_content(args)
    if not os.path.exists(path): os.makedirs(path)
    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience, delta=0.001)

    trained_parameters, llm_parameters = [], []
    for name, param in model.named_parameters():
        if param.requires_grad is False: continue
        if 'llm_model' in name:
            llm_parameters.append(param)
        else:
            trained_parameters.append(param)

    if 'llm=transformer' in mode: learning_rate = args.learning_rate * 0.1
    else: learning_rate = args.learning_rate

    model_optim = optim.Adam([
        {'params':trained_parameters, 'lr':learning_rate}, 
        {'params':llm_parameters, 'lr':learning_rate}])                                 # idea weight_decay x

    scheduler = lr_scheduler.ReduceLROnPlateau(model_optim, mode='min', factor=0.6, patience=2, verbose=False, threshold=args.scheduler_threshold)  # idea factor5
    criterion = torch.nn.HuberLoss()

    mse_metric = nn.MSELoss()

    vali_best_loss = -1
    save_txt_best = ''
    vali_loss_dict = {'JiNan':100, 'PEMS03': 100, 'PEMS04': 100, 'PEMS07': 100, 'PEMS08': 100, 'PEMS-BAY': 1000, 'METR-LA': 1000, 'PEMSD7M': 1000, 'PEMSD7L': 1000}
    test_loss = None

    if args.load:
        unwanted_keys = ["plugin2.time_val", "plugin2.ln_t.weight", "plugin2.ln_t.bias", "plugin2.ln_q_map.weight", "plugin2.ln_q_map.bias", "plugin2.att_w.FC_Q.mlp.weight", "plugin2.att_w.FC_Q.mlp.bias", "plugin2.att_w.FC_K.mlp.weight", "plugin2.att_w.FC_K.mlp.bias", "plugin2.att_w.FC_V.mlp.weight", "plugin2.att_w.FC_V.mlp.bias", "plugin2.att_w.out_proj.mlp.weight", "plugin2.att_w.out_proj.mlp.bias", "plugin2.ln_ps.weight", "plugin2.ln_ps.bias", "plugin2.ln_p.weight", "plugin2.ln_p.bias", "mode_att.att_weight", "mode_att.att_bias", "filter_plugin.node_emb1", "filter_plugin.node_emb2"]
        filtered_state_dict = {k: v for k, v in torch.load(model_para_path).items() if k not in unwanted_keys}
        model.load_state_dict(filtered_state_dict)
        print(f'load model from {model_para_path}')
        args.train_epochs = 0

    total_pytorch_params = sum(p.numel() for p in model.parameters())
    trainable_pytorch_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("PyTorch 模型:")
    print(f"总参数数量: {total_pytorch_params}")
    print(f"可训练的参数数量: {trainable_pytorch_params}")
    print(f"PLM参数量: {sum(p.numel() for p in model.llm_model.parameters())} / {sum(p.numel() for p in model.llm_model.parameters() if p.requires_grad)}")
    # print(f"filter_layer参数量: {sum(p.numel() for p in model.filter_layer.parameters())} / {sum(p.numel() for p in model.filter_layer.parameters() if p.requires_grad)}")
    print(f"plugin参数量: {sum(p.numel() for p in model.plugin.parameters())} / {sum(p.numel() for p in model.plugin.parameters() if p.requires_grad)}")
    print(f"Mode_att参数量: {sum(p.numel() for p in model.mode_att.parameters())} / {sum(p.numel() for p in model.mode_att.parameters() if p.requires_grad)}")

    epoch = 0
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            outputs, (dec_out1, dec_out2) = model(batch_x, batch_x_mark, batch_y_mark)

            pred = outputs
            true = batch_y
            dec_pred1, dec_pred2 = dec_out1, dec_out2

            loss = criterion(pred, true).mean()
            dec_loss = criterion(dec_pred1, true) if args.mode_att_mode!=0 else 0
            
            loss_two = loss + dec_loss
            loss_two.requires_grad_(True)
            train_loss.append(mse_metric(pred, true).item())
            if (i + 1) % 500 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            
            loss_two.backward()

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         if torch.isnan(param.grad).any():
            #             print(f"NaN梯度出现在: {name}")

            if 'llm=transformer' in mode: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            model_optim.step()
        
        model.eval()
        # print(f'x_mean: {model.config.train_mean}, x_std: {model.config.train_std}')
        train_epoch_time = time.time() - epoch_time
        print("Epoch: {}, train cost time: {}".format(epoch + 1, train_epoch_time))
        train_loss = np.sqrt(np.average(train_loss))
        epoch_time = time.time()
        vali_loss, vali_mae_loss, vali_mape_loss = vali_graph(args, model, vali_data, vali_loader, args.scaler_loss, device=device)
        vali_epoch_time = time.time() - epoch_time
        print("vali cost time: {}".format(vali_epoch_time))
        save_txt = "Epoch: {0} | train time: {3}, vali time: {4}\n\tTrain Loss: {1:.7f} Vali Loss: {2:.7f} ".format(
                            epoch + 1, train_loss, vali_loss, train_epoch_time, vali_epoch_time)
        
        if (vali_best_loss == -1 or vali_loss < vali_best_loss*0.9995) and (vali_loss < vali_loss_dict[dataset_name]) :
            test_loss, test_mae_loss, test_mape_loss = vali_graph(args, model, test_data, test_loader, args.scaler_loss, device=device, 
                        show_step=False, epoch=epoch, time_start=time_start)
            save_txt += f' Test RMSE Loss: {test_loss:.7f} MAE Loss: {test_mae_loss:.7f} MAPE Loss: {test_mape_loss:.7f}'
            vali_best_loss = vali_loss
            save_txt_best = save_txt + '\n'
            # save_txt += '  saved!'
            if save:
                print('model saved now!\n')
                torch.save(model.state_dict(), model_para_path)
        save_txt += f'\t\t{model_optim.param_groups[0]["lr"]}\n'
        print(save_txt)

        save_txt = bytes(save_txt, encoding='utf-8')
        f.write(save_txt)
        f_current.write(save_txt)

        if test_loss is not None: early_stopping(vali_loss, vali_mae_loss, vali_mape_loss, model, path, train_loss, test_loss, test_mae_loss, test_mape_loss)
        else: early_stopping(vali_loss, vali_mae_loss, vali_mape_loss, model, path, train_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        scheduler.step(train_loss)
        if model_optim.param_groups[0]["lr"]<8.39808e-05: break
        if 'PEMS03' in dataset_name and model_optim.param_groups[0]["lr"]<0.002: break

    if args.load:
        test_loss, test_mae_loss, test_mape_loss = vali_graph(args, model, test_data, test_loader, args.scaler_loss, device=device, 
                        show_step=False, epoch=epoch, time_start=time_start)
        save_txt = f' Test RMSE Loss: {test_loss:.7f} MAE Loss: {test_mae_loss:.7f} MAPE Loss: {test_mape_loss:.7f}'
        print(save_txt)
        save_txt = bytes(save_txt+'\n\n', encoding='utf-8')
        f.write(save_txt)
        f_current.write(save_txt)

    time_end = time.time()
    time_all = (time_end - time_start)/60
    time_each = time_all / (epoch+1)

    save_txt = f'{time.asctime()}, time_all = {time_all:.1f} min, time_each = {time_each:.1f} min\n\n{save_txt_best}'
    nums = save_txt.split(' ')
    simple_txt = '\n' + f'{nums[-7]}\t{nums[-4]}\t{nums[-1]}' + '\n'
    save_txt = bytes(save_txt + simple_txt, encoding='utf-8')
    f.write(save_txt)
    f.close()
    f_current.write(save_txt)
    f_current.close()

    print(simple_txt)

    # model param save
