import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
import os

from tqdm import tqdm

plt.switch_backend('agg')


class LRadjust:
    def __init__(self):
        self.best_test_loss = None

    def adjust_learning_rate(self, optimizer, epoch, args, printout=True, test_loss=None, lr=None):
        if (not test_loss):
            return
        if (not self.best_test_loss) or (test_loss < self.best_test_loss - 0.03):
            self.best_test_loss = test_loss
            return
        
        print(f'best test_loss:{self.best_test_loss},  current test_loss:{test_loss}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.6
        if printout:
            print('Updating learning rate to {}'.format(lr * 0.6))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_vali_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

        self.train_loss = None
        self.test_loss = None
        self.test_mae_loss = None
        self.test_mape_loss = None

    def __call__(self, vali_loss, vali_mae_loss, vali_mape_loss, model, path, train_loss, test_loss=None, test_mae_loss=None, test_mape_loss=None):
        score = -vali_loss 
        if self.best_score is None:
            self.best_score = score
            self.best_vali_loss = vali_loss
            self.train_loss = train_loss
            self.test_loss = test_loss
            self.test_mae_loss = test_mae_loss
            self.test_mape_loss = test_mape_loss
            if self.save_mode:
                self.save_checkpoint(vali_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                save_txt = f"best record -  Train Loss: {self.train_loss:.7f} Vali Loss: {-self.best_vali_loss:.7f} Test RMSE Loss: {self.test_loss:.7f} MAE Loss: {self.test_mae_loss:.7f} MAPE Loss: {self.test_mape_loss:.7f}\n"
                print(save_txt)
        else:
            self.best_score = score
            self.best_vali_loss = vali_loss
            self.train_loss = train_loss
            self.test_loss = test_loss
            self.test_mae_loss = test_mae_loss
            self.test_mape_loss = test_mape_loss
            if self.save_mode:
                self.save_checkpoint(vali_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path)

def masked_mae_torch(pred, true, mask_value):
    if mask_value != None:
        mask = true > mask_value
        true = true[mask]
        pred = pred[mask]
    MAE = torch.mean(torch.absolute(pred-true))
    return MAE

def masked_rmse_torch(pred, true, mask_value):
    if mask_value != None:
        mask = true > mask_value
        true = true[mask]
        pred = pred[mask]
    RMSE = torch.mean((pred-true)*(pred-true)).sqrt()
    return RMSE

def masked_mape_torch(pred, true, mask_value):
    if mask_value != None:
        mask = true > mask_value
        true = true[mask]
        pred = pred[mask]
    MAPE = torch.mean(torch.absolute(torch.divide((true - pred), true)))
    return MAPE

def All_Metrics(pred, true, mask_value):
    rmse = masked_rmse_torch(pred, true, mask_value=mask_value)
    mae = masked_mae_torch(pred, true, mask_value=mask_value)
    mape = masked_mape_torch(pred, true, mask_value=mask_value) * 100
    return rmse, mae, mape


def vali_graph(args, model, vali_data, vali_loader, scaler_loss, device=None, 
                show_step=False, mask_value=0.001, epoch=0, time_start=0):
    model.eval()
    y_pred = []
    y_true = []
    x_true = []
    x_mark = []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float().to(device)                                        # [B, pre_len, N]
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # encoder - decoder
            outputs, _ = model(batch_x, batch_x_mark, batch_y_mark)
            pred = outputs.detach()
            true = batch_y.detach()

            if scaler_loss==0:
                pred = vali_data.inverse_transform(pred)
                true = vali_data.inverse_transform(true)
            
            x_true.append(batch_x)
            x_mark.append(batch_x_mark)
            y_pred.append(pred)
            y_true.append(true)
    
    x_true = torch.cat(x_true, dim=0)
    x_mark = torch.cat(x_mark, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    if show_step:
        path = f'res_save/{time_start//1}'
        if not os.path.exists(path): os.makedirs(path)
        torch.save(x_mark, f'{path}/{args.model_comment}_epoch={epoch+1}_xmark.pt')
        torch.save(x_true, f'{path}/{args.model_comment}_epoch={epoch+1}_xtrue.pt')
        torch.save(y_pred, f'{path}/{args.model_comment}_epoch={epoch+1}_pred.pt')
        torch.save(y_true, f'{path}/{args.model_comment}_epoch={epoch+1}_true.pt')
        torch.save(model.mode_att.state_dict(), f'{path}/{args.model_comment}_epoch={epoch+1}_mode_att.pth')
    rmse, mae, mape = All_Metrics(y_pred, y_true, mask_value=mask_value)

    model.train()
    return rmse, mae, mape


def load_content(args):
    with open(f'./dataset/prompt_bank/{args.data}.txt', 'r') as f:
        content = f.read()
    return content
