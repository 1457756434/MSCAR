import argparse
import os
import torch
from torch.utils.data import DataLoader
import yaml

import copy

from dataset import GRIDSAT_dataset_full_crop
from Model import Transformer_TC_test
import pandas as pd
import numpy as np
import time
import random
def rand_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def MAE(pred, gt):
        """
        MAE metric.

        Parameters
        ----------
        pred: tensor, required, the predicted

        gt: tensor, required, the ground-truth

        Returns
        -------
        
        The MAE metric.
        """
    
        sample_mae = torch.mean(torch.abs(pred - gt), dim=0)
        return sample_mae



def evaluate_batch( data_dict):
    """
    Evaluate a batch of the samples.

    Parameters
    ----------

    data_dict: pred and gt


    Returns
    -------

    The metrics dict.
    """
    pred = data_dict['pred']           
    gt = data_dict['gt']
  
    losses = {}
    
    loss = MAE(pred, gt,)
    if isinstance(loss, torch.Tensor):
        for i in range(len(loss)):
            losses["MAE"+str(i)] = loss[i].item()
    else:
        losses["MAE"] = loss

    return losses


class loss_logger():
    def __init__(self,vnames=2, device="cpu") -> None:
        self.count = torch.zeros((vnames,)).to(device=device)
        self.total = torch.zeros((vnames,)).to(device=device)
    def update(self, n=1,**kwargs):
        i = 0
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))

            self.count[i] += n
            self.total[i] += v * n
            i += 1
    def return_mean(self):
        return self.total/self.count
    


def data_preprocess_include_tcinfo(data, device):
   
    ir_inp=0
    new_era5_data=0
    inp_lable=0
    label=0
    tc_info = None

    if len(data)==4:
        #new_GridSat_data, new_era5_data, inp_lable, label
        assert len(data) == 4, f" data length must be 4"
        
        ir_inp = data[0].to(torch.float32).to(device, non_blocking=True)
        new_era5_data = data[1].to(torch.float32).to(device, non_blocking=True)
        inp_lable = data[-2].to(torch.float32).to(device, non_blocking=True)
        label = data[-1].to(torch.float32).to(device, non_blocking=True)
    elif len(data)==5:
        #new_GridSat_data, new_era5_data, inp_lable, label
        assert len(data) == 5, f" data length must be 5"
        ir_inp = data[0].to(torch.float32).to(device, non_blocking=True)
        new_era5_data = data[1].to(torch.float32).to(device, non_blocking=True)
        inp_lable = data[-3].to(torch.float32).to(device, non_blocking=True)
        label = data[-2].to(torch.float32).to(device, non_blocking=True)
        tc_info = data[-1]
    else:
        raise ValueError("data size error")
    return ir_inp, new_era5_data, inp_lable, label, tc_info

# def cmp_effort():
#     from torchinfo import summary
#     import torchvision

#     net = Transformer_TC_test.Multiscale_STAR_ViT_TC_IR_test()
#     input_size = [(1, 4, 1, 140, 140), (1, 4, 69, 40, 40), (1, 4, 2)]
#     a = torch.randn((1, 4, 1, 140, 140))
#     b = torch.randn((1, 4, 69, 40, 40))
#     c = torch.randn((1, 4, 2))
#     summary(net, input_size)

def main(args):
    rand_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_dataset = GRIDSAT_dataset_full_crop.GRIDSAT_crop_dataset(split='valid', data_dir=args.data_dir)

    test_dataLoader = DataLoader(
            test_dataset,
            batch_size = 1,
            shuffle=False,
            drop_last=False,
            
        )
    data_mean, data_std = test_dataset.get_meanstd()
    data_mean, data_std = data_mean.to(torch.float32).to(device), data_std.to(torch.float32).to(device)
    model = Transformer_TC_test.Multiscale_STAR_ViT_TC_IR_test()
    #checkpoint path
    #path = "E:\\MSCAR\check_point\\checkpoint_best.pth"  
    path = args.checkpoint_path

    model_dict = torch.load(path)
    model.load_state_dict(model_dict["model"]["mstar_vit_tc_gridsa_test"])
    model.to(device=device)
    model.eval()
    # print(model)
    
    total_step = len(test_dataLoader)
    metric_logger = []
    pre_len = 4
    for i in range(pre_len):
        metric_logger.append(loss_logger(vnames=2, device=device))
    for step, batch_data in enumerate(test_dataLoader):
        metrics_losses=[]
        ir_inp, new_era5_data, inp_lable, label, tc_info = data_preprocess_include_tcinfo(batch_data, device)
        with torch.no_grad():
            predict = []
            T_all = label.shape[1]
            starttime = time.time()
            for t in range(T_all):
                predict_i = model(ir_inp, new_era5_data, inp_lable, t+1)
                predict.append(predict_i)
            endtime = time.time()
            print('Time cost of a batch: {:.5f} s'.format(endtime-starttime))
            predict = torch.cat(predict, dim=1)

            for i in range(T_all):
                data_dict = {}
                data_dict['gt'] = label[:, i] * data_std + data_mean
                data_dict['pred'] = predict[:, i] * data_std + data_mean
            
                metrics_losses.append(evaluate_batch(data_dict))
            
            for i in range(len(metrics_losses)):
                metric_logger[i].update(**metrics_losses[i])
            # index += batch_len

            print("#"*80)
            print(step)
            if step % 10 == 0 or step == total_step-1:
                for i in range(T_all):
                    print('  '.join(
                            [f'final valid {i}th step predict (val stats)',
                            "{meters}"]).format(
                                meters=str(metric_logger[i].return_mean())
                            ))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_model_parallel_size', type = int,     default = 1,                                                        help = 'tensor_model_parallel_size')
    parser.add_argument('--data_dir',                   type = str,     default = 'E:\\代码上传\\SETCD_download\\GRIDSAT\\npy_fengwu_era5',  help = 'dataset dir')
    parser.add_argument('--checkpoint_path',            type = str,     default = 'E:\\MSCAR\check_point\\checkpoint_best.pth',             help = 'checkpoint path')
    parser.add_argument('--seed',                       type = int,     default = 0,                                                        help = 'seed')
     
    args = parser.parse_args()
   
    main(args)
