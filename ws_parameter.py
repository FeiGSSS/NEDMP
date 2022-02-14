# -*- encoding: utf-8 -*-
'''
@File    :   train_er_structure.py
@Time    :   2021/07/25 19:52:23
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import time
import argparse
import numpy as np
from tqdm import tqdm
import pickle as pkl
from functools import partial

import torch
from torch import optim
from torch.utils.data import DataLoader

from src.utils.dataset import graph_dataset
from src.utils.utils import aligning, L1_error
from src.model.model import NodeGNN, EdgeGNN, NEDMP

def eval(model, loader,  saving=False):
    # Eval
    model.eval()
    test_predict = []
    test_label = []
    dmp_predict = []
    for i, inputs in enumerate(loader):
        """
        inputs = *, simu_marginal, dmp_marginal, adj
        """
        data4model, label, dmp = inputs[:-2], inputs[-2], inputs[-1]
        dmp = aligning(label, dmp).cpu().numpy()
        # 1. forward
        pred, _ = model(data4model)
        # 2. loss: label and pred both have size [T, N, K]
        pred = aligning(label, pred)
        pred = np.exp(pred.detach().cpu().numpy())
        # 3. record training L1 error
        test_predict.append(pred)
        test_label.append(label.detach().cpu().numpy())
        dmp_predict.append(dmp)
    model_l1 = L1_error(test_predict, test_label)
    dmp_l1 = L1_error(dmp_predict, test_label)

    if saving:
        return test_predict, test_label, dmp_predict, model_l1, dmp_l1
    else:
        return model_l1, dmp_l1

def infer(model, device, args):
    model_path = "./data_process/ws_parameter/{}5/train_data/{}_100.pkl_{}_{}.pt".format(args.parameter, args.diff, args.model, args.diff)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    res = []
    for i in range(10):
        data_path = "data_process/ws_parameter/{}{}/train_data/{}_100.pkl".format(args.parameter, i, args.diff)
        loaded_data = graph_dataset(root=data_path, device=device, nedmp = args.model == "nedmp")
        test_pred_l1, dmp_l1 = eval(model, loaded_data, saving=False)
        res.append((test_pred_l1, dmp_l1))
        print("Final Test: L1 = {:.3f} {:.3f} Base = {:.3f} {:.3f} ".format(test_pred_l1[0], test_pred_l1[1], dmp_l1[0], dmp_l1[1]))
    save_path = "./data_process/ws_parameter/{}_{}_{}_results.pkl".format(args.parameter, args.diff, args.model)
    with open(save_path, "wb") as f:
        pkl.dump(res, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gnn")
    parser.add_argument("--num_status", type=int, default=3)
    parser.add_argument("--diff", type=str, default="SIR")
    parser.add_argument("--parameter", type=str, default="beta")
    args = parser.parse_args()

    device = torch.device("cpu")

    if args.model == "gnn":
        model = NodeGNN(node_feat_dim=32,
                        edge_feat_dim=32,
                        message_dim=32,
                        number_layers=30,
                        num_status=3,
                        device=device)
    elif args.model == "nedmp":
        model = NEDMP(in_dim=1, 
                     hid_dim=32, 
                     number_layers=30,
                     device=device)

    infer(model, device, args)