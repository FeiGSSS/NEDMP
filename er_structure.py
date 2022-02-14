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

torch.manual_seed(42)

from train import eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_feat_dim", type=int, default=32)
    parser.add_argument("--edge_feat_dim", type=int, default=32)
    parser.add_argument("--message_dim", type=int, default=32)
    parser.add_argument("--cuda_id", type=int, default=0, help="-1 for cpu")
    parser.add_argument("--number_layers", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5, help="patience of LR reduce schema")
    parser.add_argument("--train_ratio", type=float, default=0.5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--model", type=str, default="lgnn")
    parser.add_argument("--num_status", type=int, default=3)
    args = parser.parse_args()

    # print args
    print("=== Setting Args ===")
    for arg, value in vars(args).items():
        print("{:>20} : {:<20}".format(arg, value))

    if args.cuda_id >=0 :
        device = torch.device("cuda:{}".format(args.cuda_id))
    else:
        device = torch.device("cpu")

    if args.model == "lgnn":
        from src.model.lgnn import MP_model
    elif args.model == "gnn":
        from src.model.gnn import MP_model
    elif args.model == "nedmp":
        from src.model.nedmp import NEDMP as MP_model

    if args.model == "nedmp":
        model = MP_model(in_dim=1, 
                        hid_dim=args.message_dim, 
                        number_layers=args.number_layers,
                        device=device)
    else:
        model = MP_model(node_feat_dim=args.node_feat_dim,
                            edge_feat_dim=args.edge_feat_dim,
                            message_dim=args.message_dim,
                            number_layers=args.number_layers,
                            num_status=args.num_status,
                        device=device)
    nedmp_flag = args.model == "nedmp"

    # model.load_state_dict(torch.load("./data/er_structure/p5/train_data/SIR_100.pkl"+"_{}.pt".format(args.model), map_location='cpu'))
    # for i in range(10):
    #     data_path = "data/er_structure/p{}/train_data/SIR_100.pkl".format(i)
    #     # dataset
    #     loaded_data = graph_dataset(root=data_path, nedmp=nedmp_flag)
    #     train_size = int(args.train_ratio*len(loaded_data))
    #     val_size = int(args.val_ratio*len(loaded_data))
    #     test_size = len(loaded_data) - train_size - val_size

    #     train_data, val_data, test_data = torch.utils.data.random_split(loaded_data, \
    #         [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    #     collat_fn = partial(loaded_data.collat_fn, nedmp=nedmp_flag)
    #     test_loader = DataLoader(test_data, batch_size=1, num_workers=20,\
    #         collate_fn=collat_fn)

    #     test_predict, test_label, dmp_predict, test_pred_l1, dmp_l1 = eval(model, test_loader, testing=True, saving=True)
    #     print("Final Test: L1 = {:.6f} Base = {:.6f}".format(test_pred_l1, dmp_l1))

    #     with open(data_path+"_{}_testResults_basedOnP5.pkl".format(args.model), "wb") as f:
    #         pkl.dump({"test_predict": test_predict, "test_label": test_label, "dmp_predict": dmp_predict}, f)

    model.load_state_dict(torch.load("./data/er_structure/n5/train_data/SIR_100.pkl"+"_{}.pt".format(args.model), map_location='cpu'))
    for i in range(10):
        data_path = "data/er_structure/n{}/train_data/SIR_100.pkl".format(i)
        # dataset
        loaded_data = graph_dataset(root=data_path, nedmp=nedmp_flag)
        train_size = int(args.train_ratio*len(loaded_data))
        val_size = int(args.val_ratio*len(loaded_data))
        test_size = len(loaded_data) - train_size - val_size

        train_data, val_data, test_data = torch.utils.data.random_split(loaded_data, \
            [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
        collat_fn = partial(loaded_data.collat_fn, nedmp=nedmp_flag)
        test_loader = DataLoader(test_data, batch_size=1, num_workers=20,\
            collate_fn=collat_fn)

        test_predict, test_label, dmp_predict, test_pred_l1, dmp_l1 = eval(model, test_loader, testing=True, saving=True)
        print("Final Test: L1 = {:.6f} Base = {:.6f}".format(test_pred_l1, dmp_l1))

        with open(data_path+"_{}_testResults_basedOnN5.pkl".format(args.model), "wb") as f:
            pkl.dump({"test_predict": test_predict, "test_label": test_label, "dmp_predict": dmp_predict}, f)