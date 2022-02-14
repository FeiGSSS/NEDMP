import argparse
import numpy as np
import pickle as pkl
from collections import defaultdict

import torch
from torch.utils import data
from src.utils.dataset import graph_dataset
from src.utils.utils import aligning, L1_error
from src.model.model import NodeGNN, NEDMP


def eval(model, loader):
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

    return model_l1, dmp_l1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gnn")
    parser.add_argument("--num_status", type=int, default=3)
    parser.add_argument("--diff", type=str, default="SIR")
    parser.add_argument("--data", type=str, default="syn")
    parser.add_argument("--cuda", type=int, default=-1)
    args = parser.parse_args()

    if args.cuda == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(args.cuda))

    if args.model == "gnn":
        model = NodeGNN(node_feat_dim = 32,
                        edge_feat_dim = 32,
                        message_dim   = 32,
                        number_layers = 30,
                        num_status    = 3,
                        device        = device)
    elif args.model == "nedmp":
        model = NEDMP(hid_dim       = 32, 
                     number_layers = 30,
                     device        = device)
    if args.data == "syn":
        data_names = ["tree", "grid", "barbell", "regular_graph", "er03", "er05", "er08", "complete"]
        model_paths = ["./data/synthetic/{}/train_data/{}_200.pkl_{}_{}.pt".format(name, args.diff, args.model, args.diff) for name in data_names]
        data_paths  = ["./data/synthetic/{}/train_data/{}_200.pkl".format(name, args.diff) for name in data_names]
    elif args.data == "real":
        data_names = ["dolphins", "fb-food", "fb-social", "norwegain", "openflights", "top-500"]
        model_paths = ["./data/realnets/{}/train_data/{}_150.pkl_{}_{}.pt".format(name, args.diff, args.model, args.diff) for name in data_names]
        data_paths  = ["./data/realnets/{}/train_data/{}_150.pkl".format(name, args.diff) for name in data_names]

    preds = defaultdict(list)
    dmps  = defaultdict(list)
    for mp, model_name in zip(model_paths, data_names):
        model.load_state_dict(torch.load(mp))
        tmp_pred = []
        tmp_dmps = []
        for dp in data_paths:
            # dataset
            loaded_data = graph_dataset(root=dp, device=device, nedmp = args.model == "nedmp")
            test_pred_l1, dmp_l1 = eval(model, loaded_data)
            tmp_pred.append(test_pred_l1)
            tmp_dmps.append(dmp_l1)
        preds[model_name] = tmp_pred
        dmps[model_name]  = tmp_dmps
        print(tmp_pred)
        print(tmp_dmps)
        print("*"*72)

    if args.data == "syn":
        with open("./data/synthetic/structure_generalization_{}.pkl".format(args.model), "wb") as f:
            pkl.dump([preds, dmps], f)
    elif args.data == "real":
        with open("./data/realnets/structure_generalization_{}.pkl".format(args.model), "wb") as f:
            pkl.dump([preds, dmps], f)
