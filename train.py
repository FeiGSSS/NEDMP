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
from src.model.model import NodeGNN, NEDMP

torch.manual_seed(42)

def train(model, optimizer, loader):
    # Train
    model.train()
    train_loss = 0
    train_predict = []
    train_label = []
    for i, inputs in tqdm(enumerate(loader)):
        optimizer.zero_grad()
        """
        inputs = *, simu_marginal, dmp_marginal, adj
        """
        data4model, label = inputs[:-2], inputs[-2]
        pred, _ = model(data4model)
        pred = aligning(label, pred)
        loss = model.loss_function(pred, label)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        pred = np.exp(pred.detach().cpu().numpy())
        train_predict.append(pred)
        train_label.append(label.detach().cpu().numpy())

    train_loss /= len(loader)
    train_pred_l1, _ = L1_error(train_predict, train_label)
    return train_loss, train_pred_l1

def eval(model, loader, testing=False, saving=False):
    # Eval
    model.eval()
    device = model.device
    if not testing:
        val_predict = []
        val_label = []
        for i, inputs in enumerate(loader):
            """
            inputs = *, simu_marginal, dmp_marginal, adj
            """
            data4model, label = inputs[:-2], inputs[-2]
            # 1. forward
            pred, _ = model(data4model)
            # 2. loss: label and pred both have size [T, N, K]
            #print("Train Shape: ", label.shape[0], pred.shape[0])
            pred = aligning(label, pred)
            # 3. record training L1 error
            pred = np.exp(pred.detach().cpu().numpy())
            val_predict.append(pred)
            val_label.append(label.detach().cpu().numpy())
        val_pred_l1, _ = L1_error(val_predict, val_label)
        return val_pred_l1
    else:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_feat_dim", type=int, default=32)
    parser.add_argument("--edge_feat_dim", type=int, default=32)
    parser.add_argument("--message_dim", type=int, default=32)
    parser.add_argument("--cuda_id", type=int, default=0, help="-1 for cpu")
    parser.add_argument("--number_layers", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--factor", type=float, default=0.5, help="LR reduce factor")
    parser.add_argument("--patience", type=int, default=5, help="patience of LR reduce schema")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--model", type=str, default="gnn")
    parser.add_argument("--num_status", type=int, default=3)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--diff", type=str, default="SIR")
    parser.add_argument("--tmp", type=str, default="")
    args = parser.parse_args()

    # print args
    print("=== Setting Args ===")
    for arg, value in vars(args).items():
        print("{:>20} : {:<20}".format(arg, value))

    if args.cuda_id >=0 :
        device = torch.device("cuda:{}".format(args.cuda_id))
    else:
        device = torch.device("cpu")

    if args.model == "gnn":
        model = NodeGNN(node_feat_dim=args.node_feat_dim,
                        edge_feat_dim=args.edge_feat_dim,
                        message_dim=args.message_dim,
                        number_layers=args.number_layers,
                        num_status=args.num_status,
                        device=device)
    elif args.model == "nedmp":
        model = NEDMP(hid_dim=args.message_dim, 
                     number_layers=args.number_layers,
                     device=device)

    # optimizer
    optimizer = optim.Adamax(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=args.factor, patience=args.patience, verbose=True)

    # dataset
    loaded_data = graph_dataset(root=args.data_path, device=device, nedmp = args.model == "nedmp")

    train_size = int(args.train_ratio*len(loaded_data))
    val_size = int(args.val_ratio*len(loaded_data))
    test_size = len(loaded_data) - train_size - val_size

    train_data, val_data, test_data = torch.utils.data.random_split(loaded_data, 
                                                                    [train_size, val_size, test_size],
                                                                    generator=torch.Generator().manual_seed(42))
    print("DataSize: Train={} Val={} Test={}".format(len(train_data), len(val_data), len(test_data)))
    
    model_save_path = args.data_path+"_{}_{}{}.pt".format(args.model, args.diff, args.tmp)
    test_save_path = args.data_path+"_{}_{}{}_testResults.pkl".format(args.model, args.diff, args.tmp)

    # Training 
    if not args.testing:
        train_time = time.time()
        best_eval_l1 = 1E+10
        early_stop = 0
        # Traing and Eval
        for epoch in range(100):
            # Train
            train_loss, train_pred_l1 = train(model, optimizer, train_data)
            # Eval
            eval_pred_l1 = eval(model, val_data)
            print(" Epoch={:<3}, Loss={:<.3f} || Train L1 : model={:.3f} || Eval L1 : model={:.3f} || Time = {:.1f}s".format(epoch, train_loss, train_pred_l1, eval_pred_l1, time.time()-train_time))
            if eval_pred_l1<best_eval_l1:
                early_stop = 0
                best_eval_l1 = eval_pred_l1
                torch.save(model.state_dict(), model_save_path)
                # Test
                test_pred_l1, dmp_l1 = eval(model, test_data, testing=True)
                print("-"*72)
                print("Test: L1 = {:.3f} Base = {:.3f}".format(test_pred_l1[0], dmp_l1[0]))
                print("-"*72)
            else:
                early_stop += 1
            
            if early_stop > args.early_stop:
                break
            scheduler.step(eval_pred_l1)

        # Test
        model.load_state_dict(torch.load(model_save_path))
        test_pred_l1, dmp_l1 = eval(model, test_data, testing=True)
        print("Final Test: L1 = {:.3f} Base = {:.3f}".format(test_pred_l1[0], dmp_l1[0]))
        
    else:
        # Test
        device = "cpu" if args.cuda_id==-1 else "cuda:{}".format(args.cuda_id)
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        test_predict, test_label, dmp_predict, test_pred_l1, dmp_l1 = eval(model, test_data, testing=True, saving=True)
        print("Final Test: L1 = {:.3f} $\\pm$ {:.3f} Base = {:.3f} \\pm {:.3f} ".format(test_pred_l1[0], test_pred_l1[1], dmp_l1[0], dmp_l1[1]))

        with open(test_save_path, "wb") as f:
            pkl.dump({"test_predict": test_predict, "test_label": test_label, "dmp_predict": dmp_predict}, f)
