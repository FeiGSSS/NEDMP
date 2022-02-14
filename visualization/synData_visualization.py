# -*- encoding: utf-8 -*-
'''
@File    :   synData_visualization.py
@Time    :   2021/07/20 14:44:36
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle as pkl

def plot_scatter_dmp(x, y, save_path):
    l1 = np.mean(np.abs(x-y))
    plt.figure(figsize=(4,4))
    plt.scatter(x, y, c="r",s=10, marker=".", alpha=0.8, edgecolors="k")
    plt.plot([0,1], "k-.")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300,bbox_inches='tight')

def plot_scatter(x, y, save_path):
    l1 = np.mean(np.abs(x-y))
    plt.figure(figsize=(4,4))
    plt.scatter(x, y, c="r",s=10, marker=".", alpha=0.8, edgecolors="k")
    plt.plot([0,1], "k-.")
    plt.axis("off")
    # plt.title("{:.4f}".format(l1),fontsize = 28)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300,bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--model", type=str, default="lgnn")
    parser.add_argument("--diff", type=str, default="SIR")
    args = parser.parse_args()

    load_path = "../data/synthetic/{}/train_data/{}_200.pkl_{}_testResults.pkl".format(args.data_name, args.diff, args.model)
    save_path_dmp = "./figs/synData/{}_{}_{}.png".format(args.data_name, args.diff, "DMP")
    save_path_model= "./figs/synData/{}_{}_{}.png".format(args.data_name, args.diff, args.model)

    with open(load_path, "rb") as f:
        data = pkl.load(f)
    test_predict, test_label, dmp_predict = data.values()
    
    true_value = np.hstack([x[-1, :, -1].squeeze() for x in test_label])
    model_value = np.hstack([x[-1, :, -1].squeeze() for x in test_predict])
    dmp_value = np.hstack([x[-1, :, -1].squeeze() for x in dmp_predict])

    plot_scatter_dmp(dmp_value, true_value, save_path_dmp)
    plot_scatter(model_value, true_value, save_path_model)

