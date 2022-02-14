# -*- encoding: utf-8 -*-
'''
@File    :   merge_sample.py
@Time    :   2021/09/22 21:10:50
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import pickle as pkl

# # beta

# with open("beta2/train_data/SIR_100.pkl", "rb")  as f:
#     data2 = pkl.load(f)

# with open("beta5/train_data/SIR_100.pkl", "rb")  as f:
#     data5 = pkl.load(f)

# with open("beta8/train_data/SIR_100.pkl", "rb")  as f:
#     data8 = pkl.load(f)

# merge = {}
# for k in list(data2.keys())[1:]:
#     merge[k] = data2[k]+data5[k]+data8[k]
# with open("./beta_sample/SIR.pkl", "wb") as f:
#     pkl.dump(merge, f)

# gamma
with open("gamma2/train_data/SIR_100.pkl", "rb")  as f:
    data2 = pkl.load(f)

with open("gamma5/train_data/SIR_100.pkl", "rb")  as f:
    data5 = pkl.load(f)

with open("gamma8/train_data/SIR_100.pkl", "rb")  as f:
    data8 = pkl.load(f)

merge = {}
for k in list(data2.keys())[1:]:
    merge[k] = data2[k]+data5[k]+data8[k]
with open("./gamma_sample/SIR.pkl", "wb") as f:
    pkl.dump(merge, f)