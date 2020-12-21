import torch
import numpy as np
from collections import OrderedDict
import os
import sys

src_para_num, src_path, tar_path = sys.argv[1:]

src_para_num = int(src_para_num)
tar_para_num = 1
model_name = 'mp_rank_0%s_model_states.pt'
src_model_folder = src_path
tar_model_folder = tar_path
if not os.path.exists(tar_model_folder):
    os.makedirs(tar_model_folder)

src_model = dict(zip(range(src_para_num), range(src_para_num)))
src_module = dict(zip(range(src_para_num), range(src_para_num)))
tar_model = {}
tar_module = OrderedDict()
for i in range(src_para_num):
    src_model[i] = torch.load(os.path.join(src_model_folder, model_name%i), map_location='cpu')
    src_module[i] = src_model[i]['module']

para_count = 0
for key in src_module[0].keys():
    print(key, src_module[0][key].numpy().shape, src_module[0][key].dtype)
    para_count += np.prod(src_module[0][key].numpy().shape)
print("Number of parameters in SOURCE model PART 0: %d"%(para_count))

for key in src_module[0].keys():
    # merge in dim 0
    if 'word_embeddings.weight' == key or 'mlp.dense_h_to_4h' in key:
        tmp = []
        for split_part_idx in range(src_para_num):
            tmp.append(src_module[split_part_idx][key].clone())
        tar_module[key] = torch.cat(tmp, dim=0)

    # identity
    elif 'position_embeddings.weight' == key or 'layernorm' in key or 'dense.bias' in key or 'dense_4h_to_h.bias' in key:
        tar_module[key] = src_module[0][key].clone()

    # merge in dim 1
    elif 'dense.weight' in key or 'dense_4h_to_h.weight' in key:
        tmp = []
        for split_part_idx in range(src_para_num):
            tmp.append(src_module[split_part_idx][key].clone())
        tar_module[key] = torch.cat(tmp, dim=1)
        
    elif 'query_key_value' in key: #(q, k, v) ->(q0, k0, v0, q1, k1, v1)
        src_shape = src_module[0][key].numpy().shape
        q_dim = src_shape[0] // 3 # 1280
        tmp_q, tmp_k, tmp_v = [], [], []
        for split_part_idx in range(src_para_num):
            tmp_q.append(src_module[split_part_idx][key][q_dim*0:q_dim*1].clone())
            tmp_k.append(src_module[split_part_idx][key][q_dim*1:q_dim*2].clone())
            tmp_v.append(src_module[split_part_idx][key][q_dim*2:q_dim*3].clone())
        tar_module[key] = torch.cat(tmp_q+tmp_k+tmp_v, dim=0)
    else:
        RuntimeError("invalid key %s"%key)

para_count = 0
for key in tar_module.keys():
    print(key, tar_module[key].numpy().shape)
    para_count += np.prod(tar_module[key].numpy().shape)
print("Number of parameters in MERGED model: %d"%(para_count))

tar_model['module'] = tar_module
torch.save(tar_model, os.path.join(tar_model_folder, model_name%0))
