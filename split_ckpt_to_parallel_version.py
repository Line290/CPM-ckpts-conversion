import torch
import numpy as np
from collections import OrderedDict
import os
import sys

tar_para_num, src_path, tar_path = sys.argv[1:]
src_para_num = 1
tar_para_num = int(tar_para_num)
model_name = 'mp_rank_0%s_model_states.pt'
src_model_folder = src_path
tar_model_folder = tar_path
if not os.path.exists(tar_model_folder):
    os.makedirs(tar_model_folder)

src_model = dict(zip(range(src_para_num), range(src_para_num)))
src_module = dict(zip(range(src_para_num), range(src_para_num)))
tar_model = dict(zip(range(tar_para_num), [{} for i in range(tar_para_num)]))
tar_module = dict(zip(range(tar_para_num), [OrderedDict() for i in range(tar_para_num)]))
for i in range(src_para_num):
    src_model[i] = torch.load(os.path.join(src_model_folder, model_name%i), map_location='cpu')
    src_module[i] = src_model[i]['module']

para_count = 0
for key in src_module[0].keys():
    print(key, src_module[0][key].numpy().shape, src_module[0][key].dtype)
    para_count += np.prod(src_module[0][key].numpy().shape)
print("Number of parameters in SOURCE model: %d"%(para_count))

for key in src_module[0].keys():
    # split in dim 0
    if 'word_embeddings.weight' == key or 'mlp.dense_h_to_4h' in key:
        tmp = src_module[0][key]
        tmp_len = tmp.size()[0]
        assert tmp_len % tar_para_num == 0
        part_len = tmp_len // tar_para_num
        for split_part_idx in range(tar_para_num):
            start_idx = split_part_idx*part_len
            end_idx = (split_part_idx+1)*part_len
            tar_module[split_part_idx][key] = tmp[start_idx:end_idx].clone()

    # identity
    elif 'position_embeddings.weight' == key or 'layernorm' in key or 'dense.bias' in key or 'dense_4h_to_h.bias' in key:
        for split_part_idx in range(tar_para_num):
            tar_module[split_part_idx][key] = src_module[0][key].clone()

    # split in dim 1
    elif 'dense.weight' in key or 'dense_4h_to_h.weight' in key:
        tmp = src_module[0][key]
        tmp_len = tmp.size()[1]
        assert tmp_len % tar_para_num == 0
        part_len = tmp_len // tar_para_num
        for split_part_idx in range(tar_para_num):
            start_idx = split_part_idx*part_len
            end_idx = (split_part_idx+1)*part_len
            tar_module[split_part_idx][key] = tmp[:, start_idx:end_idx].clone()
        
    elif 'query_key_value' in key: #(q, k, v) ->(q0, k0, v0, q1, k1, v1)
        tmp = src_module[0][key]
        src_shape = tmp.numpy().shape
        q_dim = src_shape[0] // 3 # 7680/3
        q_dim_part = q_dim // tar_para_num
        tmp_q, tmp_k, tmp_v = tmp[:q_dim].clone(), tmp[q_dim:q_dim*2].clone(), tmp[q_dim*2:].clone()

        for split_part_idx in range(tar_para_num):
            start_idx = split_part_idx*q_dim_part
            end_idx = (split_part_idx+1)*q_dim_part
            tar_module[split_part_idx][key] = torch.cat([tmp_q[start_idx:end_idx].clone(), 
                                                         tmp_k[start_idx:end_idx].clone(), 
                                                         tmp_v[start_idx:end_idx].clone()], dim=0)
    
    else:
        RuntimeError("invalid key %s"%key)

for split_part_idx in range(tar_para_num):
    para_count = 0
    for key in tar_module[split_part_idx].keys():
        para_count += np.prod(tar_module[split_part_idx][key].numpy().shape)
    print("Number of parameters in PART %d : %d"%(split_part_idx, para_count))
    tar_model[split_part_idx]['module'] = tar_module[split_part_idx]
    torch.save(tar_model[split_part_idx], os.path.join(tar_model_folder, model_name%split_part_idx))

