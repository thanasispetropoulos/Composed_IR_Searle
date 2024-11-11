import os
import sys
import numpy as np
import pickle
import torch
from pdb import set_trace as st

device = "cuda"
batch_size = 10000
num_batches = 4
topk = 50
q_feats = torch.rand(1000,768).to(device)
# laion_feats_all = torch.Tensor(np.load("laion_file.npz")) # 40 M x 768
laion_feats_all = torch.Tensor(np.load("/mnt/personal/efthynik/2023_Composed_Image_retrieval/clip_features/laion_40m/ViT-L-14_openai_train_image_feats.npy", mmap_mode='r'))

def load_laion_feats(i, batch_size):
	laion_feats = []
	for f in range(i * batch_size, (i+1) * batch_size):
		# fp = open(f"{f}.pkl", "rb")
		# laion_feats.append(torch.Tensor(pickle.load(fp['vecs'])).squeeze())
		laion_feats.append(torch.rand(768))

	return torch.stack(laion_feats)

S = []
indices = []
for i in range(num_batches):
	# load laion features
	# from i x batch_size: (i+1) x batch_size
	laion_feats_b = laion_feats_all[i*batch_size: (i+1)*batch_size, :].to(device)
	# laion_feats_b = load_laion_feats(i, batch_size).to(device)
	idx_map = torch.arange(i*batch_size, (i+1)*batch_size)
	# indices.extend(idx_map)
	# batch_wise similarity
	import time  
	start = time.time()
	S_b = q_feats @ laion_feats_b.T
	S_b, idx = torch.sort(S_b, descending=True, dim=-1)
	print(f"time taken: {time.time()-start}")
	S_b = S_b[:,:topk].cpu() # q x (topk)
	idx = idx[:,:topk].cpu() # 0 - batch_size

	# idx = idx_map[idx.reshape(-1)].reshape(S_idx.shape[0],-1) # map to i based index
	idx += (i * batch_size) # ->  0-40M

	if i == 0:
		S_best = S_b
		idx_best = idx

	else:
		S_comb = torch.cat((S_b, S_best), dim=-1) # q x  (topk x 2)
		idx_comb = torch.cat((idx, idx_best), dim=-1) # q x  (topk x 2) # 0-40M
		
		S_best, indices_local = torch.sort(S_comb, descending=True, dim=-1)
		idx_best = torch.take_along_dim(idx_comb, indices_local, dim=-1)[:,:topk] # 0- (2 x topk) -> (0-40M)
		# S_best = torch.take_along_dim(S_comb, )
		S_best = S_best[:,:topk]