import os
import sys
from open_clip.clip import load, tokenize, _transform
import copy
import torch 
from torch import nn, optim
'''
Part of the code was taken from
https://github.com/miccunifi/SEARLE/tree/main
'''

def ti_opt(text, image_feature, model, optim_vars):

	# len_text = len(text.split(" ")) # number of words in text
	text_tokens = tokenize(f"{text}") # 1 x 77
	len_text = len(torch.where(text_tokens[0] != 0)[0][1:-1])
	
	device = "cuda"
	x_input = model.token_embedding(text_tokens.to(device)) # 77 * 512
	if optim_vars["init"] == "knn":
		token_emb = nn.Parameter(x_input[0,1:len_text+1].clone())
	else: #rnd
		token_emb = torch.empty((len_text,x_input.shape[-1]), requires_grad=True, device=device)
		nn.init.normal_(token_emb, std=0.02)
		token_emb = nn.Parameter(token_emb)
	# EMA token
	ema_token_emb = token_emb.clone().detach()

	criterion = nn.CosineEmbeddingLoss()
	criterion_target = torch.as_tensor([1], device=device)

	lr = optim_vars["lr"]
	iters = optim_vars["iters"]
	alpha = optim_vars["alpha"]
	weight_decay = optim_vars["weight_decay"]
	optimizer = optim.SGD([token_emb], lr=lr, weight_decay=weight_decay)
	scaler = torch.cuda.amp.GradScaler()

	# init
	with torch.no_grad():
		x_input = model.token_embedding(text_tokens.to(device)) # 77 * 512		
		x_input[0,1:len_text+1] = token_emb

		x = x_input + model.positional_embedding.type(model.dtype)
		x = x.permute(1, 0, 2)  # NLD -> LND
		x = model.transformer(x.type(model.dtype))
		x = x.permute(1, 0, 2)  # LND -> NLD
		x = model.ln_final(x).type(model.dtype).detach()
		# x.shape = [batch_size, n_ctx, transformer.width]
		# take features from the eot embedding (eot_token is the highest number in each sequence)
		x_init = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ model.text_projection

	for _ in range(iters):
		# print(ema_token_emb[0,:10])
		with torch.cuda.amp.autocast():

			x_input = model.token_embedding(text_tokens.to(device)) # 77 * 512		
			x_input[0,1:len_text+1] = token_emb

			x = x_input + model.positional_embedding.type(model.dtype)
			x = x.permute(1, 0, 2)  # NLD -> LND
			x = model.transformer(x)
			x = x.permute(1, 0, 2)  # LND -> NLD
			x = model.ln_final(x).type(model.dtype)

			# x.shape = [batch_size, n_ctx, transformer.width]
			# take features from the eot embedding (eot_token is the highest number in each sequence)
			x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ model.text_projection
			cosine_loss = criterion(x, image_feature.unsqueeze(0), criterion_target)
			cosine_loss += criterion(x, x_init, criterion_target)

			scaler.scale(cosine_loss).backward()
			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad()

			ema_token_emb = alpha * ema_token_emb + (1 - alpha) * token_emb

	return ema_token_emb.detach()

def compose_token(text, token_opt_emb, model):

	device = "cuda"
	with torch.no_grad():
		text_tokens = tokenize(text) # 1 x 77
		token_emb = model.token_embedding(text_tokens.to(device))
		# token_emb_copy = copy.deepcopy(token_emb)
		
		# replace token embs of
		# knn text with optimized emb
		token_emb[0,1:len(token_opt_emb)+1] = token_opt_emb

		# sanity check with lr = 0
		# assert torch.sum(token_emb - token_emb_copy).abs() <  1e-7
		
		# valid_tokens_idx = torch.where(text_tokens[0] != 0)[0]
		# end_token = token_emb[0,valid_tokens_idx[-1]]
		# text_token = token_emb[0,valid_tokens_idx[1:-1]]
		# len_text = len(text_token) # len of text exc ""
		# end_token_loc = len(token_opt_emb)+1+len_text

		# # form new token_emb
		# token_emb[0,1:len(token_opt_emb)+1] = token_opt_emb # from knn queries and optimized
		# token_emb[0,len(token_opt_emb)+1:len(token_opt_emb)+1+len_text] = text_token # domain token for eg
		# token_emb[0,len(token_opt_emb)+1+len_text] = end_token
		
		x = token_emb + model.positional_embedding.type(model.dtype)
		x = x.permute(1, 0, 2)  # NLD -> LND
		x = model.transformer(x.type(model.dtype))
		x = x.permute(1, 0, 2)  # LND -> NLD
		x = model.ln_final(x).type(model.dtype)

		# x.shape = [batch_size, n_ctx, transformer.width]
		# take features from the eot embedding (eot_token is the highest number in each sequence)
		x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ model.text_projection

	return x