import pickle
import numpy as np
import torch
import torch.nn.functional as F
from open_clip.clip import load, tokenize, _transform
import pdb
from pdb import set_trace as st
import time
import faiss 
import re
import math
from PIL import Image
import sys
from utils import *
import pandas as pd
import os,sys
import argparse

parser = argparse.ArgumentParser(description='Training the Model')
parser.add_argument('--dataset', type=str, help='define dataset')
parser.add_argument('--source', type=str, help='define source domain')
parser.add_argument('--target', type=str, help='define target domain')
parser.add_argument('--lr', type=float, default=0.01, help="lr")
parser.add_argument('--iters', type=int, default=50, help="opt iters")
parser.add_argument('--disentangle_loss', type=float)
parser.add_argument('--close_to_image_loss', type=float)
parser.add_argument('--topk', type=int, default=1, help="miu for searle")
parser.add_argument('--prompt_format', type=str, default="ours", help='prompt format', choices=["pic2word", "ours", "compodiff"])
args = parser.parse_args()

def read_dataset_features(pickle_dir):
    with open(pickle_dir, 'rb') as f:
        data = pickle.load(f)
    all_image_features = torch.from_numpy(data['feats'].astype("float32")).float().to('cuda')
    all_paths = data['path']
    all_classes = data['classes']
    all_domains = data['domains']
    return all_image_features, all_paths, all_classes, all_domains

def read_patch_dataset_features(pickle_dir):
    with open(pickle_dir, 'rb') as f:
        data = pickle.load(f)
    all_cls_features = torch.from_numpy(data['cls'].astype("float32")).float()
    all_patch_features = torch.from_numpy(data['patches'].astype("float32")).float()
    all_atention_6 = torch.from_numpy(data['attn@6'].astype("float32")).float()
    all_atention_24 = torch.from_numpy(data['attn@24'].astype("float32")).float()
    all_paths = data['path']
    all_classes = data['classes']
    all_domains = data['domains']
    return all_cls_features, all_patch_features, all_atention_6, all_atention_24, all_paths, all_classes, all_domains

def read_corpus_features(pickle_dir):
    with open(pickle_dir, "rb") as data:
        data_dict = pickle.load(data)
        #descr = torch.from_numpy(data_dict["feats"].astype("float32")).float().to('cuda')
        descr = np.array(data_dict["feats"]).astype("float32")
        names = data_dict["prompts"]
    return descr, names

def read_laion_corpus(dataset):
    with open(root+'/clip_features/laion_40m/'+dataset+'_laion_corpus_feats.pkl', 'rb') as f:
        laion_corpus_feats = pickle.load(f)
    laion_corpus_feats = torch.from_numpy(laion_corpus_feats['feats'].astype("float32")).float()
    laion_corpus_feats = laion_corpus_feats / laion_corpus_feats.norm(dim=-1, keepdim=True)
    laion_corpus_feats = np.array(laion_corpus_feats).astype("float32")
    with open(root+'/clip_features/laion_40m/'+dataset+'_laion_real_text.pkl', 'rb') as f:
        laion_real_text = pickle.load(f)
    laion_real_text = laion_real_text['actual_text']
    return laion_corpus_feats, laion_real_text

def read_cirr_features(pickle_dir, read_mode):
    with open(pickle_dir, 'rb') as f:
        data = pickle.load(f)
    if 'database' in read_mode.lower():
        all_image_features = torch.from_numpy(data['image_feats'].astype("float32")).float().to('cuda')
        paths = np.array(data['path'])
        return all_image_features, paths
    elif 'val' in read_mode.lower():
        query_features = torch.from_numpy(data['image_feats'].astype("float32")).float().to('cuda')
        query_filenames = np.array(data['path'])
        query_caption_features = torch.from_numpy(data['caption_feats'].astype("float32")).float().to('cuda')
        actual_captions = data['actual_captions']
        query_target_filenames = np.array(data['target_path'])
        return query_features, query_filenames, query_caption_features, actual_captions, query_target_filenames
    elif 'test' in read_mode.lower():
        query_features = torch.from_numpy(data['image_feats'].astype("float32")).float().to('cuda')
        query_filenames = np.array(data['path'])
        query_caption_features = torch.from_numpy(data['caption_feats'].astype("float32")).float().to('cuda')
        actual_captions = data['actual_captions']
        pair_ids = data['pair_ids']
        return query_features, query_filenames, query_caption_features, actual_captions, pair_ids

def read_fashion(pickle_dir, read_mode):
    with open(pickle_dir, 'rb') as f:
        data = pickle.load(f)
    if 'database' in read_mode.lower():
        database_feats = torch.from_numpy(data['database_feats'].astype("float32")).float().to('cuda')
        database_paths = np.array(data['database_paths'])
        return database_feats, database_paths
    elif 'query' in read_mode.lower():
        query_image_feats = torch.from_numpy(data['query_feats'].astype("float32")).float().to('cuda')
        query_image_filenames = np.array(data['query_paths'])
        query_answer_filenames = np.array(data['database_paths'])
        actual_caption_1 = data['actual_caption_1']
        actual_caption_2 = data['actual_caption_2']
        actual_caption_comb = data['actual_caption_comb']
        caption_1_features = torch.from_numpy(data['caption_feat_1'].astype("float32")).float().to('cuda')
        caption_2_features = torch.from_numpy(data['caption_feat_2'].astype("float32")).float().to('cuda')
        caption_comb_features = torch.from_numpy(data['caption_feat_comb'].astype("float32")).float().to('cuda')
        return query_image_feats, query_image_filenames, query_answer_filenames, actual_caption_1, actual_caption_2, actual_caption_comb, caption_1_features, caption_2_features, caption_comb_features


def calculate_rankings(method, image_features, text_features, real_text, database_features, db_patch_features=None, q_patch_features=None,
                       laion_feats=None, sim_with_laion_img=None, ranks_laion_img=None, database_paths=None, actual_label_names=None, use_lambda=False, dataset=None):
    if "1k" in method:
        real_corpus_text = in_corpus_names
        text_corpus_features = in_corpus_feats
    elif "20k" in method:
        real_corpus_text = open_images_corpus_names
        text_corpus_features = open_images_corpus_feats
    elif "236k" in method:
        real_corpus_text = nltk_corpus_names
        text_corpus_features = nltk_corpus_feats
    elif "40m" in method:
        real_corpus_text = laion_corpus_names
        text_corpus_features = laion_corpus_feats

    invfreq_dict = {}
    if 'invfreq' in method.lower():
        den = 1.0
        den = extract_float(method, "den=")
        invfreq_dict = find_inv_freq(database_features, text_corpus_features, real_corpus_text, den)
        
    kappa = 20
    lam = 0.05

    miu = 3
    inversion = False

    if "inversion" in method.lower():
        inversion = True
    if "miu=" in method:
        miu = extract_int(method, "miu=")

    if np.array([x in method for x  in ['Image','Add Similarities','Multiply Similarities', 'Minimum Similarity']]).any():
        sim_img = (image_features @ database_features.t())
    if np.array([x in method for x  in ['Text','Add Similarities','Multiply Similarities', 'Minimum Similarity']]).any():    
        sim_text = (text_features @ database_features.t())

    if "cast" in method.lower():
        # pdb.set_trace()
        ranks_text = torch.argsort(sim_text, descending=True)
        text2img_features = database_features[ranks_text[:, :kappa]]
        text2img_features_std = text2img_features.std(1)
        text2img_features = text2img_features.mean(1)         
        if 'castw' in method.lower():
            mn = text2img_features_std.min(1)[0]
            mx = text2img_features_std.max(1)[0]
            w = (text2img_features_std - mn.unsqueeze(1)) / (mx.unsqueeze(1) - mn.unsqueeze(1))
            text2img_features *= (1-w**.2)
        text2img_features /= (torch.norm(text2img_features, p=2, dim=1, keepdim=True) + 1e-6).expand_as(text2img_features)            
        sim_text =  text2img_features @ database_features.t()

    if "norm" in method.lower():

        sim_img = norm_cdf(sim_img)
        sim_text = norm_cdf(sim_text)

    if "image only" in method.lower():
        sim_total = sim_img
        ranks = torch.argsort(sim_total, descending=True)
    elif "text only" in method.lower():
        sim_total = sim_text
        ranks = torch.argsort(sim_total, descending=True)
    elif "Add Similarities" in method:
        if  'rankagr' in method.lower():
            rank_img = calculate_ranks(sim_img)
            rank_text = calculate_ranks(sim_text)
            ranks = torch.argsort(rank_img + rank_text, descending=False)
        else:
            ranks = torch.argsort(sim_img + sim_text, descending=True)
    elif "Multiply Similarities" in method:
        if  'rankagr' in method.lower():
            rank_img = calculate_ranks(sim_img)
            rank_text = calculate_ranks(sim_text)
            ranks = torch.argsort((1/rank_img) * (1/rank_text), descending=True)
        else:
            ranks = torch.argsort(torch.mul(sim_img, sim_text), descending=True)
    elif "Minimum Similarity" in method:
        if  'rankagr' in method.lower():
            rank_img = calculate_ranks(sim_img)
            rank_text = calculate_ranks(sim_text)
            ranks = torch.argsort(torch.maximum(rank_img, rank_text), descending=False)
        else:
            ranks = torch.argsort(torch.minimum(sim_img, sim_text), descending=True)
    elif "Super-Text w Corpus mNN" in method:

        kappa = 20
        ni=7
        miu=3
        temp_image=1.0
        temp_label=1.0

        if "miu=" in method:
            miu = extract_int(method, "miu=")
        if "ni=" in method:
            ni = extract_int(method, "ni=")
        if "kappa=" in method:
            kappa = extract_int(method, "kappa=")
        if "temp_image=" in method:
            temp_image = extract_float(method, "temp_image=")
        if "temp_label=" in method:
            temp_label = extract_float(method, "temp_label=")

        weighted = False
        if  'weighted' in method.lower():
            weighted = True
        if 'laioni' in method.lower():
            sim_img = sim_with_laion_img
            img_ranks = ranks_laion_img
        else:
            sim_img = (image_features @ database_features.t())
            img_ranks = torch.argsort(sim_img, descending=True)
            laion_feats = None

        if 'multi' in method.lower():
            if 'clean labels' in method.lower():
                multi_texts, weights = clean_labels_from_images(sim_img, text_features, database_features, text_corpus_features, real_corpus_text, real_text, 
                                                       kappa, miu, ni, temp_image, temp_label, multi=True, weighted=weighted, laion_feats=laion_feats, img_ranks=img_ranks)
                for idx, multi_text in enumerate(multi_texts):
                    current_weight = []
                    for idx2 in range(len(multi_text)):
                        if multi_text[idx2].replace(' '+real_text[idx2], '') in invfreq_dict.keys():
                            current_weight.append(invfreq_dict[multi_text[idx2].replace(' '+real_text[idx2], '')])
                        else:
                            current_weight.append(1)
                    current_feature = text_list_to_features(model, multi_text, 256)
                    current_sim = current_feature @ database_features.t()
                    current_sim = torch.Tensor(current_weight).to('cuda').unsqueeze(1)*current_sim
                    if idx == 0:
                        if not weighted:
                            sim_total = current_sim.clone()
                        else:
                            sim_total = current_sim.clone()*torch.tensor(weights[idx]).view(current_sim.shape[0], 1).to('cuda')
                    else:
                        if not weighted:
                            sim_total = sim_total+current_sim
                        else:
                            sim_total = sim_total+(current_sim*torch.tensor(weights[idx]).view(current_sim.shape[0], 1).to('cuda'))
            else:
                multi_text_features, multi_text_queries, weights = img2text_viacorpus(image_features, text_corpus_features, real_corpus_text, real_text, model, miu, 
                                                         multi=True, inversion=inversion, weighted=weighted, temp_label=temp_label)
                for idx, multi_text_feature in enumerate(multi_text_features):
                    current_weight = []
                    for idx2 in range(len(multi_text_queries[idx])):
                        if multi_text_queries[idx][idx2].replace(' '+real_text[idx2], '') in invfreq_dict.keys():
                            current_weight.append(invfreq_dict[multi_text_queries[idx][idx2].replace(' '+real_text[idx2], '')])
                        else:
                            current_weight.append(1)
                    current_sim = multi_text_feature @ database_features.t()
                    current_sim = torch.Tensor(current_weight).to('cuda').unsqueeze(1)*current_sim
                    if idx == 0:
                        if not weighted:
                            sim_total = current_sim.clone()
                        else:
                            sim_total = current_sim.clone()*torch.tensor(weights[idx]).view(current_sim.shape[0], 1).to('cuda')
                    else:
                        if not weighted:
                            sim_total = sim_total+current_sim
                        else:
                            sim_total = sim_total+(current_sim*torch.tensor(weights[idx]).view(current_sim.shape[0], 1).to('cuda'))
        else:
            if 'clean labels' in method.lower():
                sim_img = (image_features @ database_features.t())
                _, text = clean_labels_from_images(sim_img, text_features, database_features, text_corpus_features, real_corpus_text, real_text, 
                                                kappa, miu, ni, temp_image, temp_label, multi=False, weighted=weighted, laion_feats=laion_feats, img_ranks=img_ranks)
                text_features = text_list_to_features(model, text, 256)
                sim_total = (text_features @ database_features.t())
            else:
                text_features = img2text_viacorpus(image_features, text_corpus_features, real_corpus_text, real_text, model, miu, inversion=inversion)
                sim_total = (text_features @ database_features.t())
        ranks = torch.argsort(sim_total, descending=True)
    elif "Super-Text Oracle" in method and actual_label_names is not None:
        if 'multi' in method.lower():
            domain_list_text = [x.split(' ') for x in real_text]
            multi_text_features = []
            for idx1, domains in enumerate(domain_list_text[0]):
                query_list = [actual_label_names[idx] + ' ' + domain_list_text[idx][idx1] for idx in range(len(domain_list_text))]
                multi_text_features.append(text_list_to_features(model, query_list, 256))
            for idx, multi_text_feature in enumerate(multi_text_features):
                current_sim = multi_text_feature @ database_features.t()
                if idx == 0:
                    sim_total = current_sim.clone()
                else:
                    sim_total = sim_total+current_sim
        else:
            oracle_text = [actual_label_names[idx] + ' ' + real_text[idx] for idx in range(len(real_text))]
            text_features = text_list_to_features(model, oracle_text, 256)
            sim_total = (text_features @ database_features.t())
        ranks = torch.argsort(sim_total, descending=True)
    elif "Super-Image" in method:
        kappa = 10
        sim_img = (image_features @ database_features.t())
        multi_texts = clean_labels_from_images(sim_img, text_features, database_features, text_corpus_features, real_corpus_text, real_text, 
                                               kappa, miu=7, ni=7, miu_text=1, multi=True)
        for idx, multi_text in enumerate(multi_texts):
            current_weight = []
            for idx2 in range(len(multi_text)):
                if multi_text[idx2].replace(' '+real_text[idx2], '') in invfreq_dict.keys():
                    current_weight.append(invfreq_dict[multi_text[idx2].replace(' '+real_text[idx2], '')])
                else:
                    current_weight.append(1)
            current_feature = text_list_to_features(model, multi_text, 256)
            current_sim = current_feature @ database_features.t()
            current_sim = torch.Tensor(current_weight).to('cuda').unsqueeze(1)*current_sim
            if idx == 0:
                sim_total = current_sim.clone()
            else:
                sim_total = sim_total+current_sim
        sim_text = (text_features[:1,:] @ database_features.t())
        current_sim = current_sim.cpu()
        sim_total = sim_total.cpu()
        sim_text = sim_text.cpu()
        sim_img = sim_img.cpu()
        text_features = text_features.to('cuda')

        database_features = database_features.cpu()
        image_features = image_features.cpu()

        batch_size = 256
        num_batches = len(image_features) // batch_size
        num_batches_database = len(database_features) // batch_size
        sim_patch_img = torch.zeros(len(image_features), len(database_features))
        sim_domain = torch.zeros(len(database_features))
        start = time.time()
        for idx in range(num_batches + 1):
            print('Database batch', idx, '/', (num_batches + 1), time.time()-start, 'sec', end='\r')
            start = time.time()
            if idx == num_batches:
                current_patches = current_query_patch_features[idx * batch_size:, :, :]
            else:
                current_patches = current_query_patch_features[idx * batch_size:(idx + 1) * batch_size, :, :]
            original_shape_query = current_patches.shape
            current_patches = current_patches.view(-1, current_patches.shape[-1])
            current_patches = current_patches.to('cuda')
            for idx2 in range(num_batches_database + 1):
                if idx2 == num_batches_database:
                    current_database = database_patch_features[idx2 * batch_size:, :, :]
                else:
                    current_database = database_patch_features[idx2 * batch_size:(idx2 + 1) * batch_size, :, :]
                current_database = current_database.view(-1, current_database.shape[-1])

                with torch.no_grad():
                    current_database = current_database.to('cuda')
                    current_sim = (current_patches @ current_database.t()).cpu()

                    height, width = current_sim.shape
                    patch_size = original_shape_query[1]
                    num_patches_height = height // patch_size
                    num_patches_width = width // patch_size
                    reshaped_sim = current_sim.view(num_patches_height, patch_size, num_patches_width, patch_size)
                    max_values, _ = reshaped_sim.max(dim=-1)
                    aggregated_sim = max_values.sum(dim=1)
                    sim_patch_img[idx * batch_size:(idx + 1) * batch_size, idx2 * batch_size:(idx2 + 1) * batch_size] = aggregated_sim

                    if idx == 0:
                        current_text_sim = (text_features[:1,:] @ current_database.t()).squeeze(0).cpu()
                        current_text_sim_reshaped = current_text_sim.view(-1, patch_size)
                        sim_domain[idx2 * batch_size:(idx2 + 1) * batch_size] = current_text_sim_reshaped.sum(dim=1)

        #sim_patch_img = torch.cat(sim_patch_img, dim=0).cpu()
        sim_domain = sim_domain.unsqueeze(0).expand(current_query_patch_features.shape[0], -1).float().cpu()
        sim_domain = norm_cdf(sim_text).cpu() + norm_cdf(sim_domain).cpu()
        #sim_domain = norm_cdf(sim_text).cpu()
        sim_supertext = sim_total.cpu()

        if 'nrm' in method.lower():
            sim_patch_img = norm_cdf(sim_patch_img).cpu()
            sim_domain = norm_cdf(sim_domain).cpu()
            sim_superimage = norm_cdf(sim_patch_img+sim_domain).cpu()
            sim_supertext = norm_cdf(sim_supertext).cpu()
        else:
            sim_superimage = sim_patch_img.cpu()+sim_domain.cpu()

        if 'add' in method.lower():
            sim_total = sim_superimage + sim_supertext
        elif 'multiplication' in method.lower():
            sim_total = torch.mul(sim_superimage, sim_supertext)
        elif 'min' in method.lower():
            sim_total = torch.maximum(sim_superimage, sim_supertext)
        ranks = torch.argsort(sim_total, descending=True)
    elif "pic2word" in method:
        def convert_models_to_fp32(model):
            for p in model.parameters():
                p.data = p.data.float()
                if p.grad:
                    p.grad.data = p.grad.data.float()

        from pic2word.model import IM2TEXT
        from pic2word.clip import load as load_pic2word
        model_path = "./pic2word/pic2word_model.pt"
        temp_clip, preprocess_train_temp_clip, preprocess_val_temp_clip = load_pic2word('ViT-L/14', jit=False)
        convert_models_to_fp32(temp_clip)
        temp_clip = temp_clip.float()
        
        img2text = IM2TEXT(embed_dim=temp_clip.embed_dim, output_dim=temp_clip.token_embedding.weight.shape[1]).float().to('cuda')
        checkpoint = torch.load(model_path)
        sd = checkpoint["state_dict"]
        sd_img2text = checkpoint["state_dict_img2text"]
        sd = {k[len('module.'):]: v for k, v in sd.items()}
        sd_img2text = {k[len('module.'):]: v for k, v in sd_img2text.items()}
        temp_clip.load_state_dict(sd)
        img2text.load_state_dict(sd_img2text)
        temp_clip.eval()
        img2text.eval() 

        if dataset in ['imagenet_r', 'minidn']:
            text_query = ['a '+real_text[0]+' of *']
        elif dataset == 'nico':
            text_query = ['a * in '+real_text[0]]
        elif dataset in ['lueven', 'tokyo247']:
            text_query = ['a '+real_text[0]+' photo of *']
        else:
            text_query = ['* '+real_text[0]]
        print(text_query)

        text = []
        id_split = tokenize(["*"])[0][1]
        for p in text_query:
            text_tokens = tokenize(p)
            text.append(text_tokens)
            assert id_split in text_tokens
        text = torch.cat(text, dim=0).cuda(0, non_blocking=True)
        #transform = _transform(temp_clip.visual.input_resolution)

        query_features = []
        for idx, query in enumerate(current_query_paths):
            with torch.no_grad():
                print(idx, end='\r')
                query_img = preprocess_val_temp_clip(Image.open(query)).cuda(0, non_blocking=True)

                query_img = torch.unsqueeze(query_img, 0).float()
                img_feature = temp_clip.encode_image(query_img)
                query_img_feature = img2text(img_feature.float())
                composed_feature = temp_clip.encode_text_img_retrieval(text, query_img_feature, split_ind=id_split)
                query_descriptor = (composed_feature / composed_feature.norm(dim=-1, keepdim=True)).squeeze().detach()
                query_features.append(query_descriptor)

        query_features = torch.stack(query_features).float()
        sim_total = (query_features @ database_features.t())
        ranks = torch.argsort(sim_total, descending=True)

    if 'qe' in method.lower():
        knn = 5

        final_features = database_features[ranks[:,:knn]].mean(axis=1)
        final_features = final_features / torch.linalg.norm(final_features, axis=1, keepdims=True)
        sim_total = (final_features @ database_features.t())
        ranks = torch.argsort(sim_total, descending=True)
    return ranks.detach().cpu()

def metrics_calc(rankings, target_domain, current_query_classes, database_classes, database_domains, at):

    metrics = {}
    
    class_id_map = {class_name: idx for idx, class_name in enumerate(database_classes)}
    domain_id_map = {domain_name: idx for idx, domain_name in enumerate(database_domains)}

    database_classes_ids = [class_id_map[class_name] for class_name in database_classes]
    database_domains_ids = [domain_id_map[domain_name] for domain_name in database_domains]
    query_classes_ids = [class_id_map[class_name] for class_name in current_query_classes]
    target_domain_id = domain_id_map[target_domain]

    database_classes_tensor = torch.tensor(database_classes_ids).to(rankings.device)
    database_domains_tensor = torch.tensor(database_domains_ids).to(rankings.device)
    query_classes_tensor = torch.tensor(query_classes_ids).to(rankings.device)
    target_domain_tensor = torch.tensor(target_domain_id).to(rankings.device)

    class_tensor = (database_classes_tensor[rankings] == torch.unsqueeze(query_classes_tensor, 1).expand_as(rankings)).float()
    domain_tensor = (database_domains_tensor[rankings] == target_domain_tensor).float()

    correct = domain_tensor*class_tensor
    metrics[f"mAP"] = compute_map(correct.cpu().numpy())
    for k in at:
        correct_k = correct[:, :k]
        num_correct = torch.sum(correct_k, dim=1)
        num_predicted = torch.sum(torch.ones_like(correct_k), dim=1)
        num_total = torch.sum(correct, dim=1)
        recall = torch.mean(num_correct / (num_total+1e-5))
        precision = torch.mean(num_correct / (torch.minimum(num_total,num_predicted) + 1e-5))
        metrics[f"R@{k}"] = round(recall.item()*100, 2)
        metrics[f"P@{k}"] = round(precision.item()*100, 2)
        #metrics[f"mAP@{k}"]
    return metrics

def metrics_cirr_calc(rankings, val_query_filenames, val_query_target_filenames, val_database_filenames):
    metrics = {}
    metrics["Recall"] = {}
    metrics["mAP"] = {}
    sorted_index_names = np.array(val_database_filenames)[rankings]
    reference_mask = torch.tensor(sorted_index_names != np.repeat(np.array(val_query_filenames), len(val_database_filenames)).reshape(len(val_query_target_filenames), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], sorted_index_names.shape[1] - 1)
    labels = torch.tensor(sorted_index_names == np.repeat(np.array(val_query_target_filenames),len(val_database_filenames) - 1).reshape(len(val_query_target_filenames), -1))

    #pdb.set_trace()
    metrics[f"mAP"]['mAP'] = compute_map(labels.cpu().numpy())
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(val_query_target_filenames)).int())
    for k in [1, 5, 10, 50]:
        metrics["Recall"][f"R@{k}"] = (torch.sum(labels[:, :k]) / len(labels)).item() * 100
    return metrics

def metrics_fashion_calc(rankings, database_paths, query_answer_paths):
    metrics = {}
    # metrics["Recall"] = {}
    sorted_index_names = np.array(database_paths)[rankings]
    labels = torch.tensor(sorted_index_names == np.repeat(np.array(query_answer_paths), len(database_paths)).reshape(len(query_answer_paths), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(query_answer_paths)).int())
    metrics['mAP'] = compute_map(labels.cpu().numpy())
    for k in [10, 50]:
        metrics[f"R@{k}"] = (torch.sum(labels[:, :k]) / len(labels)).item() * 100
    return metrics

#examples of valid methods
methods = ["Text only", "Text only qe", "Add Similarities", "Add Similarities Cast", "Add Similarities Cast qe", "Add Similarities Norm",
           "Add Similarities Norm Cast", "Add Similarities Norm Cast qe", "Add Similarities Norm qe", "Add Similarities qe", "Image only",
           "Image only qe", "Minimum Similarity", "Minimum Similarity Cast", "Minimum Similarity Cast qe", "Minimum Similarity Norm",
           "Minimum Similarity Norm Cast", "Minimum Similarity Norm Cast qe", "Minimum Similarity Norm qe", "Minimum Similarity qe",
           "Multiply Similarities", "Multiply Similarities Cast", "Multiply Similarities Cast qe", "Multiply Similarities Norm",
           "Multiply Similarities Norm Cast", "Multiply Similarities Norm Cast qe", "Multiply Similarities Norm qe", "Multiply Similarities qe",
           "Super-Text w Corpus mNN 20k clean labels miu=1",  "Super-Text w Corpus mNN 20k clean labels miu=2", "Super-Text w Corpus mNN 20k clean labels miu=3",
           "Super-Text w Corpus mNN 20k clean labels miu=4", "Super-Text w Corpus mNN 20k clean labels miu=5", "Super-Text w Corpus mNN 20k clean labels miu=6",
           "Super-Text w Corpus mNN 20k clean labels miu=7", "Super-Text w Corpus mNN 20k clean labels miu=8", "Super-Text w Corpus mNN 20k clean labels miu=9",
           "Super-Text w Corpus mNN 20k clean labels miu=10", 
           "Super-Text w Corpus mNN 20k clean labels miu=1 Multi", "Super-Text w Corpus mNN 20k clean labels miu=2 Multi",
           "Super-Text w Corpus mNN 20k clean labels miu=3 Multi", "Super-Text w Corpus mNN 20k clean labels miu=4 Multi", "Super-Text w Corpus mNN 20k clean labels miu=5 Multi",
           "Super-Text w Corpus mNN 20k clean labels miu=6 Multi", "Super-Text w Corpus mNN 20k clean labels miu=7 Multi", "Super-Text w Corpus mNN 20k clean labels miu=8 Multi",
           "Super-Text w Corpus mNN 20k clean labels miu=9 Multi", "Super-Text w Corpus mNN 20k clean labels miu=10 Multi", 
           "Super-Text w Corpus mNN 20k miu=1", "Super-Text w Corpus mNN 20k miu=2", "Super-Text w Corpus mNN 20k miu=3", "Super-Text w Corpus mNN 20k miu=4",
           "Super-Text w Corpus mNN 20k miu=5", "Super-Text w Corpus mNN 20k miu=6", "Super-Text w Corpus mNN 20k miu=7", "Super-Text w Corpus mNN 20k miu=8",
           "Super-Text w Corpus mNN 20k miu=9", "Super-Text w Corpus mNN 20k miu=10", 
           "Super-Text w Corpus mNN 20k miu=1 Multi", "Super-Text w Corpus mNN 20k miu=2 Multi", "Super-Text w Corpus mNN 20k miu=3 Multi",
           "Super-Text w Corpus mNN 20k miu=4 Multi", "Super-Text w Corpus mNN 20k miu=5 Multi", "Super-Text w Corpus mNN 20k miu=6 Multi",
           "Super-Text w Corpus mNN 20k miu=7 Multi", "Super-Text w Corpus mNN 20k miu=8 Multi", "Super-Text w Corpus mNN 20k miu=9 Multi",
           "Super-Text w Corpus mNN 20k miu=10 Multi",
           "searle iter=10 lr=0.0002", "searle iter=10 lr=0.002", "searle iter=10 lr=0.02", "searle iter=10 lr=0.2", "searle iter=200 lr=0.0002",
           "searle iter=200 lr=0.002", "searle iter=200 lr=0.02", "searle iter=200 lr=0.2", "searle iter=350 lr=0.0002", "searle iter=350 lr=0.002",
           "searle iter=350 lr=0.02", "searle iter=350 lr=0.2", "searle iter=5 lr=0.0002", "searle iter=5 lr=0.002", "searle iter=5 lr=0.02",
           "searle iter=5 lr=0.2", "searle iter=50 lr=0.0002", "searle iter=50 lr=0.002", "searle iter=50 lr=0.02", "searle iter=50 lr=0.2",
           "searle iter=500 lr=0.0002", "searle iter=500 lr=0.002", "searle iter=500 lr=0.02", "searle iter=500 lr=0.2",
           "pic2word"]

mode = 'pacs'
print(vars(args)['dataset'])
if vars(args)['dataset'] is not None:
    if vars(args)['dataset'].lower() in ['in', 'imgnet', 'imagenet', 'imagenet_r', 'imagenet-r', 'imagenetr']:
        mode = 'imagenet_r'
    elif vars(args)['dataset'].lower() in ['nico', 'nico++']:
        mode = 'nico'
    elif vars(args)['dataset'].lower() in ['pacs']:
        mode = 'pacs'
    elif vars(args)['dataset'].lower() in ['dn', 'minidn', 'domainnet', 'minidomainnet']:
        mode = 'minidn'
    elif vars(args)['dataset'].lower() in ['lueven']:
        mode = 'lueven'
    elif vars(args)['dataset'].lower() in ['tokyo', 'tokyo247']:
        mode = 'tokyo247'
    elif vars(args)['dataset'].lower() in ['cirr']:
        mode = 'cirr'
    elif vars(args)['dataset'].lower() in ['fashion']:
        mode = 'fashion'

model, preprocess_val = load('ViT-L/14', jit=False)
model = model.eval()
root = "/mnt/personal/efthynik/2023_Composed_Image_retrieval"

'''
disentangle_loss = [3.2, 3.5, 3.7, 4.0]
close_to_image_loss = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
lr = [0.01]
methods = ["Test 20k inversion miu=5 late clean labels iter=10 d_loss=" + str(x) for x in  disentangle_loss]
methods = [method + ' ci_loss=' + str(x) for method in methods for x in  close_to_image_loss]
methods = [method + ' lr=' + str(x) for method in methods for x in  lr]

#methods = ["Super-Text w Corpus mNN 20k clean labels miu=1 Multi invfreq den=2.0"]
#denominator = [0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
#methods = [x+' invfreq' for x in methods]
#methods = [method + ' den=' + str(x) for method in methods for x in  denominator]
'''

#methods = ["Super-Image w Corpus 20k nrm multiplication", "Super-Image w Corpus 20k nrm add", "Super-Image w Corpus 20k nrm min",]
#methods = ["Super-Image w Corpus 20k nrm multiplication", "Super-Image w Corpus 20k nrm add"]
#methods = ["Super-Image w Corpus 20k nrm multiplication"]
methods = ["Super-Text w Corpus mNN 20k clean labels weighted miu=1 Multi", "Super-Text w Corpus mNN 20k clean labels weighted miu=2 Multi",
           "Super-Text w Corpus mNN 20k clean labels weighted miu=3 Multi", "Super-Text w Corpus mNN 20k clean labels weighted miu=4 Multi", 
           "Super-Text w Corpus mNN 20k clean labels weighted miu=5 Multi", "Super-Text w Corpus mNN 20k clean labels weighted miu=6 Multi", 
           "Super-Text w Corpus mNN 20k clean labels weighted miu=7 Multi", "Super-Text w Corpus mNN 20k clean labels weighted miu=8 Multi",
           "Super-Text w Corpus mNN 20k clean labels weighted miu=9 Multi", "Super-Text w Corpus mNN 20k clean labels weighted miu=10 Multi",
           "Super-Text w Corpus mNN 20k clean labels weighted miu=11 Multi", "Super-Text w Corpus mNN 20k clean labels weighted miu=12 Multi",
           "Super-Text w Corpus mNN 20k clean labels weighted miu=13 Multi", "Super-Text w Corpus mNN 20k clean labels weighted miu=14 Multi", 
           "Super-Text w Corpus mNN 20k clean labels weighted miu=15 Multi", "Super-Text w Corpus mNN 20k clean labels weighted miu=16 Multi", 
           "Super-Text w Corpus mNN 20k clean labels weighted miu=17 Multi", "Super-Text w Corpus mNN 20k clean labels weighted miu=18 Multi",
           "Super-Text w Corpus mNN 20k clean labels weighted miu=19 Multi", "Super-Text w Corpus mNN 20k clean labels weighted miu=20 Multi"]
best_method = ["Super-Text w Corpus mNN 20k Multi clean labels weighted miu=7 ni=7 kappa=20"]

methods = ["Text only", "Image only", "Add Similarities", "Add Similarities Norm", 
           "Super-Text w Corpus mNN 20k Multi clean labels weighted miu=7 ni=7 kappa=20", 
           "pic2word"]
methods = ["searle iter=350 lr=0.0002 oti"]
methods = ["Super-Text w Corpus mNN 20k Multi clean labels weighted miu=7 ni=7 kappa=20"]
#,
# "Super-Text w Corpus mNN 20k Multi clean labels weighted miu=9 ni=7 kappa=1",
# "Super-Text w Corpus mNN 20k Multi clean labels weighted miu=9 ni=9 kappa=1",
# "Super-Text w Corpus mNN 20k Multi clean labels weighted miu=7 ni=9 kappa=1",
# "Super-Text w Corpus mNN 20k Multi clean labels weighted miu=5 ni=9 kappa=1",
# ]

in_corpus_feats, in_corpus_names = read_corpus_features(f'{root}/clip_features/corpus/imagenet_names.pkl')
open_images_corpus_feats, open_images_corpus_names = read_corpus_features(f'{root}/clip_features/corpus/open_image_v7_class_names.pkl')
nltk_corpus_feats, nltk_corpus_names = read_corpus_features(f'{root}/clip_features/corpus/nltk_words.pkl')
if any("40m" in string for string in methods):
    laion_corpus_feats, laion_corpus_names = read_laion_corpus(mode)

metrics = {}
for method in methods:
    metrics[method] = {}
if mode == 'imagenet_r':
    query_image_features, query_paths, query_classes, query_domains = read_dataset_features(root+'/clip_features/imagenet_r/full_imgnet_features2.pkl')
    database_image_features, database_paths, database_classes, database_domains = read_dataset_features(root+'/clip_features/imagenet_r/full_imgnet_features2.pkl')
    #database_cls_features, database_patch_features, database_atention_6, database_atention_24, database_paths, database_classes, database_domains = read_patch_dataset_features(root+'/clip_features/imagenet_r/full_imgnet_patch_features.pkl')
    #query_cls_features, query_patch_features, query_atention_6, query_atention_24, query_paths, query_classes, query_domains = read_patch_dataset_features(root+'/clip_features/imagenet_r/full_imgnet_patch_features.pkl')
    domains = ['real', 'cartoon', 'origami', 'toy', 'sculpture']
    domain_change = {'real': 'photo', 'cartoon': 'cartoon', 'origami': 'origami', 'toy': 'toy', 'sculpture': 'sculpture'}
    at = [10, 50]
elif mode == 'nico':
    query_image_features, query_paths, query_classes, query_domains = read_dataset_features(root+'/clip_features/nico/query_nico_features.pkl')
    database_image_features, database_paths, database_classes, database_domains = read_dataset_features(root+'/clip_features/nico/database_nico_features.pkl')
    #database_cls_features, database_patch_features, database_atention_6, database_atention_24, database_paths, database_classes, database_domains = read_patch_dataset_features(root+'/clip_features/nico/database_nico_patch_features.pkl')
    #query_cls_features, query_patch_features, query_atention_6, query_atention_24, query_paths, query_classes, query_domains = read_patch_dataset_features(root+'/clip_features/nico/query_nico_patch_features.pkl')
    domains = ['autumn', 'dim', 'grass', 'outdoor', 'rock', 'water']
    domain_change = {'autumn': 'autumn', 'dim': 'dimlight', 'grass': 'grass', 'outdoor': 'outdoor', 'rock': 'rock', 'water': 'water'}
    at = [50, 100, 150, 200]
elif mode == 'minidn':
    query_image_features, query_paths, query_classes, query_domains = read_dataset_features(root+'/clip_features/minidn/query_minidn_features.pkl')
    database_image_features, database_paths, database_classes, database_domains = read_dataset_features(root+'/clip_features/minidn/database_minidn_features.pkl')
    #database_cls_features, database_patch_features, database_atention_6, database_atention_24, database_paths, database_classes, database_domains = read_patch_dataset_features(root+'/clip_features/minidn/database_minidn_patch_features.pkl')
    #query_cls_features, query_patch_features, query_atention_6, query_atention_24, query_paths, query_classes, query_domains = read_patch_dataset_features(root+'/clip_features/minidn/query_minidn_patch_features.pkl')
    domains = ['clipart', 'painting', 'real', 'sketch']
    domain_change = {'clipart': 'clipart', 'painting': 'painting', 'real': 'photo', 'sketch': 'sketch'}
    at = [50, 100, 150, 200]
elif mode == 'lueven':
    query_image_features, query_paths, query_classes, query_domains = read_dataset_features(root+'/clip_features/lueven/full_lueven_features.pkl')
    database_image_features, database_paths, database_classes, database_domains = read_dataset_features(root+'/clip_features/lueven/full_lueven_features.pkl')
    #database_cls_features, database_patch_features, database_atention_6, database_atention_24, database_paths, database_classes, database_domains = read_patch_dataset_features(root+'/clip_features/lueven/full_lueven_patch_features.pkl')
    #query_cls_features, query_patch_features, query_atention_6, query_atention_24, query_paths, query_classes, query_domains = read_patch_dataset_features(root+'/clip_features/lueven/full_lueven_patch_features.pkl')
    domains = ['New', 'Old']
    #domains = ['Old', 'New']
    domain_change = {'New': 'Today', 'Old': 'Archive'}
    at = [5, 10, 20]
elif mode == 'tokyo247':
    query_image_features, query_paths, query_classes, query_domains = read_dataset_features(root+'/clip_features/tokyo247/full_tokyo247_features.pkl')
    database_image_features, database_paths, database_classes, database_domains = read_dataset_features(root+'/clip_features/tokyo247/full_tokyo247_features.pkl')
    #database_cls_features, database_patch_features, database_atention_6, database_atention_24, database_paths, database_classes, database_domains = read_patch_dataset_features(root+'/clip_features/tokyo247/full_tokyo247_patch_features.pkl')
    #query_cls_features, query_patch_features, query_atention_6, query_atention_24, query_paths, query_classes, query_domains = read_patch_dataset_features(root+'/clip_features/tokyo247/full_tokyo247_patch_features.pkl')
    domains = ['Day', 'Night']
    domain_change = {'Day': 'daytime', 'Night': 'nighttime'}
    at = [1, 5]
elif mode == 'pacs':
    query_image_features, query_paths, query_classes, query_domains = read_dataset_features(root+'/clip_features/PACS/query_pacs_features.pkl')
    database_image_features, database_paths, database_classes, database_domains = read_dataset_features(root+'/clip_features/PACS/database_pacs_features.pkl')
    #database_cls_features, database_patch_features, database_atention_6, database_atention_24, database_paths, database_classes, database_domains = read_patch_dataset_features(root+'/clip_features/tokyo247/full_tokyo247_patch_features.pkl')
    #query_cls_features, query_patch_features, query_atention_6, query_atention_24, query_paths, query_classes, query_domains = read_patch_dataset_features(root+'/clip_features/tokyo247/full_tokyo247_patch_features.pkl')
    domains = ['photo', 'art_painting', 'cartoon', 'sketch']
    domain_change = {'photo': 'photo', 'art_painting': 'art_painting', 'cartoon': 'cartoon', 'sketch': 'sketch'}
    at = [1, 5]
elif mode == 'cirr':
    val_query_features, val_query_filenames, val_query_caption_features, val_actual_captions, val_query_target_filenames = read_cirr_features(root+'/clip_features/cirr/cirr_val_query_features.pkl', read_mode='val query')
    val_database_features, val_database_filenames = read_cirr_features(root+'/clip_features/cirr/cirr_val_database_features.pkl', read_mode='val database')
    test_query_features, test_query_filenames, test_query_caption_features, test_actual_captions, test_pair_ids = read_cirr_features(root+'/clip_features/cirr/cirr_test_query_features.pkl', read_mode='test query')
    test_database_features, test_database_filenames = read_cirr_features(root+'/clip_features/cirr/cirr_test_database_features.pkl', read_mode='test database')
    #pdb.set_trace()
    for method in methods:
        time1 = time.time()
        rankings = calculate_rankings(method, val_query_features, val_query_caption_features, val_actual_captions, val_database_features)
        metrics[method] = metrics_cirr_calc(rankings, val_query_filenames, val_query_target_filenames, val_database_filenames)
        print("seconds: {:4.1f} {:40s} R@1 {:5.2f} R@5 {:5.2f} R@10 {:5.2f} R@50 {:5.2f}".format(round(time.time()-time1, 1), method, metrics[method]['Recall'][f'R@1'], metrics[method]['Recall'][f'R@5'], metrics[method]['Recall'][f'R@10'], metrics[method]['Recall'][f'R@50']))
    dict_to_csv(metrics, mode+'_'+time.strftime("%Y_%m_%d_%H_%M_%S")+'.csv')
    sys.exit()
elif mode == 'fashion':
    clothes = ['dress', 'shirt', 'toptee']
    for cloth in clothes:
        print(cloth)
        # metrics[cloth] = {}
        database_feats, database_paths = read_fashion(root+'/clip_features/fashion/fashion_'+cloth+'_database_features.pkl', 'database')
        query_image_feats, query_image_paths, query_answer_paths, actual_caption_1, actual_caption_2, actual_caption_comb, caption_1_features, caption_2_features, caption_comb_features = read_fashion(root+'/clip_features/fashion/fashion_'+cloth+'_query_features.pkl', 'query')
        for method in methods:
            # metrics[cloth][method] = {}
            time1 = time.time()
            rankings = calculate_rankings(method, query_image_feats, caption_comb_features, actual_caption_comb, database_feats)
            # metrics[cloth][method] = metrics_fashion_calc(rankings, database_paths, query_answer_paths)
            metrics[method][cloth] = metrics_fashion_calc(rankings, database_paths, query_answer_paths)
            # print("seconds: {:4.1f} {:40s} R@10 {:5.2f} R@50 {:5.2f}".format(round(time.time()-time1, 1), method, metrics[method][cloth]['Recall'][f'R@10'], metrics[method][cloth]['Recall'][f'R@50']))
            # print("seconds: {:4.1f} {:40s} R@10 {:5.2f} R@50 {:5.2f}".format(round(time.time()-time1, 1), method, metrics[cloth][method]['Recall'][f'R@10'], metrics[cloth][method]['Recall'][f'R@50']))
    dict_to_csv(metrics, mode+'_'+time.strftime("%Y_%m_%d_%H_%M_%S")+'.csv')
    sys.exit()
    
domains = replace_domain_names(domains, domain_change)
query_domains = replace_domain_names(query_domains, domain_change)
database_domains = replace_domain_names(database_domains, domain_change)
source = domains
target = domains

if vars(args)['source'] is not None:
    source = [vars(args)['source']]
if vars(args)['target'] is not None:
    target = [vars(args)['target']]

for idx1, source_domain in enumerate(domains):
    print(source_domain, source)
    if source_domain in source:
        current_query_patch_features, current_query_atention_6, database_patch_features = None, None, None
        current_indices = [i for i, item in enumerate(query_domains) if item == source_domain]
        current_query_features = query_image_features[current_indices, :]
        current_query_paths = [query_paths[i] for i in current_indices]
        #current_query_patch_features = query_patch_features[current_indices, :, :]
        #current_query_atention_6 = query_atention_6[current_indices, :]
        current_query_classes = [query_classes[i] for i in current_indices]
        current_query_domains = [query_domains[i] for i in current_indices]

        if 'laioni' in method.lower():
            with open(root+'/clip_features/laion_40m/'+mode+'_laion_image_feats.pkl', 'rb') as f:
                laion_feats = pickle.load(f)
            laion_feats = torch.from_numpy(laion_feats['feats'].astype("float32")).float().to('cuda')
            laion_feats = laion_feats / laion_feats.norm(dim=-1, keepdim=True)
            if 'laioni only' not in method.lower():
                laion_feats = torch.cat((laion_feats, database_image_features))
            start = time.time()
            sim_with_laion_img = (current_query_features @ laion_feats.t()).cpu()
            ranks_laion_img = torch.argsort(sim_with_laion_img, descending=True)
            print(time.time()-start)

        else:
            laion_feats = None
            sim_with_laion_img = None
            ranks_laion_img = None
        print(domains)
        for idx2, target_domain in enumerate(domains):
            if target_domain in target and idx1 != idx2:
                text = tokenize(target_domain).to('cuda')
                text_feature = model.encode_text(text)
                text_feature = (text_feature / text_feature.norm(dim=-1, keepdim=True)).squeeze().detach().to(torch.float32)
                text_feature = text_feature.repeat((len(current_query_classes), 1))
                real_text =len(current_query_classes)*[target_domain]
                for method in methods:
                    time1 = time.time()
                    if "searle" in method.lower():
                        if "oti" in method.lower():
                            lr = extract_float(method, "lr=")
                            iters = extract_int(method, "iter=")
                            suffix = f"{lr}_{iters}"
                            searle_mode = "oti"
                        else:
                            suffix = "model"
                            searle_mode = "model"

                        rankings = searle_rankings(mode, database_image_features, target_domain, source_domain, current_query_paths, query_paths, searle_mode, suffix, topk=args.topk, prompt_format=args.prompt_format)
                    else:

                        rankings = calculate_rankings(method, current_query_features, text_feature, real_text, database_image_features, db_patch_features=database_patch_features, q_patch_features=current_query_patch_features,
                                                        laion_feats=laion_feats, sim_with_laion_img=sim_with_laion_img, ranks_laion_img=ranks_laion_img, database_paths=database_paths, actual_label_names=current_query_classes, dataset=mode)

                        save_images = False
                        if save_images:
                            # TODO: ONCE I HAVE THE FILES CHECK THAT IT SAVES EVERYTHING CORRECTLY!
                            # Set the number of rows (q) and the number of retrieved images (top_k)
                            #import random
                            #randomlist = []
                            #for i in range(0,100):
                            #    n = random.randint(1,9999)
                            #    randomlist.append(n)
                            q = 25
                            top_k = 20

                            # Create a folder named 'retrievals' if it doesn't exist
                            output_folder = 'retrievals_pic2word_nico'
                            os.makedirs(output_folder, exist_ok=True)

                            # Randomly choose q rows from the rankings tensor
                            # Select the first q rows
                            #selected_rows = torch.randint(0, len(rankings), (q,))
                            selected_rows = torch.arange(0, q)

                            for i, row in enumerate(selected_rows):
                                # Get the corresponding query image path
                                query_image_path = current_query_paths[row]
                                
                                # Rename and save the query image
                                query_image = Image.open(query_image_path)
                                query_image.save(os.path.join(output_folder, f'{method}_{mode}_{source_domain}_{target_domain}_query_{i}.png'))
                                
                                # Get the top k retrieved image indexes for the current row
                                retrieved_indexes = rankings[row][:top_k]
                                
                                # Save each retrieved image
                                for j, index in enumerate(retrieved_indexes):
                                    retrieved_image_path = database_paths[index]
                                    retrieved_image = Image.open(retrieved_image_path)
                                    if retrieved_image.mode != 'RGB':
                                        retrieved_image = retrieved_image.convert('RGB')
                                    retrieved_image.save(os.path.join(output_folder, f'{method}_{mode}_{source_domain}_{target_domain}_query_{i}_retrieved_{j}.png'))

                    metrics[method][source_domain+'-->'+target_domain] = metrics_calc(rankings, target_domain, current_query_classes, database_classes, database_domains, at)
                    print(round(time.time()-time1, 1), source_domain+'-->'+target_domain, method, metrics[method][source_domain+'-->'+target_domain])
dict_to_csv(metrics, mode+'_'+time.strftime("%Y_%m_%d_%H_%M_%S")+'.csv')