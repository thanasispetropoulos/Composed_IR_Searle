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
import pandas as pd
import os,sys
from pathlib import Path
from text_inversion import *

def dict_to_csv(data, directory):
    flat_data = []
    for top_level_key, top_level_value in data.items():
        for second_level_key, second_level_value in top_level_value.items():
            row = {'Method': top_level_key, 'source-->target': second_level_key}
            row.update(second_level_value)
            flat_data.append(row)

    df = pd.DataFrame(flat_data)
    pivoted_df = df.pivot(index='Method', columns='source-->target')
    pivoted_df.columns = pivoted_df.columns.swaplevel(0, 1)
    pivoted_df.sort_index(axis=1, level=0, inplace=True)
    average = pivoted_df.groupby(level=1,axis=1).mean()
    average.columns = pd.MultiIndex.from_product([["Avg."], average.columns])
    result = pd.concat([average, pivoted_df], axis=1)
    result = result.round(2)
    result.to_csv(directory)

def read_and_crop(img, new_width, new_height):
    img = Image.open(img).convert("RGB")
    width, height = img.size

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    img = img.crop((left, top, right, bottom))

    return img

def gridify(image_list_1, image_list_2):

    new_width, new_height = 224, 224

    total = len(image_list_1) + len(image_list_2)
    root = math.ceil(total**0.5)
    grid_sz = root**2
    grid_img = np.zeros((root*new_width, root*new_height,3))
    diff = total-grid_sz

    img_list_1 = []
    for img in image_list_1:
        img = read_and_crop(img, new_width, new_height)
        img_list_1.append(np.array(img))
    img_list_2 = []
    for img in image_list_2:
        img = read_and_crop(img, new_width, new_height)
        img_list_2.append(np.array(img))

    # grid
    img_list = []
    idx = 0
    while len(img_list)<root*root:
        img_list.append(img_list_1[idx%len(img_list_1)])
        if len(img_list)<root*root:
            img_list.append(img_list_2[idx%len(img_list_2)])
        idx += 1
    row, col = 0, 0
    for idx, img in enumerate(img_list):
        if col >= root:
            row += 1
            col = 0 
        offset_row = row*new_height
        offset_col = col*new_height
        grid_img[offset_row:offset_row+new_height,offset_col:offset_col+new_height] = img
        col+=1
    
    return grid_img

def text_inversion(class_texts, real_text, ranks, model, image_features):
        import argparse
        args = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr', type=float, default=0.01,
                            help="lr")
        parser.add_argument('--iters', type=int, default=10,
                            help="opt iters")
        parser.add_argument('--alpha', type=float, default=0.8,
                            help="ema rate")
        parser.add_argument('--weight_decay', type=float, default=0.01,
                            help="weight decay")
        parser.add_argument('--init', type=str, default="knn",
                            help="initialize token emb", choices=["rnd", "knn"])

        args = parser.parse_args(args)

        # gather optim vars
        optim_vars = {}
        optim_vars["lr"] = args.lr
        optim_vars["iters"] = args.iters
        optim_vars["alpha"] = args.alpha
        optim_vars["init"] = args.init
        optim_vars["weight_decay"] = args.weight_decay

        from text_inversion import ti_opt, compose_token
        
        # class_texts = [' '.join([real_corpus_text[index] for index in row]) for row in ranks]
        text_features = []
        for c, text in enumerate(class_texts):
            # print(c)
            # text = ' '.join(real_corpus_text[index] for index in ranks[0])
            token_opt_emb = ti_opt(text, image_features[c], model, optim_vars)
            text_feature = compose_token(text + ' ' + real_text[c], token_opt_emb, model)
            text_features.append(text_feature.squeeze())
            
        text_features = torch.stack(text_features).float()
        return text_features #, domain_text_features, class_text_features

def text_inv_converger(class_texts, real_text, model, image_features, text_corpus, text_corpus_features, index, 
                       lr, iters, miu, d_loss, ci_loss):
        import argparse
        args = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr', type=float, default=0.01, help="lr")
        parser.add_argument('--iters', type=int, default=50, help="opt iters")
        parser.add_argument('--alpha', type=float, default=0.8,
                            help="ema rate")
        parser.add_argument('--weight_decay', type=float, default=0.0,
                            help="weight decay")
        parser.add_argument('--init', type=str, default="knn",
                            help="initialize token emb", choices=["rnd", "knn"])
        parser.add_argument('--dataset', type=str)
        parser.add_argument('--disentangle_loss', type=float)
        parser.add_argument('--close_to_image_loss', type=float)


        args = parser.parse_args(args)

        # gather optim vars
        optim_vars = {}
        optim_vars["lr"] = lr
        optim_vars["iters"] = iters
        optim_vars["d_loss"] = d_loss
        optim_vars["ci_loss"] = ci_loss
        optim_vars["alpha"] = args.alpha
        optim_vars["init"] = args.init
        optim_vars["weight_decay"] = args.weight_decay

        text_features = []
        text_out_individual = [[] for _ in range(miu)]
        text_out_all = []

        for c, text in enumerate(class_texts):
            # print(c)
            #print(c, end='\r')
            token_opt_emb = ti_opt_converger(text, image_features[c], model, optim_vars, text_corpus, text_corpus_features, index)
            for idx, token in enumerate(token_opt_emb):
                text_out_individual[idx].append(compose_token(text.split(' ')[idx].replace('_', ' ') + ' ' + real_text[c], token, model))
            text_out_all.append(compose_token(text.replace('_', ' ') + ' ' + real_text[c], torch.vstack(token_opt_emb), model))

        text_features_all = torch.stack(text_out_all).float().squeeze(1)
        text_features_individual = []
        for idx in range(len(text_out_individual)):
            text_features_individual.append(torch.stack(text_out_individual[idx]).float().squeeze(1))
        return text_features_all, text_features_individual

def text_list_to_features(model, text_list, batch_size=1):
    text_features = []
    text_tokens = []
    num_batches = len(text_list) // batch_size

    for i in range(num_batches + 1):
        if i == num_batches:
            batch = text_list[i * batch_size:]
        else:
            batch = text_list[i * batch_size:(i + 1) * batch_size]
        text_tokens = tokenize(batch).to('cuda')
        text_feature = model.encode_text(text_tokens)
        text_feature = text_feature / text_feature.norm(dim=1, keepdim=True)
        text_feature = text_feature.detach().to(torch.float32)
        text_features.append(text_feature)

    return torch.cat(text_features, dim=0)

def split_string_into_n_strings(original_string, n):
    words = original_string.split()
    total_words = len(words)
    
    if n <= 0:
        raise ValueError("Number of strings must be greater than zero.")
    
    words_per_string = total_words // n
    remainder_words = total_words % n
    
    split_strings = []
    start_index = 0
    
    for i in range(n):
        end_index = start_index + words_per_string
        if remainder_words > 0:
            end_index += 1
            remainder_words -= 1
        
        split_strings.append(' '.join(words[start_index:end_index]))
        start_index = end_index
    
    return split_strings

def norm_cdf(matrix):
    row_means = matrix.mean(dim=1, keepdim=True)
    row_stds = matrix.std(dim=1, keepdim=True)
    matrix = 0.5 * (1 + torch.erf((matrix - row_means) / (row_stds * torch.sqrt(torch.tensor(2.0)))))
    return matrix

def norm_gaussian(matrix):
    row_means = matrix.mean(dim=1, keepdim=True)
    row_stds = matrix.std(dim=1, keepdim=True)
    matrix = (matrix-row_means)/row_stds
    return matrix

def extract_int(input_string, prefix):
    pattern = rf'{prefix}(\d+)'
    match = re.search(pattern, input_string)
    if match:
        number = int(match.group(1))
        return number
    else:
        return None

def extract_float(input_string, prefix):
    pattern = rf'{prefix}(\d+.\d+)'
    match = re.search(pattern, input_string)
    if match:
        number = float(match.group(1))
        return number
    else:
        return None

def calculate_ranks(input_tensor):
    num_rows, num_cols = input_tensor.shape
    ranked_tensor = torch.zeros_like(input_tensor, dtype=torch.int64).to('cuda')
    for i in range(num_rows):
        row = input_tensor[i]
        sorted_indices = torch.argsort(row, descending=True)
        ranked_tensor[i, sorted_indices] = torch.arange(1, num_cols + 1).to('cuda')
    return(ranked_tensor)

def list_from_tensor_index(indices, list0):
    result = []
    for row in indices:
        row_result = [list0[idx] for idx in row]
        result.append(row_result)
    return result

def replace_domain_names(input_list, mapping_dict):
    updated_list = []
    for element in input_list:
        if element in mapping_dict:
            updated_list.append(mapping_dict[element])
        else:
            updated_list.append(element)
    return updated_list

def img2text_list(image_features, text_corpus_features, real_corpus_text, ni):
    d = image_features.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatIP(res, d, flat_config)
    index.add(text_corpus_features)
    sim, ranks = index.search(np.array(image_features.cpu().detach().numpy().astype("float32")), ni)
    class_text = [[real_corpus_text[index] for index in row] for row in ranks]
    class_vectors = [[text_corpus_features[index] for index in row] for row in ranks]
    return class_text, class_vectors, sim

def get_unique_indices(list1, list2):
    unique_indices = set()
    new_list1 = []
    new_list2 = []

    for idx, item in enumerate(list1):
        if item not in new_list1:
            new_list1.append(list1[idx])
            new_list2.append(list2[idx])

    return new_list1, new_list2

def create_dict_from_lists(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same size")

    my_dict = {}  # Initialize an empty dictionary

    for key, value in zip(list1, list2):
        my_dict[key] = value

    return my_dict

def keep_k_most_frequent(nested_list_of_text, miu):
    labels = []
    weights = []
    for idx1, querry in enumerate(nested_list_of_text):
        text_counts = {}
        for idx2, query_neighbors in enumerate(querry):
            for idx3, text in enumerate(query_neighbors):
                if text in text_counts:
                    text_counts[text] += 1
                else:
                    text_counts[text] = 1
        most_common_texts = sorted(text_counts.keys(), key=lambda x: text_counts[x], reverse=True)[:miu]
        most_common_texts_values = [text_counts[key] for key in most_common_texts]
        most_common_texts_values = [x/max(most_common_texts_values) for x in most_common_texts_values]
        labels.append(most_common_texts)
        weights.append(most_common_texts_values)
    return labels, weights

def clean_labels_from_images(sim_img, text_features, database_features, text_corpus_features, real_corpus_text, real_text, kappa, miu, ni, temp_image, temp_label,
                             multi=False, inversion=False, converger=False, model=None, image_features=None, 
                             lr=None, iters=None, d_loss=None, ci_loss=None, weighted=False, laion_feats=None, img_ranks=None):
    
    # text_list_from_text = [x.split(' ') for x in real_text]
    text_list_from_text = [[x] for x in real_text]
    ranks_img = img_ranks[:, :kappa].cpu()

    if laion_feats is None:
        image_and_neighbor_features = database_features.cpu()[ranks_img]
    else:
        image_and_neighbor_features = laion_feats.cpu()[ranks_img]

    idx = torch.arange(ranks_img.shape[0]).unsqueeze(1).expand(-1, ranks_img.shape[1])

    text_list_from_img = []
    class_vectors_list = []
    start = time.time()
    for idx in range(image_and_neighbor_features.shape[1]):

        class_text, class_vectors, _ = img2text_list(image_and_neighbor_features[:,idx,:], text_corpus_features, real_corpus_text, ni)
        class_vectors_list.append(class_vectors)
        text_list_from_img.append(class_text)

    text_list_from_img = [[text_list_from_img[j][i] for j in range(len(text_list_from_img))] for i in range(len(text_list_from_img[0]))]

    class_vectors_list = [[class_vectors_list[j][i] for j in range(len(class_vectors_list))] for i in range(len(class_vectors_list[0]))]

    text_list_from_img, weights = keep_k_most_frequent(text_list_from_img, miu)


    for idx in range(len(text_list_from_img)):
        while len(text_list_from_img[idx]) < max([len(x) for x in text_list_from_img]):
            text_list_from_img[idx].append('')
    for idx in range(len(weights)):
        while len(weights[idx]) < max([len(x) for x in weights]):
            weights[idx].append(0)

    if inversion and converger:
        class_text = [' '.join([word.replace(' ', '_') for word in line]) for line in text_list_from_img]

        d = image_features.shape[1]
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        index = faiss.GpuIndexFlatIP(res, d, flat_config)
        index.add(text_corpus_features)

        text_features_all, text_features_individual = text_inv_converger(class_text, real_text, model, image_features, 
                                                                             real_corpus_text, text_corpus_features, index, 
                                                                             lr, iters, miu, d_loss, ci_loss)
        return text_features_all, text_features_individual
    if multi:
        return text_to_multi(text_list_from_text, text_list_from_img), invert_levels(weights)
    else:
        text_class = [' '.join(sublist) for sublist in text_list_from_img]
        text_domain = [' '.join(sublist) for sublist in text_list_from_text]
        text = [text_class[idx] + ' ' + text_domain[idx] for idx in range(len(text_list_from_img))]
        return text_list_from_img, text

def text_to_multi(text_list_from_text, text_list_from_img):
    text = [[] for _ in range(len(text_list_from_img))]
    for idx in range(len(text_list_from_text)):
        for domain in text_list_from_text[idx]:
            for clas in text_list_from_img[idx]:
                text[idx].append(clas + ' ' + domain)
    return invert_levels(text)

def img2text_viacorpus(image_features, text_corpus_features, real_corpus_text, real_text, model, miu = 3, multi=False, inversion=False, converger=False, 
                       lr=None, iters=None, d_loss=None, ci_loss=None, weighted=False, temp_label=1):
    d = image_features.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatIP(res, d, flat_config)
    index.add(text_corpus_features)
    label_sim, ranks = index.search(np.array(image_features.cpu().detach().numpy().astype("float32")), miu)
    
    if multi:
        # domain_list_text = [x.split(' ') for x in real_text]
        domain_list_text = [[x] for x in real_text]
        class_text = [[real_corpus_text[index] for index in row] for row in ranks]
        multi_text_queries = []
        if inversion:
            multi_text_features = []
            for idx1, domains in enumerate(domain_list_text[0]):
                for idx2, labels in enumerate(class_text[0]):
                    class_text_q = [x[idx2].replace(' ', '_') for x in class_text]
                    real_text_q = [x[idx1] for x in domain_list_text]
                    multi_text_features.append(text_inversion(class_text_q, real_text_q, ranks, model, image_features))
            return multi_text_features
        else:
            for idx1, domains in enumerate(domain_list_text[0]):
                for idx2, labels in enumerate(class_text[0]):
                    #pdb.set_trace()
                    multi_text_queries.append([class_text[idx][idx2] + ' ' + domain_list_text[idx][idx1] for idx in range(len(domain_list_text))])
            multi_text_features = []

            weights = []
            for idx in range(len(label_sim)):
                if weighted:
                    current_label_sim = temp_label*label_sim[idx]
                    current_label_sim = np.exp(current_label_sim)/np.exp(current_label_sim).sum()
                    weights.append(current_label_sim)
                else:
                    weights.append(np.ones_like(label_sim[idx]))
            weights = [arr.tolist() for arr in weights]
            for multi_text_querie in multi_text_queries:
                multi_text_features.append(text_list_to_features(model, multi_text_querie, 256))

            return multi_text_features, multi_text_queries, invert_levels(weights)
    else:
        if inversion:
            class_text = [' '.join([real_corpus_text[index].replace(' ', '_') for index in row]) for row in ranks]
            if converger:
                text_features_all, text_features_individual = text_inv_converger(class_text, real_text, model, image_features, 
                                                                                 real_corpus_text, text_corpus_features, index, 
                                                                                 lr, iters, miu, d_loss, ci_loss)
                return text_features_all, text_features_individual
            else:
                text_features = text_inversion(class_text, real_text, ranks, model, image_features)
        else:
            final_text_query = [' '.join([real_corpus_text[index] for index in row]) + ' ' + real_text[idx] for idx, row in enumerate(ranks)]
            text_features = text_list_to_features(model, final_text_query, 256)
        return text_features

# def searle_rankings(dataset, database_features, target_domain, source_domain, current_query_paths, all_query_paths=None, suffix=None):
#     assert suffix is not None
#     # if multiple domain names
#     if len(target_domain.split(' ')) > 1:
#         target_domain_join = '_'.join(target_domain.split(' '))
#     else:
#         target_domain_join = target_domain
#     if len(source_domain.split(' ')) > 1:
#         source_domain_join = '_'.join(source_domain.split(' '))
#     else:
#         source_domain_join = source_domain

#     searl_path = f"SEARLE_data/{dataset.lower()}-ViT-L14"
#     searl_features_fp = f"{searl_path}/{source_domain_join}-{target_domain_join}_{suffix}.pkl"
#     if os.path.exists(searl_features_fp):
#         fp = open(searl_features_fp, "rb")
#         searl_data = pickle.load(fp)
#         text_features = searl_data["feats"]
#         img_paths = searl_data["path"]
#     else:
#         print("Extracting SEARLE text_features")
#         from SEARLE.src.validate import generate_oti_predictions
#         text_features, img_paths = generate_oti_predictions(dataset, 'ViT-L/14', target_domain, current_query_paths, all_query_paths, suffix)

#         # dump to path
#         Path(searl_path).mkdir(parents=True, exist_ok=True)
#         fp = open(searl_features_fp, "wb")
#         dict_save = {}
#         dict_save['feats'] = text_features
#         dict_save['path'] = img_paths
#         pickle.dump(dict_save,fp)

#     sim_total = (text_features @ database_features.t())
#     rankings = torch.argsort(sim_total, descending=True)
#     return rankings.detach().cpu()

def searle_rankings(dataset, database_features, target_domain, source_domain, current_query_paths, 
                    all_query_paths=None, searle_mode="oti", suffix=None, topk=1, prompt_format=None):
    assert suffix is not None
    # if multiple domain names
    if len(target_domain.split(' ')) > 1:
        target_domain_join = '_'.join(target_domain.split(' '))
    else:
        target_domain_join = target_domain
    if len(source_domain.split(' ')) > 1:
        source_domain_join = '_'.join(source_domain.split(' '))
    else:
        source_domain_join = source_domain

    if searle_mode == "oti":
        rdir = f"SEARLE_oti_data_{topk}_knn_{prompt_format}_format"
    else: # full
        rdir = f"SEARLE_model_data_{topk}_knn_{prompt_format}_format"

    searl_path = f"{rdir}/{dataset.lower()}-ViT-L14"
    searl_features_fp = f"{searl_path}/{source_domain_join}-{target_domain_join}_{suffix}.pkl"
    if os.path.exists(searl_features_fp):
        fp = open(searl_features_fp, "rb")
        searl_data = pickle.load(fp)
        text_features = searl_data["feats"]
        img_paths = searl_data["path"]
    else:
        print("Extracting SEARLE text_features")
        from SEARLE.src.validate import generate_predictions
        text_features, img_paths = generate_predictions(dataset, 'ViT-L/14', target_domain, current_query_paths, all_query_paths, searle_mode, suffix, topk=topk, prompt_format=prompt_format)

        # dump to path
        Path(searl_path).mkdir(parents=True, exist_ok=True)
        fp = open(searl_features_fp, "wb")
        dict_save = {}
        dict_save['feats'] = text_features
        dict_save['path'] = img_paths
        pickle.dump(dict_save,fp)

    sim_total = (text_features @ database_features.t())
    rankings = torch.argsort(sim_total, descending=True)
    return rankings.detach().cpu()

def invert_levels(input_list):
    if not input_list:
        return []
    d1 = len(input_list)
    d2 = len(input_list[0])
    
    if any(len(sublist) != d2 for sublist in input_list):
        raise ValueError("Sublists do not have consistent lengths")
    transposed = [[input_list[i][j] for i in range(d1)] for j in range(d2)]
    return transposed

def object_or_domain(input_text, model, real_corpus_text, text_corpus_features,
                     object_descript_list=['item', 'building', 'animal', 'mammal', 'fish', 'insect', 'plant'], 
                     domain_descript_list=['domain', 'art', 'craft', 'setting', 'texture', 'material', 'technique']):
    if isinstance(input_text, str):
        input_text = [input_text]

    object_features = text_list_to_features(model, object_descript_list, batch_size=1)
    domain_features = text_list_to_features(model, domain_descript_list, batch_size=1)
    
    input_features = text_list_to_features(model, input_text, batch_size=50)
    sim = np.array(input_features.detach().cpu().numpy().astype("float32")) @ text_corpus_features.T
    ranks = np.argsort(-sim)[0]
    expanded_input = [real_corpus_text[rank] for rank in ranks[:20]]
    expanded_input_features = text_list_to_features(model, expanded_input, batch_size=100)

    object_sim = expanded_input_features @ object_features.t()
    domain_sim = expanded_input_features @ domain_features.t()

    object_sim = object_sim.sum(0)
    domain_sim = domain_sim.sum(0)
    print(object_descript_list)
    print(object_sim)
    print(domain_descript_list)
    print(domain_sim)
    if object_sim.max()>=domain_sim.max():
        return 'object'
    else:
        return 'domain'

def filter_domain(input_text, model, real_corpus_text, text_corpus_features,
                     object_descript_list=['item', 'building', 'animal', 'mammal', 'fish', 'insect', 'plant'], 
                     domain_descript_list=['domain', 'art', 'craft', 'setting', 'texture', 'material']):

    d = text_corpus_features.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatIP(res, d, flat_config)
    index.add(text_corpus_features)
    #_, ranks = index.search(np.array(image_features.cpu().detach().numpy().astype("float32")), miu)
    
    input_text = invert_levels(input_text)

    object_features = text_list_to_features(model, object_descript_list, batch_size=10)
    domain_features = text_list_to_features(model, domain_descript_list, batch_size=10)
    
    is_object_list = []
    for input_list in input_text:
        features = text_list_to_features(model, input_list, batch_size=256)
        _, ranks = index.search(np.array(features.cpu().detach().numpy().astype("float32")), 20)
        expanded_input = [[real_corpus_text[index] for index in row] for row in ranks]
        expanded_input = invert_levels(expanded_input)
        object_sim = []
        domain_sim = []
        for idx in range(len(expanded_input)):
            expanded_feature = text_list_to_features(model, expanded_input[idx], batch_size=256)
            object_sim.append(expanded_feature @ object_features.t())
            domain_sim.append(expanded_feature @ domain_features.t())
        object_sim = torch.stack(object_sim, dim=0).sum(0)
        domain_sim = torch.stack(domain_sim, dim=0).sum(0)
        critical_object_values, _ = torch.max(object_sim, dim=1)
        critical_domain_values, _ = torch.max(domain_sim, dim=1)

        #critical_object_values = torch.mean(object_sim, dim=1)
        #critical_domain_values = torch.mean(domain_sim, dim=1)
        #st()
        is_object_list.append((critical_object_values + 0.5 >= critical_domain_values).cpu().numpy())
    
    input_text = invert_levels(input_text)
    is_object_list = np.array(is_object_list).T
    
    new_data = [[] for _ in range(len(input_text))]
    for idx, instance in enumerate(input_text):
        for idx2, concept in enumerate(instance):
            if is_object_list[idx, idx2]:
                new_data[idx].append(concept)
            elif idx2 == len(instance)-1 and len(new_data[idx]) == 0:
                new_data[idx].append(concept)

    return new_data

def word_to_corpus(input_text, model):
    if isinstance(input_text, str):
        input_text = [input_text]
    real_corpus_text = open_images_corpus_names
    text_corpus_features = open_images_corpus_feats
    input_features = text_list_to_features(model, input_text, batch_size=1)

    sim = np.array(input_features.detach().cpu().numpy().astype("float32")) @ text_corpus_features.T
    ranks = np.argsort(-sim)[0]
    print([real_corpus_text[rank]+' '+str(sim[0][rank]) for rank in ranks[:100]])


def find_inv_freq(image_features, text_corpus_features, real_corpus_text, den):
    d = image_features.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatIP(res, d, flat_config)
    index.add(text_corpus_features)
    _, ranks = index.search(np.array(image_features.cpu().detach().numpy().astype("float32")), 5)
    class_text = [[real_corpus_text[index] for index in row] for row in ranks]
    class_text = [item for sublist in class_text for item in sublist]
    word_counts = {}

    max_value = 0
    for word in class_text:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
        if word_counts[word] > max_value:
            max_value = word_counts[word]

    for key, value in word_counts.items():
        #word_counts[key] = max_value/(value+1)
        #word_counts[key] = max(math.log10(word_counts[key]), 0) # non linear penalties
        #word_counts[key] = (-1 / (1+math.exp(-word_counts[key]+max_value*0.5))) + 1
        if word_counts[key] > max_value*0.6:
            word_counts[key] = 0
        else:
            word_counts[key] = 1
    
    return word_counts

def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / (nres + 1e-5)

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap

def compute_map(correct):
    """
    Computes the mAP for a given set of returned results.

         Usage: 
           map = compute_map (ranks, gnd) 
                 computes mean average precsion (map) only
        
           map, aps, pr, prs = compute_map (ranks, gnd, kappas) 
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
        
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = correct.shape[0]

    for i in np.arange(nq):
        
        # pos = correct[i].nonzero()[0] # a bit scary
        pos = np.where(correct[i] != 0)[0]

        # compute ap
        ap = compute_ap(pos, len(pos))
        
        map = map + ap
    
    map = map / (nq)
    
    return np.around(map*100, decimals=2)