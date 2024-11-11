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
from pyvis.network import Network
import networkx as nx
import random

def read_dataset_features(pickle_dir):
    with open(pickle_dir, 'rb') as f:
        data = pickle.load(f)
    all_image_features = np.array(data['feats']).astype("float32")
    all_paths = data['path']
    all_classes = data['classes']
    all_domains = data['domains']
    return all_image_features, all_paths, all_classes, all_domains

def read_corpus_features(pickle_dir):
    with open(pickle_dir, "rb") as data:
        data_dict = pickle.load(data)
        #descr = torch.from_numpy(data_dict["feats"].astype("float32")).float().to('cuda')
        descr = np.array(data_dict["feats"]).astype("float32")
        names = data_dict["prompts"]
    return descr, names

root = "/mnt/personal/efthynik/2023_Composed_Image_retrieval"
dataset = 'lueven'
corpus = '20k'
mode = 'normal'

if corpus == '1k':
    corpus_feats, corpus_names = read_corpus_features(f'{root}/clip_features/corpus/imagenet_names.pkl')
elif corpus == '20k':
    corpus_feats, corpus_names = read_corpus_features(f'{root}/clip_features/corpus/open_image_v7_class_names.pkl')
elif corpus == '236k':
    corpus_feats, corpus_names = read_corpus_features(f'{root}/clip_features/corpus/nltk_words.pkl')

if dataset == 'imagenet_r':
    query_image_features, query_paths, query_classes, query_domains = read_dataset_features(root+'/clip_features/imagenet_r/full_imgnet_features2.pkl')
    database_image_features, database_paths, database_classes, database_domains = read_dataset_features(root+'/clip_features/imagenet_r/full_imgnet_features2.pkl')
    domains = ['real', 'cartoon', 'origami', 'toy', 'sculpture']
    domain_change = {'real': 'real', 'cartoon': 'cartoon', 'origami': 'origami', 'toy': 'toy', 'sculpture': 'sculpture'}
elif dataset == 'nico':
    query_image_features, query_paths, query_classes, query_domains = read_dataset_features(root+'/clip_features/nico/query_nico_features.pkl')
    database_image_features, database_paths, database_classes, database_domains = read_dataset_features(root+'/clip_features/nico/database_nico_features.pkl')
    domains = ['autumn', 'dim', 'grass', 'outdoor', 'rock', 'water']
    domain_change = {'autumn': 'autumn', 'dim': 'dimlight', 'grass': 'grass', 'outdoor': 'outdoor', 'rock': 'rock', 'water': 'water'}
elif dataset == 'minidn':
    query_image_features, query_paths, query_classes, query_domains = read_dataset_features(root+'/clip_features/minidn/query_minidn_features.pkl')
    database_image_features, database_paths, database_classes, database_domains = read_dataset_features(root+'/clip_features/minidn/database_minidn_features.pkl')
    domains = ['clipart', 'painting', 'real', 'sketch']
    domain_change = {'clipart': 'clipart', 'painting': 'painting', 'real': 'real', 'sketch': 'sketch'}
elif dataset == 'lueven':
    query_image_features, query_paths, query_classes, query_domains = read_dataset_features(root+'/clip_features/lueven/full_lueven_features.pkl')
    database_image_features, database_paths, database_classes, database_domains = read_dataset_features(root+'/clip_features/lueven/full_lueven_features.pkl')
    domains = ['New', 'Old']
    domain_change = {'New': 'today', 'Old': 'gravure'}
elif dataset == 'tokyo247':
    query_image_features, query_paths, query_classes, query_domains = read_dataset_features(root+'/clip_features/tokyo247/full_tokyo247_features.pkl')
    database_image_features, database_paths, database_classes, database_domains = read_dataset_features(root+'/clip_features/tokyo247/full_tokyo247_features.pkl')
    domains = ['Day', 'Night']
    domain_change = {'Day': 'day', 'Night': 'night'}

def set_index(database):
    d = database.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatIP(res, d, flat_config)
    index.add(database)
    return index

def invert_levels(input_list):
    if not input_list:
        return []
    d1 = len(input_list)
    d2 = len(input_list[0])
    
    if any(len(sublist) != d2 for sublist in input_list):
        raise ValueError("Sublists do not have consistent lengths")
    transposed = [[input_list[i][j] for i in range(d1)] for j in range(d2)]
    return transposed

def ids_ranks_sim(total_name_ids, image_absolute_ids, im2corp_ranks, im2corp_sim):
    im2corp_list_ids = []
    im2corp_list_sim = []
    for idx in range(im2corp_ranks.shape[1]):
        im2corp_list_ids.append(np.concatenate((image_absolute_ids, im2corp_ranks[:, idx:idx+1]), 1))
        im2corp_list_sim.append(im2corp_sim[:, idx:idx+1])
    im2corp_list_ids = np.concatenate(im2corp_list_ids)
    im2corp_list_ids = invert_levels([[total_name_ids[row_index] for row_index in row] for row in im2corp_list_ids])
    im2corp_list_sim = invert_levels(np.concatenate(im2corp_list_sim).tolist())
    return im2corp_list_ids, im2corp_list_sim

def choose_color_group_shape(node_1, node_2, is_node=False):
    color_25 = ["#FFEFDB", "#76EEC6", "#8A2BE2", "#EE3B3B", "#8A3324", "#8EE5EE", "#FF9912", "#66CD00", "#EEE8CD", "#EEAD0E", "#8B6508", 
                "#6959CD", "B9D3EE", " #00EE76", "#FF3E96", "#8B2252", "#EED8AE", "#B7B7B7", "#515151", "#282828", "#00FFFF", "#FF1493", 
                "#8B814C", "#7CFC00", "#8B8378"]
    class_lueven = ['Q001', 'Q221', 'GalleFaceHotel', 'BigBen', 'Q138', 'EiffelTower', 'Arcdetriomphe', 'SacreCoeur', 'Q_999', 'TajMahal', 
                    'Notredame', 'Pettah', 'Q140', 'Q302', 'GareDeLyonParis', 'Gaffor', 'Q003', 'TempleTooth', 'Q141', 'Q005', 'Q198', 'Q024', 
                    'Q016', 'Q147', 'Q_000']
    shape = 'dot'
    if node_1[:6] == 'corpus' and node_2[:6] == 'corpus':
        color = '#0F52BA'
        group = 'corpus'
    elif node_1[:5] == 'image' and node_2[:5] == 'image':
        color = '#CA3433'
        group = 'image'
        if dataset == 'lueven' and is_node:
            if 'Old' in node_1:
                shape = 'star'
            for idx, label in enumerate(class_lueven):
                if label in node_1:
                    color = color_25[idx]
    else:
        color = '#3b8132'
        group = 'other'
    return color, group, shape

def choose_color2(node_1):
    if node_1[:6] == 'Domain':
        color = '#3b8132'
    elif node_1[:6] == 'Object':
        color = '#CA3433'
    else:
        color = '#0F52BA'
    return color

def nn_graph(corpus_names, paths, classes, domains, corpus_set, image_set, k, first_run=True):
    total_set = np.concatenate((corpus_set, image_set))
    total_ids = np.expand_dims(np.arange(len(total_set)), axis=1)
    corpus_absolute_ids = np.expand_dims(np.arange(len(corpus_set)), axis=1)
    image_absolute_ids = np.expand_dims(np.arange(len(image_set)) + len(corpus_set), axis=1)
    corpus_name_ids = ['corpus ' + x for x in corpus_names]
    image_name_ids = ['image '+ classes[idx] + ' ' + domains[idx] + ' ' + paths[idx] for idx in range(len(classes))]
    total_name_ids = corpus_name_ids + image_name_ids

    index = set_index(corpus_set)
    im2corp_sim, im2corp_ranks = index.search(image_set, k)
    corp2corp_sim, corp2corp_ranks = index.search(corpus_set, k+1)
    corp2corp_ranks = corp2corp_ranks[:,1:]
    corp2corp_sim = corp2corp_sim[:,1:]
    im2corp_vertices, im2corp_weights = ids_ranks_sim(total_name_ids, image_absolute_ids, im2corp_ranks, im2corp_sim)
    corp2corp_vertices, corp2corp_weights = ids_ranks_sim(total_name_ids, corpus_absolute_ids, corp2corp_ranks, corp2corp_sim)

    index = set_index(image_set)
    im2im_sim, im2im_ranks = index.search(image_set, k+1)
    corp2im_sim, corp2im_ranks = index.search(corpus_set, k)
    im2im_ranks = im2im_ranks[:,1:]
    im2im_sim = im2im_sim[:,1:]
    im2im_ranks += len(corpus_set)
    corp2im_ranks += len(corpus_set)
    im2im_vertices, im2im_weights = ids_ranks_sim(total_name_ids, image_absolute_ids, im2im_ranks, im2im_sim)
    corp2im_vertices, corp2im_weights = ids_ranks_sim(total_name_ids, corpus_absolute_ids, corp2im_ranks, corp2im_sim)

    index = set_index(total_set)
    all2all_sim, all2all_ranks = index.search(total_set, k+1)
    all2all_ranks = all2all_ranks[:,1:]
    all2all_sim = all2all_sim[:,1:]
    all2all_vertices, all2all_weights = ids_ranks_sim(total_name_ids, total_ids, all2all_ranks, all2all_sim)

    if first_run:
        corpus_names = list(set(im2corp_vertices[1]))
        corpus_names = [x.replace('corpus ', '') for x in corpus_names]
        model, preprocess_val = load('ViT-L/14', jit=False)
        corpus_set = np.array(text_list_to_features(model, corpus_names, batch_size=10).cpu().detach().numpy().astype("float32"))
        im2corp_vertices, im2corp_weights, corp2corp_vertices, corp2corp_weights, im2im_vertices, im2im_weights, corp2im_vertices, corp2im_weights, all2all_vertices, all2all_weights = nn_graph(corpus_names, paths, classes, domains, corpus_set, image_set, k, False)
    return im2corp_vertices, im2corp_weights, corp2corp_vertices, corp2corp_weights, im2im_vertices, im2im_weights, corp2im_vertices, corp2im_weights, all2all_vertices, all2all_weights

def nn_graph_text_classification(corpus_names, corpus_set, 
                                 object_descript_list=['item', 'building', 'animal', 'mammal', 'fish', 'insect', 'plant', 'car', 'dog', 'food', 'clothing'], 
                                 domain_descript_list=['domain', 'art', 'craft', 'setting', 'texture', 'material', 'technique']):
    model, preprocess_val = load('ViT-L/14', jit=False)

    object_features = np.array(text_list_to_features(model, object_descript_list, batch_size=10).cpu().detach().numpy().astype("float32"))
    domain_features = np.array(text_list_to_features(model, domain_descript_list, batch_size=10).cpu().detach().numpy().astype("float32"))
    total_features = np.concatenate((object_features, domain_features))
    object_descript_list = ['Object ' + x for x in object_descript_list]
    domain_descript_list = ['Domain ' + x for x in domain_descript_list]
    corpus_names = ['corpus ' + x for x in corpus_names]
    total_descript_list = object_descript_list + domain_descript_list
    index = set_index(total_features)
    sim, ranks = index.search(corpus_set, 1)
    classification = invert_levels([[total_descript_list[row_index] for row_index in row] for row in ranks])
    weights = invert_levels(sim.tolist())[0]
    return classification, corpus_names, total_descript_list, weights

net = Network(height="1000px", width="75%", select_menu=True, filter_menu=True)
net.force_atlas_2based()
net.toggle_physics(False)
net.show_buttons(filter_=["physics", "edges"])
net.set_edge_smooth("dynamic")

if mode == 'text_classification':
    classification, corpus_names, total_descript_list, weights = nn_graph_text_classification(corpus_names, corpus_feats)
    for idx in range(len(weights)):
        if random.randint(1,100)>80:
            print(idx, end='\r')
            color = choose_color2(classification[0][idx])
            net.add_node(classification[0][idx], classification[0][idx], color=color, title=classification[0][idx])
            color = choose_color2(corpus_names[idx])
            net.add_node(corpus_names[idx], corpus_names[idx], color=color, title=corpus_names[idx])
            color = choose_color2(classification[0][idx])
            net.add_edge(classification[0][idx], corpus_names[idx], color=color, value=weights[idx])
    net.show('domain_classification.html', notebook=False)
else:
    im2corp_vertices, im2corp_weights, corp2corp_vertices, corp2corp_weights, im2im_vertices, im2im_weights, corp2im_vertices, corp2im_weights, all2all_vertices, all2all_weights = nn_graph(corpus_names, query_paths, query_classes, query_domains, corpus_feats, query_image_features, 2)
    for idx in range(len(all2all_weights[0])):
        print(idx, end='\r')

        color, group, shape = choose_color_group_shape(all2all_vertices[0][idx], all2all_vertices[0][idx], is_node=True)
        net.add_node(all2all_vertices[0][idx], all2all_vertices[0][idx], color=color, shape=shape, title=all2all_vertices[0][idx])
        color, group, shape = choose_color_group_shape(all2all_vertices[1][idx], all2all_vertices[1][idx], is_node=True)
        net.add_node(all2all_vertices[1][idx], all2all_vertices[1][idx], color=color, shape=shape, title=all2all_vertices[1][idx])
        color, group, shape = choose_color_group_shape(all2all_vertices[0][idx], all2all_vertices[1][idx])
        net.add_edge(all2all_vertices[0][idx], all2all_vertices[1][idx], color=color, value=all2all_weights[0][idx])

    for idx in range(len(im2corp_weights[0])):
        print(idx, end='\r')
        color, group, shape = choose_color_group_shape(im2corp_vertices[0][idx], im2corp_vertices[1][idx])
        net.add_edge(im2corp_vertices[0][idx], im2corp_vertices[1][idx], color=color, value=im2corp_weights[0][idx])

net.show('test_small_lueven.html', notebook=False)

st()
