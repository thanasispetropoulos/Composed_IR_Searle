import pickle
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
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
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
from plotly.offline import plot


def read_dataset_features(pickle_dir):
    with open(pickle_dir, 'rb') as f:
        data = pickle.load(f)
    all_image_features = np.array(data['feats']).astype("float32")
    all_paths = data['path']
    all_classes = data['classes']
    all_domains = data['domains']
    return all_image_features, all_paths, all_classes, all_domains

def save_patch_dataset_features(pickle_dir):
    with open(pickle_dir, 'rb') as f:
        data = pickle.load(f)
    all_cls_features = np.array(data['cls']).astype("float32")
    all_patch_features = np.array(data['patches']).astype("float32")
    all_atention_6 = np.array(data['attn@6']).astype("float32")
    all_atention_24 = np.array(data['attn@24']).astype("float32")
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

root = "/mnt/personal/efthynik/2023_Composed_Image_retrieval"
dataset = 'imagenet_r'
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
    all_cls_features, all_patch_features, all_atention_6, all_atention_24, all_paths, all_classes, all_domains = save_patch_dataset_features(root+'/clip_features/imagenet_r/full_imgnet_patch_features.pkl')
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



def read_laion_40m(text_dir, text_feature_dir, image_featire_dir, size=40000000):
    with open(text_dir, 'rb') as f:
        data = pickle.load(f)
    laion_text = data['actual_text'][:size]
    laion_text_feats = np.load(text_feature_dir, mmap_mode='r')[:size]
    laion_image_feats = np.load(image_featire_dir, mmap_mode='r')[:size]
    laion_text_feats = laion_text_feats / np.linalg.norm(laion_text_feats, axis=-1, keepdims=True)
    laion_image_feats = laion_image_feats / np.linalg.norm(laion_image_feats, axis=-1, keepdims=True)
    return laion_text, laion_text_feats, laion_image_feats




laion_text, laion_text_feats, laion_image_feats = read_laion_40m(text_dir=f'{root}/clip_features/laion_40m/laion_train_actual_text.pkl',
                                                                 text_feature_dir=f'{root}/clip_features/laion_40m/ViT-L-14_openai_train_text_feats.npy',
                                                                 image_featire_dir=f'{root}/clip_features/laion_40m/ViT-L-14_openai_train_image_feats.npy',
                                                                 size=20000)

sim_full_corpus = corpus_feats[:1000].dot(corpus_feats[:1000].T)
sim_full_images = database_image_features[:1000].dot(database_image_features[:1000].T)
sim_full_cross = database_image_features[:1000].dot(corpus_feats[:1000].T)

sim_full_laion_images = laion_image_feats[:1000].dot(laion_image_feats[:1000].T)
sim_full_laion_texts = laion_text_feats[:1000].dot(laion_text_feats[:1000].T)
sim_full_laion_cross = laion_image_feats[:1000].dot(laion_text_feats[:1000].T)

sim_image_cross_dataset = laion_image_feats[:1000].dot(database_image_features[:1000].T)
sim_corpus_cross_dataset = laion_text_feats[:1000].dot(corpus_feats[:1000].T)

sim_full_cross = sim_full_cross.reshape(1, sim_full_cross.shape[0]*sim_full_cross.shape[1]).squeeze(0)
sim_full_images = sim_full_images.reshape(1, sim_full_images.shape[0]*sim_full_images.shape[1]).squeeze(0)
sim_full_corpus = sim_full_corpus.reshape(1, sim_full_corpus.shape[0]*sim_full_corpus.shape[1]).squeeze(0)

sim_full_laion_cross = sim_full_laion_cross.reshape(1, sim_full_laion_cross.shape[0]*sim_full_laion_cross.shape[1]).squeeze(0)
sim_full_laion_images = sim_full_laion_images.reshape(1, sim_full_laion_images.shape[0]*sim_full_laion_images.shape[1]).squeeze(0)
sim_full_laion_texts = sim_full_laion_texts.reshape(1, sim_full_laion_texts.shape[0]*sim_full_laion_texts.shape[1]).squeeze(0)

sim_image_cross_dataset = sim_image_cross_dataset.reshape(1, sim_image_cross_dataset.shape[0]*sim_image_cross_dataset.shape[1]).squeeze(0)
sim_corpus_cross_dataset = sim_corpus_cross_dataset.reshape(1, sim_corpus_cross_dataset.shape[0]*sim_corpus_cross_dataset.shape[1]).squeeze(0)

sim_full_cross = np.random.choice(sim_full_cross, 10000, replace=False)
sim_full_images = np.random.choice(sim_full_images, 10000, replace=False)
sim_full_corpus = np.random.choice(sim_full_corpus, 10000, replace=False)

sim_full_laion_cross = np.random.choice(sim_full_laion_cross, 10000, replace=False)
sim_full_laion_images = np.random.choice(sim_full_laion_images, 10000, replace=False)
sim_full_laion_texts = np.random.choice(sim_full_laion_texts, 10000, replace=False)

sim_image_cross_dataset = np.random.choice(sim_image_cross_dataset, 10000, replace=False)
sim_corpus_cross_dataset = np.random.choice(sim_corpus_cross_dataset, 10000, replace=False)

print("hists")

#fig = plt.figure(figsize =(10, 7))
fig, axs = plt.subplots(8, 1, figsize=(15, 30))
axs[0].hist(sim_full_images, bins = np.arange(-1.0, 1.0, 0.01).tolist())
axs[0].set_title("Imagenet-r Sim")
axs[1].hist(sim_full_corpus, bins = np.arange(-1.0, 1.0, 0.01).tolist())
axs[1].set_title("Corpus 20k Sim")
axs[2].hist(sim_full_cross, bins = np.arange(-1.0, 1.0, 0.01).tolist())
axs[2].set_title("Imagenet-r vs Corpus 20k Sim")

axs[3].hist(sim_full_laion_images, bins = np.arange(-1.0, 1.0, 0.01).tolist())
axs[3].set_title("Laion Image Sim")
axs[4].hist(sim_full_laion_texts, bins = np.arange(-1.0, 1.0, 0.01).tolist())
axs[4].set_title("Laion Corpus Sim")
axs[5].hist(sim_full_laion_cross, bins = np.arange(-1.0, 1.0, 0.01).tolist())
axs[5].set_title("Laion Image vs Laion Corpus Sim")

axs[6].hist(sim_image_cross_dataset, bins = np.arange(-1.0, 1.0, 0.01).tolist())
axs[6].set_title("Laion Image vs Imagenet-r Image Sim")
axs[7].hist(sim_corpus_cross_dataset, bins = np.arange(-1.0, 1.0, 0.01).tolist())
axs[7].set_title("Laion Corpus vs Corpus 20k Sim")
plt.savefig('0_fig_clip_similarities.png')
plt.clf()
n=10
all_patch_features_n = all_patch_features[30:(30+n), :, :].reshape(all_patch_features.shape[1]*n, 768)

all_atention_6_n = all_atention_6[30:(30+n), :]
sorted_indices = np.argsort(all_atention_6_n, axis=1)
boolean_array = np.zeros_like(all_atention_6_n, dtype=bool)
boolean_array[np.arange(all_atention_6_n.shape[0])[:, np.newaxis], sorted_indices[:, :50]] = 1
boolean_array = boolean_array*1
all_atention_6_n = all_atention_6_n.reshape(all_atention_6.shape[1]*n)
boolean_array = boolean_array.reshape(boolean_array.shape[1]*n)
#all_atention_6_n = (all_atention_6_n > 0.005)*1

all_atention_24_n = all_atention_24[30:(30+n), :].reshape(all_atention_24.shape[1]*n)
#all_atention_24_n = (all_atention_24_n > 0.005)*1

feats = np.concatenate((corpus_feats, database_image_features, laion_text_feats, laion_image_feats, all_patch_features_n), axis=0)
#scaler = StandardScaler()
#data_standardized = scaler.fit_transform(feats)
pca = PCA(n_components=768)
data_reduced = pca.fit_transform(feats)

split_point_1 = len(corpus_feats)
split_point_2 = split_point_1 + len(database_image_features)
split_point_3 = split_point_2 + len(laion_text_feats)
split_point_4 = split_point_3 + len(laion_image_feats)
corpus_pca = data_reduced[:split_point_1]
images_pca = data_reduced[split_point_1:split_point_2]
laion_text_pca = data_reduced[split_point_2:split_point_3]
laion_image_pca = data_reduced[split_point_3:split_point_4]
patch_features_n_pca =  data_reduced[split_point_4:]
#patch_features_n_pca = patch_features_n_pca*all_atention_6_n
explained_variance_ratio = pca.explained_variance_ / np.sum(pca.explained_variance_)
indices = np.arange(len(explained_variance_ratio))
plt.figure(figsize =(10, 7))
plt.bar(indices, explained_variance_ratio)
plt.xlabel('Index')
plt.ylabel('% of variance per dimension')
plt.title('PCA information per dimension')
plt.savefig('0_fig_PCA_variance_hist.png')
plt.clf()

plt.figure(figsize=(15, 15))
plt.scatter(images_pca[:, 0], images_pca[:, 1], c='red', label='Images', alpha=0.4, s=1)
plt.scatter(corpus_pca[:, 0], corpus_pca[:, 1], c='blue', label='Corpus', alpha=0.4, s=1)
plt.scatter(laion_text_pca[:, 0], laion_text_pca[:, 1], c='green', label='Laion text', alpha=0.4, s=1)
plt.scatter(laion_image_pca[:, 0], laion_image_pca[:, 1], c='orange', label='Laion images', alpha=0.4, s=1)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.title('PCA CLIP features')
plt.savefig('0_fig_clip_pca_2d.png')
plt.clf()

fig = plt.figure(figsize=(30, 30))

# Define different perspectives with elevation and azimuthal angles
perspectives = [(30, 30), (30, 60), (30, 90),
               (60, 30), (60, 60), (60, 90),
               (90, 30), (90, 60), (90, 90)]

for i, (elev, azim) in enumerate(perspectives):
    ax = fig.add_subplot(3, 3, i + 1, projection='3d')
    ax.scatter(images_pca[:, 0], images_pca[:, 1], images_pca[:, 2], c='red', marker='o', label='Images', alpha=0.4, s=1)
    ax.scatter(corpus_pca[:, 0], corpus_pca[:, 1], corpus_pca[:, 2], c='blue', marker='o', label='Corpus', alpha=0.4, s=1)
    ax.scatter(laion_text_pca[:, 0], laion_text_pca[:, 1], laion_text_pca[:, 2], c='green', marker='o', label='Laion text', alpha=0.4, s=1)
    ax.scatter(laion_image_pca[:, 0], laion_image_pca[:, 1], laion_image_pca[:, 2], c='orange', marker='o', label='Laion images', alpha=0.4, s=1)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(f'Perspective: ({elev}°, {azim}°)')
    ax.view_init(elev=elev, azim=azim)

# Add a common legend
fig.legend(loc='upper right')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig('0_fig_clip_pca_3d.png')
plt.clf()

n_cls_scatters = []
n_patch_scatters = []
colors = ['goldenrod', 'orangered', 'skyblue', 'darkblue', 'greenyellow', 'purple', 'limegreen', 'maroon', 'olive', 'lightsalmon']
for idx in range(n):
    n_cls_scatters.append(go.Scatter3d(
        x=[images_pca[idx, 0]],
        y=[images_pca[idx, 1]],
        z=[images_pca[idx, 2]],
        mode='markers',
        marker=dict(size=5, color=colors[idx]),
        text=database_classes[30+idx] + ' ' + database_domains[30+idx] + ' ' + database_paths[30+idx],
        name=str(idx)+' cls token')
        )
    n_patch_scatters.append(go.Scatter3d(
        x=patch_features_n_pca[idx*256:256*(idx+1), 0]*boolean_array[idx*256:256*(idx+1)],
        y=patch_features_n_pca[idx*256:256*(idx+1), 1]*boolean_array[idx*256:256*(idx+1)],
        z=patch_features_n_pca[idx*256:256*(idx+1), 2]*boolean_array[idx*256:256*(idx+1)],
        mode='markers',
        marker=dict(size=3, color=colors[idx]),
        text=['token no ' + str(idx2) + ' act@6 ' + str(round(all_atention_6_n[30+idx], 4)) +' act@24 ' + str(round(all_atention_24_n[30+idx], 4)) + ' ' + database_classes[30+idx] + ' ' + database_domains[30+idx] + ' ' + database_paths[30+idx] for idx2 in range(256)],
        name=str(idx)+' patch tokens')
        )

scatter1 = go.Scatter3d(
    x=images_pca[:, 0],
    y=images_pca[:, 1],
    z=images_pca[:, 2],
    mode='markers',
    marker=dict(size=1, color='red'),
    text=[database_classes[idx] + ' ' + database_domains[idx] + ' ' + database_paths[idx] for idx in range(len(database_paths))],
    name='Images'
)


# Create a 3D scatter plot for array2 in blue
scatter2 = go.Scatter3d(
    x=corpus_pca[:, 0],
    y=corpus_pca[:, 1],
    z=corpus_pca[:, 2],
    mode='markers',
    marker=dict(size=1, color='blue'),
    text=corpus_names,
    name='Corpus'
)

scatter3 = go.Scatter3d(
    x=laion_image_pca[:, 0],
    y=laion_image_pca[:, 1],
    z=laion_image_pca[:, 2],
    mode='markers',
    marker=dict(size=1, color='orange'),
    text=laion_text,
    name='Laion Images'
)

# Create a 3D scatter plot for array2 in blue
scatter4 = go.Scatter3d(
    x=laion_text_pca[:, 0],
    y=laion_text_pca[:, 1],
    z=laion_text_pca[:, 2],
    mode='markers',
    marker=dict(size=1, color='green'),
    text=laion_text,
    name='Laion Corpus'
)
# Create the layout
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='X-axis'),
        yaxis=dict(title='Y-axis'),
        zaxis=dict(title='Z-axis')
    ),
    title='3D PCA CLIP'
)

# Create the figure
fig = go.Figure(data=[scatter1, scatter2, scatter3, scatter4] + n_cls_scatters + n_patch_scatters, layout=layout)

# Save the plot to an HTML file
plot(fig, filename='0_fig_3d_scatterplot.html')
st()
