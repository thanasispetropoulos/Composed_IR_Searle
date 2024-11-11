# Adapted from https://github.com/miccunifi/SEARLE/tree/main


### Install Python dependencies

```sh
RCI modules
ml PyTorch/2.0.1-foss-2022a-CUDA-11.7.0
module load diffusers/0.18.2-foss-2022a-CUDA-11.7.0
ml faiss/1.7.3-foss-2022a-CUDA-11.7.0
pip install git+https://github.com/openai/CLIP.git
```

### Data Preparation

Download and extract inside here resulting in the file SEARLE/data
https://drive.google.com/file/d/1n-cJ9WUnMtxEJYIbB1cSE4CY48Ms5JR1/view?usp=sharing

### Datasets

1. Put dataset class in datasets.py

2. Create dataloaders in
   
   a. image_concepts_association.py https://github.com/NikosEfth/Composed_IR/blob/main/SEARLE/src/image_concepts_association.py#L58
   
   b. oti_inversion.py https://github.com/NikosEfth/Composed_IR/blob/main/SEARLE/src/oti_inversion.py#L122 

4. both 2a and 2b expects the images and image_names to be returned as shown below links. If changed, then change here:
   
   a. image_concepts_association.py https://github.com/NikosEfth/Composed_IR/blob/main/SEARLE/src/image_concepts_association.py#L116
   
   b. oti_inversion.py https://github.com/NikosEfth/Composed_IR/blob/main/SEARLE/src/oti_inversion.py#L151
   

### Commands

Set paths from root code directory:
```
export PYTHONPATH=$PYTHONPATH:SEARLE/
export PYTHONPATH=$PYTHONPATH:SEARLE/src/
```

Find similar concepts from corpus
```
python SEARLE/src/image_concepts_association.py --clip-model-name ViT-L/14 --dataset imagenet --split val
```
Perform inversion
```
python SEARLE/src/oti_inversion.py --exp-name imagenet --clip-model-name ViT-L/14 --gpt-exp-name GPTNeo27B --dataset imagenet --split val
```

### Final feature extraction by combining with domain name

1. Pre-extracttion to SEARLE_data/
```
python SEARLE/src/validate.py --exp-name imagenet --eval-type oti --dataset imagenet --clip-model-name ViT-L/14
```

2. Call directly from validation.py - it will check if pre-extracted feats are available from step 1 above.
https://github.com/NikosEfth/Composed_IR/blob/main/validation.py#L682



## Authors

* [**Alberto Baldrati**](https://scholar.google.com/citations?hl=en&user=I1jaZecAAAAJ)**\***
* [**Lorenzo Agnolucci**](https://scholar.google.com/citations?user=hsCt4ZAAAAAJ&hl=en)**\***
* [**Marco Bertini**](https://scholar.google.com/citations?user=SBm9ZpYAAAAJ&hl=en)
* [**Alberto Del Bimbo**](https://scholar.google.com/citations?user=bf2ZrFcAAAAJ&hl=en)

**\*** Equal contribution. Author ordering was determined by coin flip.

## Acknowledgements

This work was partially supported by the European Commission under European Horizon 2020 Programme, grant number
101004545 - ReInHerit.

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />All material is made available under [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicate any changes** that you've made.
