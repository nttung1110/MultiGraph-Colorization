# Deep Graph Matching via Blackbox Differentiation of Combinatorial Solvers 

This repository contains PyTorch implementation of the paper: [Deep Graph Matching via Blackbox Differentiation of Combinatorial Solvers .](https://arxiv.org/abs/2003.11657) 

## Prepare data for training:

You should prepare data folder for the script utils_zed/prepare_train_data to produce standard training data

## Training:

```
# Training on geek.json config
python train_eval_infer_geek.py ./experiments/geek.json
```

## Inference:

You should prepare data folder for the script inference/main.py to perform the inference task

## Important Folder:

+ BB_GM/model_geek.py: implements the model 

+ experiments: stores config for training

+ hades_painting: stores tools supporting training

+ inference: inference module

+ utils: scripts support training of orginal repo

+ utils_zed: scripts support training of my repo

+ eval_infer_geek.py: do the evaluation after training

+ train_eval_infer_geek.py: train and do the evaluation

## Citation

```text
@article{rolinek2020deep,
    title={Deep Graph Matching via Blackbox Differentiation of Combinatorial Solvers},
    author={Michal Rolínek and Paul Swoboda and Dominik Zietlow and Anselm Paulus and Vít Musil and Georg Martius},
    year={2020},
    eprint={2003.11657},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
