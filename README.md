# FgSty + CPL

This repository contains the code of our foreground-aware stylization (FgSty) and consensus pseudo-labeling (CPL), and the synthesized dataset used in our experiments, ObMan-Ego (see [DATASET.md](./DATASET.md)). If you have some requests or questions, please contact the first author.

## Paper
[Foreground-Aware Stylization and Consensus Pseudo-Labeling for Domain Adaptation of First-Person Hand Segmentation](https://ieeexplore.ieee.org/document/9469781) \
Takehiko Ohkawa, Takuma Yagi, Atsushi Hashimoto, Yoshitaka Ushiku, and Yoichi Sato \
IEEE Access, 2021 \
Project page: https://tkhkaeio.github.io/projects/21_FgSty-CPL/
## Requirements
Python 3.7 \
PyTorch 1.6.0

Data directory structure should be
```
- root / source-dataset (e.g., EGTEA, Ego2Hands, ObMan-Ego)
    - train
    - trainannot (segmentation mask)
    - test
    - testannot (segmentation mask)
- root / target-datasets (e.g., GTEA, EDSH12, EDSH1K, UTG, YHG)
    - train
    - trainannot (segmentation mask)
    - test
    - testannot (segmentation mask)
```

## Stylization
1. Please download pretrained models of [PhotoWCT](https://github.com/NVIDIA/FastPhotoStyle) from [[here]](https://drive.google.com/drive/folders/1a43zm4mLnPUIsA5ZJC6sE6nY-M3383nm?usp=sharing) and set them to `FgSty/pretrained_models`.

2. `cd FgSty` and specify your data root directory in `make_script_rand.py`.

3. Run `python make_script_rand.py` to create files with arguments for stylization.

4. Run `./scripts/EGTEA_v1_test_part00x.sh`

<img src="https://user-images.githubusercontent.com/28190044/124393668-7cff2300-dd36-11eb-9bd8-d7a7fd06616b.jpg">

## Training & Adaptation
1. Please download pretrained models of [RefineNet](https://github.com/DrSleep/refinenet-pytorch) from [[here]](https://drive.google.com/drive/folders/1jd60n_8sXalDrY7N5sz4Vg2qWSBtCwES?usp=sharing) and set them to `CPL/pretrained_models`.

2. `cd CPL` and run
```
python train_refinenet.py --dataset /path/to/your/dataset
```
for the naive training on a single dataset, or run
```
python train_refinenet_CPL.py --dataset /path/to/your/style-adapted-dataset \
                              --src_dataset /path/to/your/source-dataset \
                              --trg_dataset /path/to/your/target-dataset \
                              --src_model_path /path/to/your/pretrained-source-model \
                              --eta 0
```
for the adaptation training based on the consensus scheme without adversarial adaptation. \
Note: the training of CPL requires to use two GPUs.

<img src="https://user-images.githubusercontent.com/28190044/123940334-08de1b80-d9d4-11eb-9665-76e1226c83cd.jpg">

## Evaluation
1. `cd CPL` and specify your data root directory in `test_refinenet.py`.

2. Run
```
python test_refinenet.py --dataset /path/to/your/target-dataset \
                         --model_path /path/to/your/test-model 
```

## References
FastPhotoStyle: https://github.com/NVIDIA/FastPhotoStyle \
RefineNet: https://github.com/DrSleep/refinenet-pytorch \
UMA: https://github.com/cai-mj/UMA