# Steganographic Passport 

This repository contains the official implementation of CVPR 2024 paper "[Steganographic Passport: An Owner and User Verifiable Credential for Deep Model IP Protection Without Retraining](https://arxiv.org/pdf/2404.02889)"
Authors: Qi Cui, Ruohan Meng, Chaohui Xu, Chip-Hong Chang


## Introduction

Steganographic Passport uses an invertible steganographic network (ISN) to decouple license-to-use from ownership verification by hiding the user’s identity images into the owner-side passport and recovering them from their respective user-side passports. 
An irreversible and collision-resistant hash function is used to avoid exposing the owner-side passport from the derived user-side passports and increase the uniqueness of the model signature. To safeguard both the passport and model’s weights against advanced ambiguity attacks, an activation-level obfuscation is proposed for the verification branch of the owner’s model. 
By jointly training the verification and deployment branches, their weights become tightly coupled. The proposed method supports agile licensing of deep models by providing a strong ownership proof and license accountability without requiring a separate model retraining for the admission of every new user. 

<div align=center>
<img src='assets/illustration.png' width="60%">
<!-- <img src='assets/act_obfuscation.png' width="31%"> -->
</div>



## Installation

We build our code with `Python=3.8.16, PyTorch=2.0.4, CUDA=12.1`. Please install PyTorch first according to [official instructions](https://pytorch.org/get-started/previous-versions/).

- Clone the repository.

```sh
git clone https://github.com/TracyCuiq/Steganographic-Passport.git
cd Steganographic-Passport
```

- Install dependencies.

```sh
pip install -r requirements.txt
```


## Data

- [CIFAR-10, CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02), and [Caltech-256](https://data.caltech.edu/records/nyy15-4j048) are used in the classification experiments.

- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) is used to train the key-based [ISN](https://openaccess.thecvf.com/content/CVPR2021/papers/Lu_Large-Capacity_Image_Steganography_Based_on_Invertible_Neural_Networks_CVPR_2021_paper.pdf).

- For passport images, we randomly select them from the test set of [COCO dataset](https://cocodataset.org/#home).

- For steganographic key image, we randomly select them from [DTD dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

<!-- The organization of the dataroot is as follows:

```
coco_path/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
``` -->

## Run 

- Train ISN
  ```sh
  python train_inn.py 
  ```


- Generate user-side passports by using the owner-side passport as the cover:


- Use the following script as an example to train a DNN model integrated with the owner-side passport:

  ```sh
  CUDA_VISIBLE_DEVICES=0 python train_MOA.py \
      --epochs 200 \
      --batch-size 64 \
      --trigger-path 'data/trigger_set/pics' \
      --emb-path 'data/trigger_set/emb_pics' \
      --key-type 'image' \
      --train-private \
      --use-trigger-as-passport \
      --exp-id 1 \
      --arch 'resnet' \
      --dataset 'cifar10' \
      --norm-type 'bn'\
      --passport-config 'passport_configs/resnet18_passport_l3.json' \
  ```


## Robustness evaluation

- Ownership ambiguity attacks (ERB)
  ```sh
  CUDA_VISIBLE_DEVICES=$CUDA python run_amb_attack_moa.py \
      --epochs 100 \
      --batch-size 64 \
      --trigger-path 'data/trigger_set/pics' \
      --emb-path 'data/trigger_set/emb_pics' \
      --key-type 'image' \
      --train-private \
      --use-trigger-as-passport \
      --arch $ARCH \
      --dataset $DATASET \
      --norm-type $NORM \
      --passport-config $CONFIG \
      --exp-id $ID \
      --type 'fake2-' \
      --pro-att 'ERB' \
  ```


- License ambiguity attacks
  - Forged user-side passport
    ```sh
    CUDA_VISIBLE_DEVICES=$CUDA python run_eval_inn.py \
       --amb-att  \
       --amb-att-key \
    ```
    

  - Forged steganographic key
    ```sh
    CUDA_VISIBLE_DEVICES=$CUDA python run_eval_inn.py \
       --amb-att  \
    ```


- Removal attacks
  - Fine-tuning
    ```sh
    CUDA_VISIBLE_DEVICES=$CUDA python train_MOA.py \
    -tf \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --trigger-path 'data/trigger_set/pics' \
    --emb-path 'data/trigger_set/emb_pics' \
    --key-type 'image' \
    --train-private \
    --use-trigger-as-passport \
    --arch $ARCH \
    --dataset $DATASET \
    --tl-dataset $TAR_DATASET \
    --norm-type $NORM \
    --passport-config $CONFIG \
    --exp-id $ID \
    ```
    Replace the arguments denoted by `$` with your specific values.
    Ensure you set the correct value for `$ID`.

  - Pruning
    ```sh
      CUDA_VISIBLE_DEVICES=$CUDA python run_prun_moa.py \
    -tf \
    --epochs 200 \
    --batch-size 64 \
    --lr 0.01 \
    --trigger-path 'data/trigger_set/pics' \
    --emb-path 'data/trigger_set/emb_pics' \
    --key-type 'image' \
    --train-private \
    --use-trigger-as-passport \
    --arch $ARCH \
    --dataset $DATASET \
    --norm-type $NORM \
    --passport-config $CONFIG \
    --exp-id $ID \

      ```
      Replace the arguments denoted by $ with your specific values.

## Acknowledgment

- This research is supported by the National Research Foundation, Singapore, and Cyber Security Agency of Singapore under its National Cyber-security Research & Development Programme (Development of Secured Components & Systems in Emerging Technologies through Hardware & Software Evaluation \<NRF-NCR25-DeSNTU-0001>). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the view of National Research Foundation, Singapore and Cyber Security Agency of Singapore.

**Implementations**
- We use the codebases from [DeepIPR](https://github.com/kamwoh/DeepIPR) and [Passport-aware Normalization](https://github.com/ZJZAC/Passport-aware-Normalization) as our baselines. Besides, we build our ISN based on the codebases from [HiNet](https://github.com/TomTomTommi/HiNet). 
We extend our gratitude for their outstanding works.



## Citation

If you use Steganographic Passport in your research or wish to refer to the results published here, please cite our paper with the following BibTeX entry.

```BibTeX
@inproceedings{Steganographic-Passport,
  title={Steganographic Passport: An Owner and User Verifiable Credential for Deep Model IP Protection Without Retraining},
  author={Qi Cui, Ruohan Meng, Chaohui Xu, and Chip-Hong Chang},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024},
}
```
