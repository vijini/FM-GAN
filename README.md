# FM-GAN

This repository is the implementation for paper 
[Adversarial Text Generation via
Feature-Mover’s Distance](https://arxiv.org/pdf/1809.06297.pdf)

## Requirement:
Python 2.7, Tensorflow 1.8.0

## Run 
* Run: `python autoencoder.py` for MLE pre-train
* Run: `python text_GAN.py` for adversarial training
* Options: options can be made by changing `option` class. 

## Note:
`model.py`: sinkhorn divergence edition

`model2.py`: IPOT edition

## Dataset:
Uploading...

## Citation 
Please cite our paper if it helps with your research
```latex
@inproceedings{chen2018adversarial,
  title={Adversarial Text Generation via Feature-Mover's Distance},
  author={Chen, Liqun and Dai, Shuyang and Tao, Chenyang and Shen, Dinghan and Gan, Zhe and Zhang, Haichao and Zhang, Yizhe and Carin, Lawrence},
  Booktitle={NIPS},
  year={2018}
}
```
For any question or suggestions, feel free to contact [my email](liqun.chen@duke.edu).