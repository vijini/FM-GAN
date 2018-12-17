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
MSCOCO dataset and WMT news dataset can be downloaded from link **HERE**:
[Download link](https://www.dropbox.com/sh/ie48jptjxl3m4wk/AACMgLZZpACEMFynP55zA3Xla?dl=0)

## Evaluation:
Use `convert_new.py` to convert the indexed files to sentences.
Then use `selfbleu.py` and `testbleu.py` to evaluate the results.

Note that it is just a rough edition, you have to change the file names manually in the code, we will update this ASAP.

## TODO:
1. Clean the code, make it easy to understand.
2. Don't need to manually change the file names in the code.
3. etc.


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
