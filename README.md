## The Role of Embedding Complexity in Domain-invariant Representations
This repository is the official PyTorch implementation of the algorithm Multilayer Divergence Minimization (MDM) proposed in the following paper:   

Ching-Yao Chuang, Antonio Torralba, Stefanie Jegelka. The Role of Embedding Complecity in Domain-invariant Representations [link](https://arxiv.org/pdf/1910.05804.pdf)

#### Environment
- Pytorch 1.0
- Python 2.7

#### Dataset

Download the MNIST-M dataset from [Google Drive](https://drive.google.com/open?id=1iij6oj3akjJtaVe9eV-6UnRPJSO4GpdH) and unzip it. 
```
cd dataset
tar -zvxf mnist_m.tar.gz
```

#### Training
Simply run

```
python main.py
```
