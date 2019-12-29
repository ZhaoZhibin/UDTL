
## UDTL-based-Intelligent-Diagnosis-Benchmark

Code release for **[Unsupervised Deep Transfer Learning for Intelligent Fault Diagnosis: An Open Source and Comparative Study](https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark)** by [Zhibin Zhao](https://zhaozhibin.github.io/), Qiyang Zhang, and Xiaolei Yu.

## Guide
This project just provides the baseline (lower bound) accuracies and a unified intelligent fault diagnosis library based on unsupervised deep transfer learning (UDTL) which retains an extended interface for everyone to load their own datasets and models by themselves to carry out new studies.
Meanwhile, all the experiments are executed under Window 10 and Pytorch 1.3 through running on a computer with an Intel Core i7-9700K, GeForce RTX 2080Ti, and 16G RAM.


## Requirements
- Python 3.7
- Numpy 1.16.2
- Pandas 0.24.2
- Pickle
- tqdm 4.31.1
- sklearn 0.21.3
- Scipy 1.2.1
- opencv-python 4.1.0.25
- PyWavelets 1.0.2
- pytorch >= 1.1
- torchvision >= 0.40


## Datasets
- **[CWRU Bearing Dataset](https://csegroups.case.edu/bearingdatacenter/pages/download-data-file/)**
- **[PU Bearing Dataset](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/)**
- **[PHM 2009](https://www.phmsociety.org/competition/PHM/09/apparatus)**
- **[SEU Gearbox Dataset](https://github.com/cathysiyu/Mechanical-datasets)**
- **[JNU Bearing Dataset](http://mad-net.org:8765/explore.html?t=0.5831516555847212.)**

## References

Part of the code refers to the following open source code:
- [CORAL.py](https://github.com/SSARCandy/DeepCORAL) from the paper "[Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://link.springer.com/chapter/10.1007/978-3-319-49409-8_35)" proposed by Sun et al.
- [DAN.py and JAN.py](https://github.com/thuml/Xlearn) from the paper "[Deep Transfer Learning with Joint Adaptation Networks](https://dl.acm.org/citation.cfm?id=3305909)" proposed by Long et al.
- [AdversarialNet.py and entropy_CDA.py](https://github.com/thuml/CDAN) from the paper "[Conditional adversarial domain adaptation](http://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation)" proposed by Long et al.



## Pakages

This repository is organized as:
- [loss](https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark/tree/master/loss) contains different loss functions for Mapping-based DTL.
- [datasets](https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark/tree/master/datasets) contains the data augmentation methods and the Pytorch datasets for time and frequency domains.
- [models](https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark/tree/master/models) contains the models used in this project.
- [utils](https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark/tree/master/utils) contains the functions for realization of the training procedure.


## Usage
- download datasets
- use the train_base.py to test Basis and AdaBN (network-based DTL and instanced-based DTL)

- for example, use the following commands to test Basis for CWRU with the transfer_task 0-->1
- `python train_base.py --data_name CWRU --data_dir D:/Data/CWRU --transfer_task [0],[1] --adabn ""`
- for example, use the following commands to test AdaBN for CWRU with the transfer_task 0-->1
- `python train_base.py --data_name CWRU --data_dir D:/Data/CWRU --transfer_task [0],[1]`

- use the train_advanced.py to test (mapping-based DTL and adversarial-based DTL)
- for example, use the following commands to test DANN for CWRU with the transfer_task 0-->1
- `python train_advanced.py --data_name CWRU --data_dir D:/Data/CWRU --transfer_task [0],[1]  --last_batch "" --distance_metric "" --domain_adversarial True --adversarial_loss DA`
- for example, use the following commands to test MK-MMD for CWRU with the transfer_task 0-->1
- `python train_advanced.py --data_name CWRU --data_dir D:/Data/CWRU --transfer_task [0],[1] --last_batch True --distance_metric True --distance_loss MK-MMD --domain_adversarial "" `



## Citation


## Contact
- zhibinzhao1993@gmail.com
