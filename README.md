# $eFRSA^{2}$

This repository is the source code of Efficient Federated Recommender System with Adaptive Model Pruning and Momentum-based Batch Adjustment ($eFRSA^{2}$). $eFRSA^{2}$ is an improvement of Federated-Neural-Collaborative-Filtering (FedNCF) and the source code is originated from FedNCF's source code.

To run the code, several different commands can be found in test.sh. An example is

```py
python train_federated.py -nc 10 -miuc 1 -mauc 10 -ae 50 -le 10 -bs 128 -pa adaptive -pt quantile -cr 0.1 -ba momentum -r1 0.1 -r2 0.1 -d 0 -ld 32 -sd 0 -sr 2 
```

The original README.md of "Federated-Neural-Collaborative-Filtering" can be found at https://github.com/StratosphericD/FedNCF/blob/main/README.md. The corresponding paper[1] can be found at https://dl.acm.org/doi/abs/10.1016/j.knosys.2022.108441.

[1] Vasileios Perifanis, Pavlos S. Efraimidis. Federated Neural Collaborative Filtering. Knowledge-Based Systems, vol. 242, no. C, 2022.
