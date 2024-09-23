# $eFRSA^{2}$

This repository is the source code of Efficient Federated Recommender System with Adaptive Model Pruning and Momentum-based Batch Adjustment ($eFRSA^{2}$). $eFRSA^{2}$ is an improvement of Federated-Neural-Collaborative-Filtering (FedNCF) and the source code is originated from FedNCF's source code.

To run the code, several different commands can be found in test.sh. An example is

```py
python train_federated.py -nc 10 -miuc 1 -mauc 10 -ae 50 -le 10 -bs 128 -pa adaptive -pt quantile -cr 0.1 -ba momentum -r1 0.1 -r2 0.1 -d 0 -ld 32 -sd 0 -sr 2 
```

The original README.md of "Federated-Neural-Collaborative-Filtering" is as follows. The corresponding paper[1] can be found at https://dl.acm.org/doi/abs/10.1016/j.knosys.2022.108441.

[1] Vasileios Perifanis, Pavlos S. Efraimidis. Federated Neural Collaborative Filtering. Knowledge-Based Systems, vol. 242, no. C, 2022.

# Federated-Neural-Collaborative-Filtering

Neural Collaborative Filtering (NCF) is a paper published by National University of Singapore, Columbia University, Shandong University, and Texas A&amp;M University in 2017. It utilizes the flexibility, complexity, and non-linearity of Neural Network to build a recommender system. 

Aim to federated this!

## Demo

![demo](/output.png)

## Setting:

Each client contains a group of users, in the real world this could be considered as connecting from the same WiFi. They learn a local model for recommendation, which is then aggregated centrally.

## Metrics:

1. Hit Ratio: is the fraction of users for which the correct answer is included in the recommendation list of length N, here `N=10`.
2. NDCG: is a metric of ranking quality or the relevance of the top N listed products, here `N=10`.

## Execution:

### Run the Central Single Client Model

Using the command: `python train_single.py`

```py
dataloader = MovielensDatasetLoader()
trainer = NCFTrainer(dataloader.ratings[:50], epochs=20, batch_size=128)
ncf_optimizer = torch.optim.Adam(trainer.ncf.parameters(), lr=5e-4)
_, progress = trainer.train(ncf_optimizer, return_progress=True)
```

### Run the Federated Aggregator Multi-Client Model

Using the command: `python train_federated.py`

```py
dataloader = MovielensDatasetLoader()
fncf = FederatedNCF(dataloader.ratings, num_clients=50, user_per_client_range=[1, 10], mode="ncf", aggregation_epochs=50, local_epochs=10, batch_size=128)
fncf.train()
``` 
