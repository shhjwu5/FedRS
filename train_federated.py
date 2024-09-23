import torch
from train_single import NCFTrainer
from dataloader import MovielensDatasetLoader
import random
from tqdm import tqdm
from server_model import ServerNeuralCollaborativeFiltering,ServerGMF,ServerMLP
import copy
import argparse
import os 

import numpy as np

class Utils:
	def __init__(self, num_clients, local_path="./models/local_items/", server_path="./models/central/"):
		self.epoch = 0
		self.num_clients = num_clients
		self.local_path = local_path
		self.server_path = server_path

	def load_pytorch_client_model(self, path):
		return torch.jit.load(path)

	def get_user_models(self, loader):
		models = []
		for client_id in range(self.num_clients):
			models.append({'model':loader(self.local_path+"dp"+str(client_id)+".pt")})
		return models

	def get_previous_federated_model(self):
		self.epoch += 1
		return torch.jit.load(self.server_path+"server"+str(self.epoch-1)+".pt")

	def save_federated_model(self, model):
		torch.jit.save(model, self.server_path+"server"+str(self.epoch)+".pt")

def federate(utils):
    client_models = utils.get_user_models(utils.load_pytorch_client_model)
    server_model = utils.get_previous_federated_model()
    if len(client_models) == 0:
        utils.save_federated_model(server_model)
        return
    n = len(client_models)
    server_new_dict = copy.deepcopy(client_models[0]['model'].state_dict())
    for i in range(1, len(client_models)):
        client_dict = client_models[i]['model'].state_dict()
        for k in client_dict.keys():
            server_new_dict[k] += client_dict[k] 
    for k in server_new_dict.keys():
        server_new_dict[k] = server_new_dict[k] / n
    server_model.load_state_dict(server_new_dict)
    utils.save_federated_model(server_model)

class FederatedNCF:
	def __init__(self, ui_matrix, args):
		random.seed(args.seed)
		self.save_name = ""
		for value in vars(args).values():
			self.save_name+=str(value)+"-"
		self.save_name = self.save_name[:-1]
		self.base_path = "./models/%s/"%(self.save_name)
		if not os.path.exists(self.base_path):
			os.mkdir(self.base_path)
			for subfolder in ["central","local","local_items"]:
				os.mkdir(self.base_path+subfolder)
  
		self.ui_matrix = ui_matrix
		self.device = torch.device("cuda:%d"%(args.device) if torch.cuda.is_available() else "cpu")
		self.model_name =  args.model_name
		self.num_clients = args.num_clients
		self.latent_dim = args.latent_dim
		self.user_per_client_range = [args.min_user_per_client,args.max_user_per_client]
		self.aggregation_epochs = args.aggregation_epochs
		self.local_epochs = args.local_epochs
		self.batch_size = args.batch_size
		self.settings = [copy.deepcopy(args) for _ in range(self.num_clients)]
		self.clients = self.generate_clients()
		self.ncf_optimizers = [torch.optim.Adam(client.ncf.parameters(), lr=5e-4) for client in self.clients]
		self.utils = Utils(self.num_clients,local_path=self.base_path+"local_items/",server_path=self.base_path+"central/")
  
		self.prev_batch_adjust = [0 for _ in range(self.num_clients)]

	def generate_clients(self):
		start_index = 0
		clients = []
		for i in range(self.num_clients):
			users = random.randint(self.user_per_client_range[0], self.user_per_client_range[1])
			clients.append(NCFTrainer(i,self.ui_matrix[start_index:start_index+users],self.settings[i]))
			start_index += users
		return clients

	def single_round(self, epoch=0, first_time=False):
		single_round_results = {key:[] for key in ["num_users", "loss", "hit_ratio@10", "ndcg@10", "compress@10", "time@10"]}
		bar = tqdm(enumerate(self.clients), total=self.num_clients)
		for client_id, client in bar:
			results = client.train(self.ncf_optimizers[client_id],self.settings[client_id])
			for k,i in results.items():
				single_round_results[k].append(i)
			printing_single_round = {"epoch": epoch}
			printing_single_round.update({k:round(sum(i)/len(i), 4) for k,i in single_round_results.items()})
			model = torch.jit.script(client.ncf.to(torch.device("cpu")))
			torch.jit.save(model, self.base_path+"local/dp"+str(client_id)+".pt")
			bar.set_description(str(printing_single_round))
		bar.close()

		return single_round_results

	def extract_item_models(self):
		for client_id in range(self.num_clients):
			model = torch.jit.load(self.base_path+"local/dp"+str(client_id)+".pt")
			if self.model_name=="NCF":
				item_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
			elif self.model_name=="GMF":
				item_model = ServerGMF(item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
			elif self.model_name=="MLP":
				item_model = ServerMLP(item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
			item_model.set_weights(model)
			item_model = torch.jit.script(item_model.to(torch.device("cpu")))
			torch.jit.save(item_model, self.base_path+"local_items/dp"+str(client_id)+".pt")

	def train(self):
		first_time = True
		if self.model_name=="NCF":
			server_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
		elif self.model_name=="GMF":
			server_model = ServerGMF(item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
		elif self.model_name=="MLP":
			server_model = ServerMLP(item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
		server_model = torch.jit.script(server_model.to(torch.device("cpu")))
		torch.jit.save(server_model, self.base_path+"central/server"+str(0)+".pt")
		for epoch in range(self.aggregation_epochs):
			server_model = torch.jit.load(self.base_path+"central/server"+str(epoch)+".pt", map_location=self.device)
			_ = [client.ncf.to(self.device) for client in self.clients]
			_ = [client.ncf.load_server_weights(server_model) for client in self.clients]
			##################################
			single_round_metrics = self.single_round(epoch=epoch, first_time=first_time)
			print("Epoch: "+str(epoch))
			print("Avg # of users: "+str(np.mean(single_round_metrics["num_users"])))
			print("Avg Loss: "+str(np.mean(single_round_metrics["loss"])))
			print("Avg hit ratio: "+str(np.mean(single_round_metrics["hit_ratio@10"])))
			print("Avg ndcg: "+str(np.mean(single_round_metrics["ndcg@10"])))
			print("Avg compression ratio: "+str(np.mean(single_round_metrics["compress@10"])))
			print("Avg train time: "+str(np.mean(single_round_metrics["time@10"])))
			print("Std train time: "+str(np.std(single_round_metrics["time@10"])))
			##################################
			first_time = False
			self.extract_item_models()
			federate(self.utils)
   
			##################################
			self.adjust_setting(single_round_metrics,epoch)
			##################################

	##################################
	def adjust_setting(self,single_round_metrics,epoch):
		if epoch==0:
			self.avg_time = np.mean(single_round_metrics["time@10"])
		diff_batch_num = [0 for _ in range(self.num_clients)]
		new_batches = [0 for _ in range(self.num_clients)]
		for i in range(self.num_clients):
			if self.settings[i].batch_adjustment == "momentum":
				diff_batch_num[i] = (single_round_metrics["time@10"][i] - self.avg_time)/self.avg_time*10
				new_batches[i] = self.settings[i].local_epochs + self.settings[i].momentum*self.prev_batch_adjust[i] - self.settings[i].adjust_ratio*diff_batch_num[i]
				self.settings[i].local_epochs = max(1,round(new_batches[i]))
			# print(single_round_metrics["time@10"][i],self.avg_time,diff_batch_num[i],new_batches[i],self.settings[i].local_epochs)
			elif self.settings[i].batch_adjustment == "SGD":
				diff_batch_num[i] = (single_round_metrics["time@10"][i] - self.avg_time)/self.avg_time*10
				new_batches[i] = self.settings[i].local_epochs - self.settings[i].adjust_ratio*diff_batch_num[i]
				self.settings[i].local_epochs = max(1,round(new_batches[i]))
			elif self.settings[i].batch_adjustment == "direct":
				diff_batch_num[i] = (single_round_metrics["time@10"][i] - self.avg_time)/self.avg_time*10
				new_batches[i] = self.settings[i].local_epochs + diff_batch_num[i]
				self.settings[i].local_epochs = max(1,round(new_batches[i]))

		if self.settings[i].prune_adjustment == "adaptive":
			threshold = np.quantile(single_round_metrics["loss"],self.settings[i].adaptive_ratio)
			for i in range(self.num_clients):
				if single_round_metrics["loss"][i] > threshold:
					self.settings[i].compress_ratio += (1-self.settings[i].compress_ratio)/(epoch+2)
				else:
					self.settings[i].compress_ratio -= self.settings[i].compress_ratio/(epoch+2)

		print(list(setting.compress_ratio for setting in self.settings))
		print(list(setting.local_epochs for setting in self.settings))

	##################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nc',"--num_clients",type=int,default=50,help="Total number of clients")
    parser.add_argument('-miuc',"--min_user_per_client",type=int,default=1,help="Minimum number of user per client")
    parser.add_argument('-mauc',"--max_user_per_client",type=int,default=10,help="Maximum number of user per client")
    parser.add_argument('-ae',"--aggregation_epochs",type=int,default=50,help="Total number of aggregation epochs")
    parser.add_argument('-le',"--local_epochs",type=int,default=10,help="Total number of local epochs")
    parser.add_argument('-bs',"--batch_size",type=int,default=128,help="Local batch size during training")
    
    parser.add_argument('-pa',"--prune_adjustment",type=str,default=None,help="Type of adjust model pruning")
    parser.add_argument('-pt',"--prune_type",type=str,default=None,help="Type of model pruning")
    parser.add_argument('-cr',"--compress_ratio",type=float,default=1.0,help="The compressed ratio of the model")
    parser.add_argument('-ar',"--adaptive_ratio",type=float,default=0.5,help="The adaptive ratio during compression adjustment")
    
    parser.add_argument('-ba',"--batch_adjustment",type=str,default=None,help="Type of batch adjustment during training")
    parser.add_argument('-r1',"--momentum",type=float,default=0.0,help="Ratio of momentum during batch adjustment")
    parser.add_argument('-r2',"--adjust_ratio",type=float,default=0.0,help="Adjust ratio during batch adjustment")
	
    parser.add_argument('-d',"--device",type=int,default=0,help="Training device")
    parser.add_argument('-sr',"--slow_rate",type=int,default=0,help="Ratio of slow client")
    
    parser.add_argument('-ld',"--latent_dim",type=int,default=32,help="Latent dimension")
    parser.add_argument('-sd',"--seed",type=int,default=0,help="Random seed")
    parser.add_argument('-mn',"--model_name",type=str,default="NCF",help="Model used for training")
    
    args = parser.parse_args()
    
    dataloader = MovielensDatasetLoader()
    fncf = FederatedNCF(dataloader.ratings, args)
    fncf.train()