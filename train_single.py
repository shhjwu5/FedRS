import torch
from dataloader import MovielensDatasetLoader
from model import NeuralCollaborativeFiltering,GMF,MLP
import numpy as np
from tqdm import tqdm
from metrics import compute_metrics
import pandas as pd
import time

class MatrixLoader:
	def __init__(self, ui_matrix, default=None, seed=0):
		np.random.seed(seed)
		self.ui_matrix = ui_matrix
		self.positives = np.argwhere(self.ui_matrix!=0)
		self.negatives = np.argwhere(self.ui_matrix==0)
		if default is None:
			self.default = np.array([[0, 0]]), np.array([0])
		else:
			self.default = default

	def delete_indexes(self, indexes, arr="pos"):
		if arr=="pos":
			self.positives = np.delete(self.positives, indexes, 0)
		else:
			self.negatives = np.delete(self.negatives, indexes, 0)

	def get_batch(self, batch_size):
		if self.positives.shape[0]<batch_size//4 or self.negatives.shape[0]<batch_size-batch_size//4:
			return torch.tensor(self.default[0]), torch.tensor(self.default[1])
		try:
			pos_indexes = np.random.choice(self.positives.shape[0], batch_size//4)
			neg_indexes = np.random.choice(self.negatives.shape[0], batch_size - batch_size//4)
			pos = self.positives[pos_indexes]
			neg = self.negatives[neg_indexes]
			self.delete_indexes(pos_indexes, "pos")
			self.delete_indexes(neg_indexes, "neg")
			batch = np.concatenate((pos, neg), axis=0)
			if batch.shape[0]!=batch_size:
				return torch.tensor(self.default[0]), torch.tensor(self.default[1]).float()
			np.random.shuffle(batch)
			y = np.array([self.ui_matrix[i][j] for i,j in batch])
			return torch.tensor(batch), torch.tensor(y).float()
		except:
			return torch.tensor(self.default[0]), torch.tensor(self.default[1]).float()

class NCFTrainer:
	def __init__(self, id, ui_matrix, setting):
		self.id = id
		if setting.slow_rate != 0:
			self.slow_client = (self.id%setting.slow_rate)==0
		else:
			self.slow_client = 0
		self.ui_matrix = ui_matrix
		self.loader = None
		self.initialize_loader()
		self.device = torch.device("cuda:%d"%(setting.device) if torch.cuda.is_available() else "cpu")
		if setting.model_name=="NCF":
			self.ncf = NeuralCollaborativeFiltering(self.ui_matrix.shape[0], self.ui_matrix.shape[1], setting.latent_dim).to(self.device)
		elif setting.model_name=="GMF":
			self.ncf = GMF(self.ui_matrix.shape[0], self.ui_matrix.shape[1], setting.latent_dim).to(self.device)
		elif setting.model_name=="MLP":
			self.ncf = MLP(self.ui_matrix.shape[0], self.ui_matrix.shape[1], setting.latent_dim).to(self.device)

	def initialize_loader(self):
		self.loader = MatrixLoader(self.ui_matrix)

	def train_batch(self, x, y, optimizer):
		y_ = self.ncf(x)
		mask = (y>0).float()
		loss = torch.nn.functional.mse_loss(y_*mask, y)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		return loss.item(), y_.detach()

	def train_model(self, optimizer, print_num=10):
		epoch = 0
		##################################
		progress = {"epoch": [], "loss": [], "hit_ratio@10": [], "ndcg@10": [], "compress@10": [], "time@10": []}
		running_loss, running_hr, running_ndcg, running_compress, prev_running_time = 0, 0, 0, 1, 0
		prev_running_loss, prev_running_hr, prev_running_ndcg, prev_running_compress, running_time = 0, 0, 0, 1, 0
		
		self.epochs = self.setting.local_epochs
		self.batch_size = self.setting.batch_size
  
		start_time = time.time()
		##################################
  
		steps, prev_steps, prev_epoch = 0, 0, 0
		while epoch<self.epochs:
			x, y = self.loader.get_batch(self.batch_size)
			if x.shape[0]<self.batch_size:
				##################################
				prev_running_loss, prev_running_hr, prev_running_ndcg, prev_running_compress,prev_running_time = running_loss, running_hr, running_ndcg, running_compress, running_time
				running_loss = 0
				running_hr = 0
				running_ndcg = 0
				running_compress = 1
				running_time = 0
				##################################
				prev_steps = steps
				steps = 0
				epoch += 1
				self.initialize_loader()
				x, y = self.loader.get_batch(self.batch_size)
			x, y = x.int(), y.float()
			x, y = x.to(self.device), y.to(self.device)
			loss, y_ =	self.train_batch(x, y, optimizer)
			hr, ndcg = compute_metrics(y.cpu().numpy(), y_.cpu().numpy())
   
			##################################
			total_param = 0
			non_zero_param = 0
			for param in self.ncf.parameters():
				if self.setting.prune_type == "ratio":
					max_value = torch.max(param)
					min_value = torch.min(param)
					threshold = min_value + (1-self.setting.compress_ratio)*(max_value-min_value)
					param = param*(param>threshold)
				elif self.setting.prune_type == "quantile":
					threshold = np.quantile(param.detach().clone().cpu().numpy(),1-self.setting.compress_ratio)
					param = param*(param>threshold)
				total_param += torch.prod(torch.tensor(param.shape))
				non_zero_param += torch.sum(param!=0)
			self.compressed = non_zero_param/total_param
			# print(self.compressed)
			###################################
   
			###################################
			local_time = time.time()-start_time
			if self.slow_client:
				local_time*=2
			##################################

			running_loss += loss
			running_hr += hr
			running_ndcg += ndcg
   
			##################################
			running_compress += float(self.compressed)
			if epoch!=0 and steps==0:
				results = {"epoch": prev_epoch, "loss": prev_running_loss/(prev_steps+1), "hit_ratio@10": prev_running_hr/(prev_steps+1), "ndcg@10": prev_running_ndcg/(prev_steps+1), "compress@10": prev_running_compress/(prev_steps+1), "time@10": local_time}
			else:
				results = {"epoch": prev_epoch, "loss": running_loss/(steps+1), "hit_ratio@10": running_hr/(steps+1), "ndcg@10": running_ndcg/(steps+1), "compress@10": running_compress/(steps+1), "time@10": local_time}
			steps += 1
			if prev_epoch!=epoch:
				progress["epoch"].append(results["epoch"])
				progress["loss"].append(results["loss"])
				progress["hit_ratio@10"].append(results["hit_ratio@10"])
				progress["ndcg@10"].append(results["ndcg@10"])
				progress["compress@10"].append(results["compress@10"])
				progress["time@10"].append(results["time@10"])
				prev_epoch+=1
		r_results = {"num_users": self.ui_matrix.shape[0]}
		r_results.update({i:results[i] for i in ["loss", "hit_ratio@10", "ndcg@10", "compress@10", "time@10"]})
		##################################
  
		return r_results, progress

	def train(self, ncf_optimizer, setting, return_progress=False):
		self.ncf.join_output_weights()
		self.setting = setting
		results, progress = self.train_model(ncf_optimizer)
		if return_progress:
			return results, progress
		else:
			return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nc',"--num_clients",type=int,default=50,help="Total number of clients")
    parser.add_argument('-miuc',"--min_user_per_client",type=int,default=1,help="Minimum number of user per client")
    parser.add_argument('-mauc',"--max_user_per_client",type=int,default=10,help="Maximum number of user per client")
    parser.add_argument('-ae',"--aggregation_epochs",type=int,default=50,help="Total number of aggregation epochs")
    parser.add_argument('-le',"--local_epochs",type=int,default=10,help="Total number of local epochs")
    parser.add_argument('-bs',"--batch_size",type=int,default=128,help="Local batch size during training")
    
    parser.add_argument('-pt',"--prune_type",type=str,default=None,help="Type of model pruning")
    parser.add_argument('-cr',"--compress_ratio",type=float,default=1.0,help="The compressed ratio of the model")
    
    parser.add_argument('-ba',"--batch_adjustment",type=str,default=None,help="Type of batch adjustment during training")
    parser.add_argument('-r1',"--momentum",type=float,default=0.0,help="Ratio of momentum during batch adjustment")
    parser.add_argument('-r2',"--adjust_ratio",type=float,default=0.0,help="Adjust ratio during batch adjustment")
    
    parser.add_argument('-d',"--device",type=int,default=0,help="Training device")
    
    parser.add_argument('-ld',"--latent_dim",type=int,default=32,help="Latent dimension")
    parser.add_argument('-sd',"--seed",type=int,default=0,help="Random seed")
    
    args = parser.parse_args()
    
    dataloader = MovielensDatasetLoader()
    trainer = NCFTrainer(dataloader.ratings[:50], args)
    ncf_optimizer = torch.optim.Adam(trainer.ncf.parameters(), lr=5e-4)
    _, progress = trainer.train(ncf_optimizer, args, return_progress=True)