import os
import json
import glob
import random
import argparse
from argparse import Namespace
import numpy as np

import torch

from torch_geometric.loader import DataLoader

import prodar
from prodar import ProDAR

if __name__ == "__main__":
	
	print("Initialization...")

	cases = sorted(glob.glob('./history/*/*/*-model.pt'))

	print('\tfound {} models to evaluate'.format(len(cases)))

	for case in cases:

		if os.path.isfile(case.replace('-model.pt', '.cm')):
			print(f'Skip analyzing model {case}')
			continue

		print(f'Analyzing model {case}...')
		
		with open(case.replace('-model.pt', '.args'), 'r') as f:
			config_json = json.load(f)
			config = Namespace(**config_json)

		device = torch.device('cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu')

		np.random.seed(config.seed)
		random.seed(config.seed)
		torch.manual_seed(config.seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(config.seed)
			torch.cuda.manual_seed_all(config.seed)

		# torch.backends.cudnn.enabled = False
		# torch.backends.cudnn.benchmark = False
		# # torch.backends.cudnn.deterministic = True
		# torch.set_deterministic(True)
		# torch.use_deterministic_algorithms(True)

		def seed_worker(worker_id):
			worker_seed = torch.initial_seed() % 2**32
			np.random.seed(worker_seed)
			random.seed(worker_seed)

		g = torch.Generator()
		g.manual_seed(0)

		model = ProDAR(
			# dataset=config.dataset, 
			# model=config.model, device=config.device, 
			# num_layers=config.num_layers, 
			# dim_node_embedding=config.dim_node_embedding, 
			# dim_graph_embedding=config.dim_graph_embedding, 
			# dim_pers_embedding=config.dim_pers_embedding,
			**vars(config)
		)

		model.load(case)

		dataset = model.dataset.shuffle()

		# train_dataset = dataset[:int(len(dataset)*0.9)]

		# valid_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]

		test_dataset = dataset[int(len(dataset)*0.9):]

		test_loader = DataLoader(test_dataset,
			batch_size=config.batch_size,
			num_workers=config.num_workers,
			worker_init_fn=seed_worker,
			generator=g,
			sampler=None
		)

		thresx = np.linspace(0, 1, int(1.25e2+1))

		with open(case.replace('-model.pt', '.cm'), 'w') as f:
			f.write(f'Threshold\tTP\tFP\tTN\tFN\n')
			for thres in thresx:
				
				total_tp, total_fp, total_tn, total_fn = model.predict(
					test_loader, thres=thres, **vars(config))

				f.write(f'{thres:10.5e}\t {total_tp:10d}\t {total_fp:10d}\t {total_tn:10d}\t {total_fn:10d}\n')
				f.flush()