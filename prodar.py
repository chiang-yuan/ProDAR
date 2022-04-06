import glob
import json
import inspect
import numpy as np

from torch_geometric.data import Data, Dataset

import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts

from torchinfo import summary

from sklearn.model_selection import KFold

from tqdm import tqdm

from datasets.dataset import ContactCorr12A, Contact12A, ContactCorr10A, Contact10A, ContactCorr8A, Contact8A
from models.multilabel_classifiers.GraphSAGE import GraphSAGE
from models.multilabel_classifiers.GCN import GCN
from models.multilabel_classifiers.GAT import GAT

data_dir = 'data'

datasets_dir = 'datasets'

pdbgos_file = 'pdbmfgos-thres-50.json'

datasets = {
    'ContactCorr12A': ContactCorr12A,
    'Contact12A': Contact12A,
    'ContactCorr10A': ContactCorr10A,
    'Contact10A': Contact10A,
    'ContactCorr8A': ContactCorr8A,
    'Contact8A': Contact8A
}

graphs = {
    'ContactCorr12A': 'graphs-12A',
    'Contact12A': 'graphs-12A',
    'ContactCorr10A': 'graphs-10A',
    'Contact10A': 'graphs-10A',
    'ContactCorr8A': 'graphs-8A',
    'Contact8A': 'graphs-8A'
}

models = {
    'GraphSAGE': GraphSAGE,
    'GCN': GCN,
    'GAT': GAT
}

optimizers = {
    'SGD': SGD,
    'Adam': Adam,
    'AdamW': AdamW
}

schedulers = {
    'ReduceLROnPlateau': ReduceLROnPlateau,
    'CosAnnealWR': CosineAnnealingWarmRestarts
}

"""amino acids
standard amino acid residues:
    ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL
non-standard amino acid residues:
    ASX, GLX, CSO, HIP, HSD, HSE, HSP, MSE, SEC, SEP, TPO, PTR, XLE, XAA
"""

table = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU':10, 'LYS':11, 'MET':12, 'PHE':13, 'PRO':14,
    'SER':15, 'THR':16, 'TRP':17, 'TYR':18, 'VAL':19,
    'ASX': 3, 'GLX': 6, 'CSO': 4, 'HIP': 8, 'HSD': 8,
    'HSE': 8, 'HSP': 8, 'MSE':12, 'SEC': 4, 'SEP':15,
    'TPO':16, 'PTR':18, 'XLE':10, 'XAA':20
}

class ProDAR:

    def __init__(self, dataset, model, device, 
        num_layers, dim_node_embedding, dim_graph_embedding, 
        dim_pers_embedding, **kwargs):

        print("Loading PDB-GO pairs...")

        with open(f'{data_dir}/sifts/{pdbgos_file}', "r") as in_file:
            pdbgos = json.load(in_file)

        # flatten all graph labels in a single list and find the unique labels
        flattened = [label \
            for multilabels in list(pdbgos.values()) \
            for label in multilabels]
        uniques, counts = np.unique(flattened, return_counts=True)
        
        # find the missing labels
        labels = np.arange(np.min(uniques), np.max(uniques)+1)
        mask = np.logical_not(np.isin(labels, uniques))
        missing = labels[mask]
        sorter = np.argsort(uniques)
        idx = sorter[np.searchsorted(uniques, missing, sorter=sorter)]
        
        # patch zero for missing labels
        counts = np.insert(counts, idx, 0)
        label_counts = torch.from_numpy(counts)

        print("Loading PDB IDs...")

        pdbids = sorted(list(
            map(lambda x: x.split('.')[-2].split('/')[-1],
                glob.glob(f'{data_dir}/{graphs[dataset]}/*.json'))
        ))

        print("Loading dataset...")

        self.dataset = datasets[dataset](
            root=datasets_dir,
            data_dir=data_dir,
            pdbids=pdbids, pdbgos=pdbgos, encode_table=table,
            **kwargs
        )

        self.num_layers = num_layers
        self.dim_node_embedding = dim_node_embedding
        self.dim_graph_embedding = dim_graph_embedding

        self.dim_pers = self.dataset[0].pi.shape[1]
        self.dim_pers_embedding = dim_pers_embedding

        self.dim_target = self.dataset[0].y.shape[1]

        print(self.dataset)
        print(f'\tnumber of graphs: {len(self.dataset)}')
        print(f'\tnumber of graph classes: {self.dim_target}')
        print(f'\tnumber of node features: {self.dataset.num_node_features}')
        print(f'\tnumber of persistence features: {self.dim_pers}')

        # count label
        # label_counts = torch.zeros(self.dataset[0].y.shape)
        # for data in tqdm(self.dataset):
        #     data.x = data.x.float()
        #     data.y = data.y.float()
        #     label_counts += data.y

        # calculate positional weight for criterion
        numer = len(self.dataset) - label_counts
        denom = label_counts
        self.pos_weight = torch.full_like(label_counts, len(self.dataset), dtype=torch.float)
        mask = (denom != 0)
        self.pos_weight[mask] = torch.div(numer[mask], denom[mask])

        print("Building model...")

        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        print(f'\tusing device: {self.device}')

        model = models[model](
            dim_features=self.dataset.num_node_features,
            dim_pers=self.dim_pers,
            dim_target=self.dim_target,
            num_layers=self.num_layers,
            dim_node_embedding=self.dim_node_embedding,
            dim_graph_embedding=self.dim_graph_embedding,
            dim_pers_embedding=self.dim_pers_embedding,
            **kwargs
        )
        
        self.model = nn.DataParallel(model).to(self.device)

        print(self.model)
        summary(self.model)
    
    def save(self, f):
        torch.save(self.model, f)

    def load(self, f):
        self.model = torch.load(f)

    def set_optimizer(self, optimizer, lr, **kwargs):

        kwargs_optimizer = {}
        for name in inspect.getfullargspec(optimizers[optimizer].__init__).args:
            if name in kwargs and kwargs[name] is not None:
                kwargs_optimizer[name] = kwargs[name]

        self.optimizer = optimizers[optimizer](
            self.model.parameters(),
            lr = lr,
            **kwargs_optimizer
        )

        return self.optimizer

    def set_scheduler(self, scheduler, **kwargs):
        
        if scheduler is None:
            return None
        
        kwargs_scheduler = {}
        for name in inspect.getfullargspec(schedulers[scheduler].__init__).args:
            if name in kwargs and \
             kwargs[name] is not None and \ 
             name != "optimizer":
                kwargs_scheduler[name] = kwargs[name]

        self.scheduler = schedulers[scheduler](
            self.optimizer,
            verbose=True,
            **kwargs_scheduler
        )

        return self.scheduler 

    def train(self, train_loader, epoch, thres=0.5,
        optimizer='Adam', lr=1e-5, scheduler=None, 
        **kwargs):

        self.model.train()

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(self.device))

        self.set_optimizer(optimizer, lr, **kwargs)

        self.set_scheduler(scheduler, **kwargs)
        
        # kwargs_optimizer = {}
        # for name in inspect.getfullargspec(optimizers[optimizer].__init__).args:
        #     if name in kwargs and kwargs[name] is not None:
        #         kwargs_optimizer[name] = kwargs[name]

        # self.optimizer = optimizers[optimizer](
        #     self.model.parameters(),
        #     lr = lr,
        #     **kwargs_optimizer
        # )

        # if scheduler is not None:
        #     kwargs_scheduler = {}
        #     for name in inspect.getfullargspec(schedulers[scheduler].__init__).args:
        #         if name in kwargs and kwargs[name] is not None:
        #             kwargs_scheduler[name] = kwargs[name]

        #     self.scheduler = schedulers[scheduler](
        #         self.optimizer,
        #         verbose=True,
        #         **kwargs_scheduler
        #     )

        total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
        total_loss = 0

        for i, data in enumerate(tqdm(train_loader, desc=f'{self.model.__class__.__name__}: train')):

            data.x = data.x.float()
            data.y = data.y.float()

            data_device = data.to(self.device)

            out = self.model(data_device)

            loss = self.criterion(out, data_device.y)
            total_loss += loss.item()*out.shape[0]

            pred = torch.where(torch.sigmoid(out) >= thres, 1, 0)

            tp = torch.logical_and(pred == 1,  data_device.y == 1).cpu().detach().numpy().sum()
            fp = torch.logical_and(pred == 1,  data_device.y == 0).cpu().detach().numpy().sum()
            tn = torch.logical_and(pred == 0,  data_device.y == 0).cpu().detach().numpy().sum()
            fn = torch.logical_and(pred == 0,  data_device.y == 1).cpu().detach().numpy().sum()

            total_tp += int(tp)
            total_fp += int(fp)
            total_tn += int(tn)
            total_fn += int(fn)

            loss.backward()
            self.optimizer.step()

            if scheduler == 'CosAnnealWR':
                self.scheduler.step(epoch + i / len(train_loader))

            del data, data_device, pred, tp, fp, tn, fn
            torch.cuda.empty_cache()

        if train_loader.sampler is not None:
            total_loss = total_loss / len(train_loader.sampler.indices)
        else:
            total_loss = total_loss / len(train_loader.dataset)

        ppv = total_tp / (total_tp + total_fp) if (total_tp + total_fp) != 0 else 0
        tpr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) != 0 else 0
        tnr = total_tn / (total_tn + total_fp) if (total_tn + total_fp) != 0 else 0

        return total_loss, ppv, tpr, tnr

    def valid(self, loader, thres=0.5, **kwargs):

        with torch.no_grad():

            self.model.eval()

            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(self.device))
        
            total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
            total_loss = 0

            for i, data in enumerate(tqdm(loader, desc=f'{self.model.__class__.__name__}: predict')):

                data.x = data.x.float()
                data.y = data.y.float()

                data_device = data.to(self.device)

                out = self.model(data_device)

                loss = self.criterion(out, data_device.y)
                total_loss += loss.item()*out.shape[0]

                pred = torch.where(torch.sigmoid(out) >= thres, 1, 0)

                tp = torch.logical_and(pred == 1,  data_device.y == 1).cpu().detach().numpy().sum()
                fp = torch.logical_and(pred == 1,  data_device.y == 0).cpu().detach().numpy().sum()
                tn = torch.logical_and(pred == 0,  data_device.y == 0).cpu().detach().numpy().sum()
                fn = torch.logical_and(pred == 0,  data_device.y == 1).cpu().detach().numpy().sum()

                total_tp += int(tp)
                total_fp += int(fp)
                total_tn += int(tn)
                total_fn += int(fn)

                del data, data_device, pred, tp, fp, tn, fn
                torch.cuda.empty_cache()

        if loader.sampler is not None:
            total_loss = total_loss / len(loader.sampler.indices)
        else:
            total_loss = total_loss / len(loader.dataset)

        ppv = total_tp / (total_tp + total_fp) if (total_tp + total_fp) != 0 else 0
        tpr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) != 0 else 0
        tnr = total_tn / (total_tn + total_fp) if (total_tn + total_fp) != 0 else 0

        return total_loss, ppv, tpr, tnr

    def predict(self, loader, thres=0.5, **kwargs):

        with torch.no_grad():

            self.model.eval()

            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(self.device))
        
            total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
            total_loss = 0

            for i, data in enumerate(tqdm(loader, desc=f'{self.model.__class__.__name__}: predict')):

                data.x = data.x.float()
                data.y = data.y.float()

                data_device = data.to(self.device)

                out = self.model(data_device)

                loss = self.criterion(out, data_device.y)
                total_loss += loss.item()*out.shape[0]

                pred = torch.where(torch.sigmoid(out) >= thres, 1, 0)

                tp = torch.logical_and(pred == 1,  data_device.y == 1).cpu().detach().numpy().sum()
                fp = torch.logical_and(pred == 1,  data_device.y == 0).cpu().detach().numpy().sum()
                tn = torch.logical_and(pred == 0,  data_device.y == 0).cpu().detach().numpy().sum()
                fn = torch.logical_and(pred == 0,  data_device.y == 1).cpu().detach().numpy().sum()

                total_tp += int(tp)
                total_fp += int(fp)
                total_tn += int(tn)
                total_fn += int(fn)

                del data, data_device, pred, tp, fp, tn, fn
                torch.cuda.empty_cache()

        return total_tp, total_fp, total_tn, total_fn