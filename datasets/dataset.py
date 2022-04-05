import os
import json
import networkx as nx
import numpy as np

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx

from argparse import Namespace

from tqdm.auto import tqdm

class Contact12A(Dataset):
    def __init__(
        self, root, data_dir, pdbids, pdbgos, encode_table,
        transform=None, pre_transform=None, pre_filter=None, **kwargs):
        
        self.graph_dir = os.path.join(data_dir, 'graphs-12A')
        self.pi_dir = os.path.join(data_dir, 'pis')
        self.pdbids = sorted(pdbids)
        self.pdbgos = pdbgos
        self.encode_table = encode_table
        self.kwargs = kwargs
        super(Contact12A, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.graph_dir)
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'contact-12A')
    
    @property
    def raw_file_names(self):
        return list(map(lambda x: f'{x}.json', self.pdbids))

    @property
    def processed_file_names(self):
        return list(map(lambda x: '{:06d}.pt'.format(x), range(len(self.pdbids))))

    def download(self):
        pass

    def process(self):
        for i, pdbid in enumerate(tqdm(self.pdbids, desc=self.__class__.__name__)):
            
            if 'demo' in self.kwargs and 'demo_num' in self.kwargs:
                if self.kwargs['demo'] and i >= self.kwargs['demo_num']:
                    break
            
            with open(f'{self.graph_dir}/{pdbid}.json') as f:
                json_graph = json.load(f)
            G = nx.readwrite.json_graph.node_link_graph(json_graph)
            
            data = from_networkx(G)

            # delete correlation edge (edge weight == -1)

            indices = torch.nonzero(torch.where(data.weight == 1, 1, 0)).squeeze()
            
            data.edge_index = torch.index_select(data.edge_index, 1, indices)
            
            # ========
            # features
            # ========
            
            uniques, counts = np.unique(list(self.encode_table.values()), return_counts=True)
            
            x = np.zeros((len(data.resname), len(uniques)))
            
            for j, residue in enumerate(data.resname):
                if residue not in self.encode_table:
                    residue = 'XAA'
                x[j, self.encode_table[residue]] = 1
                
            data.x = torch.from_numpy(x).int()

            pi = np.load(f'{self.pi_dir}/{pdbid}.npy')

            data.pi = torch.from_numpy(pi).float()
    
            # =====
            # label
            # =====
            
            num_targets = max(list(map(lambda t: [max(t)] if t else [], list(self.pdbgos.values()))))[0]+1
            y = np.zeros((1, num_targets))
            y[0, self.pdbgos[pdbid]] += 1

            data.y = torch.from_numpy(y).int()

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # remove unnecessary objects

            del data.resname, data.weight

            torch.save(data, os.path.join(self.processed_dir, '{:06d}.pt'.format(i)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, '{:06d}.pt'.format(idx)))


class ContactCorr12A(Dataset):
    def __init__(
        self, root, data_dir, pdbids, pdbgos, encode_table, 
        transform=None, pre_transform=None, pre_filter=None, **kwargs):
        
        self.graph_dir = os.path.join(data_dir, 'graphs-12A')
        self.pi_dir = os.path.join(data_dir, 'pis')
        self.pdbids = sorted(pdbids)
        self.pdbgos = pdbgos
        self.encode_table = encode_table
        self.kwargs = kwargs
        super(ContactCorr12A, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.graph_dir)
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'contact-corr-12A')
    
    @property
    def raw_file_names(self):
        return list(map(lambda x: f'{x}.json', self.pdbids))

    @property
    def processed_file_names(self):
        return list(map(lambda x: '{:06d}.pt'.format(x), range(len(self.pdbids))))

    def download(self):
        pass

    def process(self):
        for i, pdbid in enumerate(tqdm(self.pdbids, desc=self.__class__.__name__)):
            
            if 'demo' in self.kwargs and 'demo_num' in self.kwargs:
                if self.kwargs['demo'] and i >= self.kwargs['demo_num']:
                    break
            
            with open(f'{self.graph_dir}/{pdbid}.json') as f:
                json_graph = json.load(f)
            G = nx.readwrite.json_graph.node_link_graph(json_graph)
            
            data = from_networkx(G)
            
            # ========
            # features
            # ========
            
            uniques, counts = np.unique(list(self.encode_table.values()), return_counts=True)
            
            x = np.zeros((len(data.resname), len(uniques)))
            
            for j, residue in enumerate(data.resname):
                if residue not in self.encode_table:
                    residue = 'XAA'
                x[j, self.encode_table[residue]] = 1
                
            data.x = torch.from_numpy(x).int()

            pi = np.load(f'{self.pi_dir}/{pdbid}.npy')

            data.pi = torch.from_numpy(pi).float()
    
            # =====
            # label
            # =====
            
            num_targets = max(list(map(lambda t: [max(t)] if t else [], list(self.pdbgos.values()))))[0]+1
            y = np.zeros((1, num_targets))
            y[0, self.pdbgos[pdbid]] += 1

            data.y = torch.from_numpy(y).int()

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # remove unnecessary objects

            del data.resname, data.weight

            torch.save(data, os.path.join(self.processed_dir, '{:06d}.pt'.format(i)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, '{:06d}.pt'.format(idx)))


class Contact10A(Dataset):
    def __init__(
        self, root, data_dir, pdbids, pdbgos, encode_table, 
        transform=None, pre_transform=None, pre_filter=None, **kwargs):
        
        self.graph_dir = os.path.join(data_dir, 'graphs-10A')
        self.pi_dir = os.path.join(data_dir, 'pis')
        self.pdbids = sorted(pdbids)
        self.pdbgos = pdbgos
        self.encode_table = encode_table
        self.kwargs = kwargs
        super(Contact10A, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.graph_dir)
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'contact-10A')
    
    @property
    def raw_file_names(self):
        return list(map(lambda x: f'{x}.json', self.pdbids))

    @property
    def processed_file_names(self):
        return list(map(lambda x: '{:06d}.pt'.format(x), range(len(self.pdbids))))

    def download(self):
        pass

    def process(self):
        for i, pdbid in enumerate(tqdm(self.pdbids, desc=self.__class__.__name__)):
            
            if 'demo' in self.kwargs and 'demo_num' in self.kwargs:
                if self.kwargs['demo'] and i >= self.kwargs['demo_num']:
                    break
            
            with open(f'{self.graph_dir}/{pdbid}.json') as f:
                json_graph = json.load(f)
            G = nx.readwrite.json_graph.node_link_graph(json_graph)
            
            data = from_networkx(G)

            # delete correlation edge (edge weight == -1)

            indices = torch.nonzero(torch.where(data.weight == 1, 1, 0)).squeeze()
            
            data.edge_index = torch.index_select(data.edge_index, 1, indices)
            
            # ========
            # features
            # ========
            
            uniques, counts = np.unique(list(self.encode_table.values()), return_counts=True)
            
            x = np.zeros((len(data.resname), len(uniques)))
            
            for j, residue in enumerate(data.resname):
                if residue not in self.encode_table:
                    residue = 'XAA'
                x[j, self.encode_table[residue]] = 1
                
            data.x = torch.from_numpy(x).int()

            pi = np.load(f'{self.pi_dir}/{pdbid}.npy')

            data.pi = torch.from_numpy(pi).float()
    
            # =====
            # label
            # =====
            
            num_targets = max(list(map(lambda t: [max(t)] if t else [], list(self.pdbgos.values()))))[0]+1
            y = np.zeros((1, num_targets))
            y[0, self.pdbgos[pdbid]] += 1

            data.y = torch.from_numpy(y).int()

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # remove unnecessary objects

            del data.resname, data.weight

            torch.save(data, os.path.join(self.processed_dir, '{:06d}.pt'.format(i)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, '{:06d}.pt'.format(idx)))


class ContactCorr10A(Dataset):
    def __init__(
        self, root, data_dir, pdbids, pdbgos, encode_table, 
        transform=None, pre_transform=None, pre_filter=None, **kwargs):
        
        self.graph_dir = os.path.join(data_dir, 'graphs-10A')
        self.pi_dir = os.path.join(data_dir, 'pis')
        self.pdbids = sorted(pdbids)
        self.pdbgos = pdbgos
        self.encode_table = encode_table
        self.kwargs = kwargs
        super(ContactCorr10A, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.graph_dir)
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'contact-corr-10A')
    
    @property
    def raw_file_names(self):
        return list(map(lambda x: f'{x}.json', self.pdbids))

    @property
    def processed_file_names(self):
        return list(map(lambda x: '{:06d}.pt'.format(x), range(len(self.pdbids))))

    def download(self):
        pass

    def process(self):
        for i, pdbid in enumerate(tqdm(self.pdbids, desc=self.__class__.__name__)):
            
            if 'demo' in self.kwargs and 'demo_num' in self.kwargs:
                if self.kwargs['demo'] and i >= self.kwargs['demo_num']:
                    break
            
            with open(f'{self.graph_dir}/{pdbid}.json') as f:
                json_graph = json.load(f)
            G = nx.readwrite.json_graph.node_link_graph(json_graph)
            
            data = from_networkx(G)
            
            # ========
            # features
            # ========
            
            uniques, counts = np.unique(list(self.encode_table.values()), return_counts=True)
            
            x = np.zeros((len(data.resname), len(uniques)))
            
            for j, residue in enumerate(data.resname):
                if residue not in self.encode_table:
                    residue = 'XAA'
                x[j, self.encode_table[residue]] = 1
                
            data.x = torch.from_numpy(x).int()

            pi = np.load(f'{self.pi_dir}/{pdbid}.npy')

            data.pi = torch.from_numpy(pi).float()
    
            # =====
            # label
            # =====
            
            num_targets = max(list(map(lambda t: [max(t)] if t else [], list(self.pdbgos.values()))))[0]+1
            y = np.zeros((1, num_targets))
            y[0, self.pdbgos[pdbid]] += 1

            data.y = torch.from_numpy(y).int()

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # remove unnecessary objects

            del data.resname, data.weight

            torch.save(data, os.path.join(self.processed_dir, '{:06d}.pt'.format(i)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, '{:06d}.pt'.format(idx)))
        return data


class Contact8A(Dataset):
    def __init__(
        self, root, data_dir, pdbids, pdbgos, encode_table, 
        transform=None, pre_transform=None, pre_filter=None, **kwargs):
        
        self.graph_dir = os.path.join(data_dir, 'graphs-8A')
        self.pi_dir = os.path.join(data_dir, 'pis')
        self.pdbids = sorted(pdbids)
        self.pdbgos = pdbgos
        self.encode_table = encode_table
        self.kwargs = kwargs
        super(Contact8A, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.graph_dir)
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'contact-8A')
    
    @property
    def raw_file_names(self):
        return list(map(lambda x: f'{x}.json', self.pdbids))

    @property
    def processed_file_names(self):
        return list(map(lambda x: '{:06d}.pt'.format(x), range(len(self.pdbids))))

    def download(self):
        pass

    def process(self):
        for i, pdbid in enumerate(tqdm(self.pdbids, desc=self.__class__.__name__)):
            
            if 'demo' in self.kwargs and 'demo_num' in self.kwargs:
                if self.kwargs['demo'] and i >= self.kwargs['demo_num']:
                    break
            
            with open(f'{self.graph_dir}/{pdbid}.json') as f:
                json_graph = json.load(f)
            G = nx.readwrite.json_graph.node_link_graph(json_graph)
            
            data = from_networkx(G)

            # delete correlation edge (edge weight == -1)

            indices = torch.nonzero(torch.where(data.weight == 1, 1, 0)).squeeze()
            
            data.edge_index = torch.index_select(data.edge_index, 1, indices)
            
            # ========
            # features
            # ========
            
            uniques, counts = np.unique(list(self.encode_table.values()), return_counts=True)
            
            x = np.zeros((len(data.resname), len(uniques)))
            
            for j, residue in enumerate(data.resname):
                if residue not in self.encode_table:
                    residue = 'XAA'
                x[j, self.encode_table[residue]] = 1
                
            data.x = torch.from_numpy(x).int()

            pi = np.load(f'{self.pi_dir}/{pdbid}.npy')

            data.pi = torch.from_numpy(pi).float()
    
            # =====
            # label
            # =====
            
            num_targets = max(list(map(lambda t: [max(t)] if t else [], list(self.pdbgos.values()))))[0]+1
            y = np.zeros((1, num_targets))
            y[0, self.pdbgos[pdbid]] += 1

            data.y = torch.from_numpy(y).int()

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # remove unnecessary objects

            del data.resname, data.weight

            torch.save(data, os.path.join(self.processed_dir, '{:06d}.pt'.format(i)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, '{:06d}.pt'.format(idx)))


class ContactCorr8A(Dataset):
    def __init__(
        self, root, data_dir, pdbids, pdbgos, encode_table,
        transform=None, pre_transform=None, pre_filter=None, **kwargs):
        
        self.graph_dir = os.path.join(data_dir, 'graphs-8A')
        self.pi_dir = os.path.join(data_dir, 'pis')
        self.pdbids = sorted(pdbids)
        self.pdbgos = pdbgos
        self.encode_table = encode_table
        self.kwargs = kwargs
        super(ContactCorr8A, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.graph_dir)
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'contact-corr-8A')
    
    @property
    def raw_file_names(self):
        return list(map(lambda x: f'{x}.json', self.pdbids))

    @property
    def processed_file_names(self):
        return list(map(lambda x: '{:06d}.pt'.format(x), range(len(self.pdbids))))

    def download(self):
        pass

    def process(self):
        for i, pdbid in enumerate(tqdm(self.pdbids, desc=self.__class__.__name__)):
            
            if 'demo' in self.kwargs and 'demo_num' in self.kwargs:
                if self.kwargs['demo'] and i >= self.kwargs['demo_num']:
                    break
            
            with open(f'{self.graph_dir}/{pdbid}.json') as f:
                json_graph = json.load(f)
            G = nx.readwrite.json_graph.node_link_graph(json_graph)
            
            data = from_networkx(G)
            
            # ========
            # features
            # ========
            
            uniques, counts = np.unique(list(self.encode_table.values()), return_counts=True)
            
            x = np.zeros((len(data.resname), len(uniques)))
            
            for j, residue in enumerate(data.resname):
                if residue not in self.encode_table:
                    residue = 'XAA'
                x[j, self.encode_table[residue]] = 1
                
            data.x = torch.from_numpy(x).int()

            pi = np.load(f'{self.pi_dir}/{pdbid}.npy')

            data.pi = torch.from_numpy(pi).float()
    
            # =====
            # label
            # =====
            
            num_targets = max(list(map(lambda t: [max(t)] if t else [], list(self.pdbgos.values()))))[0]+1
            y = np.zeros((1, num_targets))
            y[0, self.pdbgos[pdbid]] += 1

            data.y = torch.from_numpy(y).int()

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # remove unnecessary objects

            del data.resname, data.weight

            torch.save(data, os.path.join(self.processed_dir, '{:06d}.pt'.format(i)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, '{:06d}.pt'.format(idx)))


    def __init__(
        self, root, data_dir, pdbids, pdbgos, encode_table,
        transform=None, pre_transform=None, pre_filter=None, **kwargs):
        
        self.graph_dir = os.path.join(data_dir, 'graphs-8A')
        self.pi_dir = os.path.join(data_dir, 'pis')
        self.pdbids = sorted(pdbids)
        self.pdbgos = pdbgos
        self.encode_table = encode_table
        self.kwargs = kwargs
        super(ContactCorr8A, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.graph_dir)
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'contact-corr-8A')
    
    @property
    def raw_file_names(self):
        return list(map(lambda x: f'{x}.json', self.pdbids))

    @property
    def processed_file_names(self):
        return list(map(lambda x: '{:06d}.pt'.format(x), range(len(self.pdbids))))

    def download(self):
        pass

    def process(self):
        for i, pdbid in enumerate(tqdm(self.pdbids, desc=self.__class__.__name__)):
            
            if 'demo' in self.kwargs and 'demo_num' in self.kwargs:
                if self.kwargs['demo'] and i >= self.kwargs['demo_num']:
                    break
            
            with open(f'{self.graph_dir}/{pdbid}.json') as f:
                json_graph = json.load(f)
            G = nx.readwrite.json_graph.node_link_graph(json_graph)
            
            data = from_networkx(G)
            
            # ========
            # features
            # ========
            
            uniques, counts = np.unique(list(self.encode_table.values()), return_counts=True)
            
            x = np.zeros((len(data.resname), len(uniques)))
            
            for j, residue in enumerate(data.resname):
                if residue not in self.encode_table:
                    residue = 'XAA'
                x[j, self.encode_table[residue]] = 1
                
            data.x = torch.from_numpy(x).int()

            pi = np.load(f'{self.pi_dir}/{pdbid}.npy')

            data.pi = torch.from_numpy(pi).float()
    
            # =====
            # label
            # =====
            
            num_targets = max(list(map(lambda t: [max(t)] if t else [], list(self.pdbgos.values()))))[0]+1
            y = np.zeros((1, num_targets))
            y[0, self.pdbgos[pdbid]] += 1

            data.y = torch.from_numpy(y).int()

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # remove unnecessary objects

            del data.resname, data.weight

            torch.save(data, os.path.join(self.processed_dir, '{:06d}.pt'.format(i)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, '{:06d}.pt'.format(idx)))


class UnlabeledContact8A(Dataset):
    def __init__(
        self, root, data_dir, pdbids, encode_table, 
        transform=None, pre_transform=None, pre_filter=None, **kwargs):
        
        self.graph_dir = os.path.join(data_dir, 'unlabeled-8A')
        self.pi_dir = os.path.join(data_dir, 'pis')
        self.pdbids = sorted(pdbids)
        self.encode_table = encode_table
        self.kwargs = kwargs
        super(UnlabeledContact8A, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.graph_dir)
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'unlabeled-contact-8A')
    
    @property
    def raw_file_names(self):
        return list(map(lambda x: f'{x}.json', self.pdbids))

    @property
    def processed_file_names(self):
        return list(map(lambda x: '{:06d}.pt'.format(x), range(len(self.pdbids))))

    def download(self):
        pass

    def process(self):
        for i, pdbid in enumerate(tqdm(self.pdbids, desc=self.__class__.__name__)):
            
            if 'demo' in self.kwargs and 'demo_num' in self.kwargs:
                if self.kwargs['demo'] and i >= self.kwargs['demo_num']:
                    break
            
            with open(f'{self.graph_dir}/{pdbid}.json') as f:
                json_graph = json.load(f)
            G = nx.readwrite.json_graph.node_link_graph(json_graph)
            
            data = from_networkx(G)

            # delete correlation edge (edge weight == -1)

            indices = torch.nonzero(torch.where(data.weight == 1, 1, 0)).squeeze()
            
            data.edge_index = torch.index_select(data.edge_index, 1, indices)
            
            # ========
            # features
            # ========
            
            uniques, counts = np.unique(list(self.encode_table.values()), return_counts=True)
            
            x = np.zeros((len(data.resname), len(uniques)))
            
            for j, residue in enumerate(data.resname):
                if residue not in self.encode_table:
                    residue = 'XAA'
                x[j, self.encode_table[residue]] = 1
                
            data.x = torch.from_numpy(x).int()

            pi = np.load(f'{self.pi_dir}/{pdbid}.npy')

            data.pi = torch.from_numpy(pi).float()

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # add necessary data

            data.resids = torch.from_numpy(np.array(G.nodes)).int()

            # remove unnecessary data

            del data.resname, data.weight

            torch.save(data, os.path.join(self.processed_dir, '{:06d}.pt'.format(i)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, '{:06d}.pt'.format(idx)))


class UnlabeledContactCorr8A(Dataset):
    def __init__(
        self, root, data_dir, pdbids, encode_table,
        transform=None, pre_transform=None, pre_filter=None, **kwargs):
        
        self.graph_dir = os.path.join(data_dir, 'unlabeled-8A')
        self.pi_dir = os.path.join(data_dir, 'pis')
        self.pdbids = sorted(pdbids)
        self.encode_table = encode_table
        self.kwargs = kwargs
        super(UnlabeledContactCorr8A, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.graph_dir)
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'unlabeled-contact-corr-8A')
    
    @property
    def raw_file_names(self):
        return list(map(lambda x: f'{x}.json', self.pdbids))

    @property
    def processed_file_names(self):
        return list(map(lambda x: '{:06d}.pt'.format(x), range(len(self.pdbids))))

    def download(self):
        pass

    def process(self):
        for i, pdbid in enumerate(tqdm(self.pdbids, desc=self.__class__.__name__)):
            
            if 'demo' in self.kwargs and 'demo_num' in self.kwargs:
                if self.kwargs['demo'] and i >= self.kwargs['demo_num']:
                    break
            
            with open(f'{self.graph_dir}/{pdbid}.json') as f:
                json_graph = json.load(f)
            G = nx.readwrite.json_graph.node_link_graph(json_graph)
            
            data = from_networkx(G)
            
            # ========
            # features
            # ========
            
            uniques, counts = np.unique(list(self.encode_table.values()), return_counts=True)
            
            x = np.zeros((len(data.resname), len(uniques)))
            
            for j, residue in enumerate(data.resname):
                if residue not in self.encode_table:
                    residue = 'XAA'
                x[j, self.encode_table[residue]] = 1
                
            data.x = torch.from_numpy(x).int()

            pi = np.load(f'{self.pi_dir}/{pdbid}.npy')

            data.pi = torch.from_numpy(pi).float()

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # add necessary data

            data.resids = torch.from_numpy(np.array(G.nodes)).int()

            # remove unnecessary objects

            del data.resname, data.weight

            torch.save(data, os.path.join(self.processed_dir, '{:06d}.pt'.format(i)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, '{:06d}.pt'.format(idx)))

class UnlabeledContact12A(Dataset):
    def __init__(
        self, root, data_dir, pdbids, encode_table,
        transform=None, pre_transform=None, pre_filter=None, **kwargs):
        
        self.graph_dir = os.path.join(data_dir, 'unlabeled-12A')
        self.pi_dir = os.path.join(data_dir, 'pis')
        self.pdbids = sorted(pdbids)
        self.encode_table = encode_table
        self.kwargs = kwargs
        super(UnlabeledContact12A, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.graph_dir)
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'unlabeled-contact-12A')
    
    @property
    def raw_file_names(self):
        return list(map(lambda x: f'{x}.json', self.pdbids))

    @property
    def processed_file_names(self):
        return list(map(lambda x: '{:06d}.pt'.format(x), range(len(self.pdbids))))

    def download(self):
        pass

    def process(self):
        for i, pdbid in enumerate(tqdm(self.pdbids, desc=self.__class__.__name__)):
            
            if 'demo' in self.kwargs and 'demo_num' in self.kwargs:
                if self.kwargs['demo'] and i >= self.kwargs['demo_num']:
                    break
            
            with open(f'{self.graph_dir}/{pdbid}.json') as f:
                json_graph = json.load(f)
            G = nx.readwrite.json_graph.node_link_graph(json_graph)
            
            data = from_networkx(G)

            # delete correlation edge (edge weight == -1)

            indices = torch.nonzero(torch.where(data.weight == 1, 1, 0)).squeeze()
            
            data.edge_index = torch.index_select(data.edge_index, 1, indices)
            
            # ========
            # features
            # ========
            
            uniques, counts = np.unique(list(self.encode_table.values()), return_counts=True)
            
            x = np.zeros((len(data.resname), len(uniques)))
            
            for j, residue in enumerate(data.resname):
                if residue not in self.encode_table:
                    residue = 'XAA'
                x[j, self.encode_table[residue]] = 1
                
            data.x = torch.from_numpy(x).int()

            pi = np.load(f'{self.pi_dir}/{pdbid}.npy')

            data.pi = torch.from_numpy(pi).float()

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # add necessary data

            data.resids = torch.from_numpy(np.array(G.nodes)).int()

            # remove unnecessary data

            del data.resname, data.weight

            torch.save(data, os.path.join(self.processed_dir, '{:06d}.pt'.format(i)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, '{:06d}.pt'.format(idx)))


class UnlabeledContactCorr12A(Dataset):
    def __init__(
        self, root, data_dir, pdbids, encode_table,  
        transform=None, pre_transform=None, pre_filter=None, **kwargs):
        
        self.graph_dir = os.path.join(data_dir, 'unlabeled-12A')
        self.pi_dir = os.path.join(data_dir, 'pis')
        self.pdbids = sorted(pdbids)
        self.encode_table = encode_table
        self.kwargs = kwargs
        super(UnlabeledContactCorr12A, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.graph_dir)
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'unlabeled-contact-corr-12A')
    
    @property
    def raw_file_names(self):
        return list(map(lambda x: f'{x}.json', self.pdbids))

    @property
    def processed_file_names(self):
        return list(map(lambda x: '{:06d}.pt'.format(x), range(len(self.pdbids))))

    def download(self):
        pass

    def process(self):
        for i, pdbid in enumerate(tqdm(self.pdbids, desc=self.__class__.__name__)):
            
            if 'demo' in self.kwargs and 'demo_num' in self.kwargs:
                if self.kwargs['demo'] and i >= self.kwargs['demo_num']:
                    break
            
            with open(f'{self.graph_dir}/{pdbid}.json') as f:
                json_graph = json.load(f)
            G = nx.readwrite.json_graph.node_link_graph(json_graph)
            
            data = from_networkx(G)
            
            # ========
            # features
            # ========
            
            uniques, counts = np.unique(list(self.encode_table.values()), return_counts=True)
            
            x = np.zeros((len(data.resname), len(uniques)))
            
            for j, residue in enumerate(data.resname):
                if residue not in self.encode_table:
                    residue = 'XAA'
                x[j, self.encode_table[residue]] = 1
                
            data.x = torch.from_numpy(x).int()

            pi = np.load(f'{self.pi_dir}/{pdbid}.npy')

            data.pi = torch.from_numpy(pi).float()

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # add necessary data

            data.resids = torch.from_numpy(np.array(G.nodes)).int()

            # remove unnecessary objects

            del data.resname, data.weight

            torch.save(data, os.path.join(self.processed_dir, '{:06d}.pt'.format(i)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, '{:06d}.pt'.format(idx)))


