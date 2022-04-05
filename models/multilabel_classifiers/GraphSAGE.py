import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.nn import SAGEConv, global_max_pool, LayerNorm

class GraphSAGE(nn.Module):
    def __init__(self, dim_features, dim_pers, dim_target, num_layers,
            dim_node_embedding, dim_graph_embedding, dim_pers_embedding,
            **kwargs):
        self.dim_features = dim_features
        self.dim_pers = dim_pers
        self.dim_target = dim_target
        self.num_layers = num_layers
        self.dim_node_embedding = dim_node_embedding
        self.dim_graph_embedding = dim_graph_embedding
        self.dim_pers_embedding = dim_pers_embedding
        self.aggregation = kwargs['aggregation'] if kwargs['aggregation'] is not None else 'mean'
        self.dropout = kwargs['dropout'] if kwargs['dropout'] is not None else 0.1
        
        super().__init__()

        self.layers = nn.ModuleList([])
        for i in range(self.num_layers):
            dim_input = self.dim_features if i == 0 else self.dim_node_embedding

            conv = SAGEConv(dim_input, self.dim_node_embedding)
            # Overwrite aggregation method (default is set to mean)
            conv.aggr = self.aggregation

            self.layers.append(conv)

        if self.aggregation == 'max':
            self.fc_max = nn.Linear(self.dim_node_embedding, self.dim_node_embedding)
            self.drop_max = nn.Dropout(p=self.dropout)
            self.ln_max = nn.LayerNorm(self.dim_node_embedding)

        self.fc1 = nn.Linear(self.num_layers * self.dim_node_embedding, self.dim_graph_embedding)
        self.drop1 = nn.Dropout(p=self.dropout)
        self.ln1 = nn.LayerNorm(self.dim_graph_embedding)

        if self.dim_pers_embedding != 0:
            self.fc_pers = nn.Linear(self.dim_pers, self.dim_pers_embedding)
            self.drop_pers = nn.Dropout(p=self.dropout)
            self.ln_pers = nn.LayerNorm(self.dim_pers_embedding)
            self.fc2 = nn.Linear(self.dim_graph_embedding + self.dim_pers_embedding, self.dim_target)  
        else:
            self.fc2 = nn.Linear(self.dim_graph_embedding, self.dim_target)

        self.drop2 = nn.Dropout(p=self.dropout)
        self.ln2 = nn.LayerNorm(self.dim_target)

    def forward(self, data):
        x, pi, edge_index, batch = data.x, data.pi, data.edge_index, data.batch
    
        x_all = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = F.relu(x)
            if self.aggregation == 'max':
                x = F.relu(self.drop_max(self.ln_max(self.fc_max(x))))
            x_all.append(x)

        # skip connection and global max pooling
        x = torch.cat(x_all, dim=1)
        x = global_max_pool(x, batch)

        g = F.relu(self.drop1(self.ln1(self.fc1(x))))             # graph embedding

        if self.dim_pers_embedding != 0:
            p = F.relu(self.drop_pers(self.ln_pers(self.fc_pers(pi))))    # persistence embedding
            out = self.drop2(self.ln2(self.fc2(torch.cat((g, p), dim=1))))
        else:
            out = self.drop2(self.ln2(self.fc2(g)))

        return out
