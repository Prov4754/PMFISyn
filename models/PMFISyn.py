import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_scatter import scatter_sum, scatter_mean

def create_mlp(input_dim, layers, dropout_rate, output_dim, batch_norm=False):
    mlp_layers = []
    for i, layer_dim in enumerate(layers):
        if i > 0:
            mlp_layers.append(nn.Dropout(dropout_rate))
        mlp_layers.append(nn.Linear(input_dim if i == 0 else layers[i-1], layer_dim))
        if batch_norm:
            mlp_layers.append(nn.BatchNorm1d(layer_dim))
        mlp_layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
    mlp_layers.append(nn.Linear(layers[-1], output_dim))
    return nn.Sequential(*mlp_layers)

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

class FingerprintMLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=128, dropout=0.3):
        super(FingerprintMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x expected shape: (B, input_dim)
        return self.mlp(x)

class DrugGraphFEM(nn.Module):
    def __init__(self, num_features, gat_dims, output_dim, dropout, heads=1):
        super(DrugGraphFEM, self).__init__()
        self.conv_layers = nn.ModuleList()
        for i, dim in enumerate(gat_dims):
            input_dim = num_features if i == 0 else gat_dims[i - 1] * heads
            self.conv_layers.append(
                GATConv(input_dim, dim, heads=heads, dropout=dropout)
            )
        self.sag_pool = NodeLevelSAGPooling(gat_dims[-1] * heads, ratio=1.0)

        self.fc_g = create_mlp(gat_dims[-1] * heads, [output_dim], dropout, output_dim, batch_norm=True)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x, edge_index, batch):
        for conv in self.conv_layers:
            x = self.leaky_relu(conv(x, edge_index))

        x, attention_scores = self.sag_pool(x, edge_index, batch)

        x = self.fc_g(x)
        return x

class PMFISyn(nn.Module):
    def __init__(self, n_output=2, gat_dims=(32, 64, 128),
                num_features_xd=78, num_features_xt=954, fingerprint_dim=2048, output_dim=128,
                dropout=0.1, num_mpgr_layers=2, heads=1):
        super(PMFISyn, self).__init__()

        self.drug1_fem = DrugGraphFEM(num_features_xd, gat_dims, output_dim, dropout, heads)
        self.drug2_fem = DrugGraphFEM(num_features_xd, gat_dims, output_dim, dropout, heads)
        self.fingerprint_encoder = FingerprintMLP(input_dim=fingerprint_dim, hidden_dim=1024, output_dim=output_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.cell_fem = create_mlp(num_features_xt, [output_dim * 2, output_dim], dropout, output_dim, batch_norm=True)
        self.mpgr = MPGR(num_mpgr_layers, output_dim, dropout)
        self.se_weight = nn.Sequential(
            nn.Linear(5, 2),
            nn.ReLU(inplace=True),
            nn.Linear(2, 5),
            nn.Sigmoid()
        )
        self.gated_proj = nn.Sequential(
            nn.Linear(5 * output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.synergy_pm = nn.Sequential(
            nn.Linear(2 * output_dim, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(128, n_output)
        )
        self.apply(weights_init)

    def forward(self, data1, data2):
        x1_graph = self.drug1_fem(data1.x, data1.edge_index, data1.batch)  # (B, D)
        x2_graph = self.drug2_fem(data2.x, data2.edge_index, data2.batch)  # (B, D)

        batch_size = len(torch.unique(data1.batch))
        x1_fp = data1.fingerprint.view(batch_size, -1)
        x2_fp = data2.fingerprint.view(batch_size, -1)
        x1_fp = self.fingerprint_encoder(x1_fp)  # (B, D)
        x2_fp = self.fingerprint_encoder(x2_fp)  # (B, D)

        cell_vector = F.normalize(data1.cell, p=2, dim=1)
        cell_vector = self.cell_fem(cell_vector)  # (B, D)

        mpgr_out = self.mpgr(x1_graph, x2_graph, x1_fp, x2_fp, cell_vector)
        f_drug1_graph = mpgr_out['drug1_graph']
        f_drug2_graph = mpgr_out['drug2_graph']
        f_drug1_fp    = mpgr_out['drug1_fp']
        f_drug2_fp    = mpgr_out['drug2_fp']
        f_cell        = mpgr_out['cell']

        gated_concat = torch.cat([f_drug1_graph, f_drug2_graph, f_drug1_fp, f_drug2_fp, f_cell], dim=1)  # (B, 5*D)
        gated_proj_out = self.gated_proj(gated_concat)  # (B, D)

        features = torch.stack([x1_graph, x2_graph, x1_fp, x2_fp, cell_vector], dim=1)  # (B,5,D)
        weights = self.se_weight(features.mean(dim=-1))  # (B,5)
        weights = weights.unsqueeze(-1)  # (B,5,1)
        se_fused = (features * weights).sum(dim=1)  # (B, D)

        dual = torch.cat([gated_proj_out, se_fused], dim=1)  # (B, 2*D)

        out = self.synergy_pm(dual)  # (B, n_output)
        return out

class NodeLevelSAGPooling(nn.Module):
    def __init__(self, in_channels, ratio=1.0, nonlinearity='tanh'):
        super(NodeLevelSAGPooling, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio

    def forward(self, x, edge_index, batch):
        x_pooled = global_mean_pool(x, batch)  # (num_graphs, in_channels)
        attention_scores = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        unique_batches, counts = torch.unique(batch, return_counts=True)
        count_map = {int(b.item()): float(c.item()) for b, c in zip(unique_batches, counts)}
        for graph_id in unique_batches:
            mask = batch == graph_id
            attention_scores[mask] = 1.0 / count_map[int(graph_id.item())]
        return x_pooled, attention_scores

class MPGR(nn.Module):
    def __init__(self, num_layers, input_size, dropout_rate):
        super(MPGR, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        sources = ['drug1_graph', 'drug2_graph', 'drug1_fp', 'drug2_fp', 'cell']
        self.gate_modules = nn.ModuleDict()
        self.non_linear_modules = nn.ModuleDict()
        self.linear_modules = nn.ModuleDict()

        for source in sources:
            gate_layers = []
            non_linear_layers = []
            linear_layers = []
            for _ in range(num_layers):
                gate_layers.append(nn.Linear(input_size, input_size))
                non_linear_layers.append(nn.Linear(input_size, input_size))
                linear_layers.append(nn.Linear(input_size, input_size))
            self.gate_modules[source] = nn.ModuleList(gate_layers)
            self.non_linear_modules[source] = nn.ModuleList(non_linear_layers)
            self.linear_modules[source] = nn.ModuleList(linear_layers)

        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def gate_process(self, x, source, layer_idx):
        gate = torch.sigmoid(self.gate_modules[source][layer_idx](x))
        non_linear = self.leaky_relu(self.non_linear_modules[source][layer_idx](x))
        linear = self.linear_modules[source][layer_idx](x)
        output = gate * non_linear + (1 - gate) * linear
        output = output + x
        return self.dropout(output)

    def forward(self, x_drug1_graph, x_drug2_graph, x_drug1_fp, x_drug2_fp, x_cell):
        for layer_idx in range(self.num_layers):
            x_drug1_graph = self.gate_process(x_drug1_graph, 'drug1_graph', layer_idx)
            x_drug2_graph = self.gate_process(x_drug2_graph, 'drug2_graph', layer_idx)
            x_drug1_fp    = self.gate_process(x_drug1_fp,    'drug1_fp',    layer_idx)
            x_drug2_fp    = self.gate_process(x_drug2_fp,    'drug2_fp',    layer_idx)
            x_cell        = self.gate_process(x_cell,        'cell',        layer_idx)
        return {
            'drug1_graph': x_drug1_graph,
            'drug2_graph': x_drug2_graph,
            'drug1_fp':    x_drug1_fp,
            'drug2_fp':    x_drug2_fp,
            'cell':        x_cell
        }