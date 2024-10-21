import os
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
import torch
import numpy as np
from torch_geometric.data import Data

# from torch.utils.data import DataLoader

from torch_geometric.loader.dataloader import DataLoader

# from dataloader import DataLoaderSubstructContext

from model import GNN
from loader import CustomDataset

# from torchsummary import summary


def raw_graph_to_data_obj(edge_list):
    """
    edge_list: list of tuples, each tuple is (src, src_label, dst, dst_label)
    edge_label = 1 for all edges
    """

    # nodes
    node_dict = {}

    for edge in edge_list:
        src, src_label, dst, dst_label = edge
        if node_dict.get(src) is None:
            node_dict[src] = src_label
        if node_dict.get(dst) is None:
            node_dict[dst] = dst_label

    node_list = list(node_dict.keys())
    x = torch.tensor(np.array(node_list), dtype=torch.long)

    # edges
    edges_list = []
    edge_features_list = []
    for edge in edge_list:
        src, src_label, dst, dst_label = edge
        i = src
        j = dst
        edge_feature = [1]  # edge label
        edges_list.append((i, j))
        edge_features_list.append(edge_feature)
        edges_list.append((j, i))
        edge_features_list.append(edge_feature)

    # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
    edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

    # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


edges_list = [
    (0, 0, 1, 1),
    (0, 0, 2, 2),
    (1, 1, 2, 2),
    (1, 1, 3, 3),
    (2, 2, 3, 3),
    (2, 2, 4, 4),
    (3, 3, 4, 4),
    (3, 3, 5, 5),
    (4, 4, 5, 5),
]


def main():
    dataset = CustomDataset(root="data")

    device = torch.device("cuda:0")

    model = GNN(num_layer=5, emb_dim=300).to(device)
    model.load_state_dict(torch.load("../chem/model_gin/contextpred.pth"))

    # loader = DataLoaderSubstructContext(dataset, batch_size=1, shuffle=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in loader:
        print(1, batch)
        y = model(batch)

    # print(dir(model))
    # summary(model)


if __name__ == "__main__":
    main()
