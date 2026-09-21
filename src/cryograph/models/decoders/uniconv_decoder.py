import torch
import torch.nn as nn
import torch_geometric as pyg
from jaxtyping import Float
from cryograph.models.decoders import Decoder
from cryograph.models.layers import OrthogonalGCNConvMessage

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import typing
from typing import Callable, Optional, Union, Tuple, List
from torch import Tensor
from torch.nn import Parameter
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn.inits import reset, glorot, zeros
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
    to_undirected,
)

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros

from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    Size,
    torch_sparse,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value

class GroupSort(nn.Module):
    """Norm-preserving activation sorting each pair of channels. Unused."""
    def forward(self, x):
        a, b = x.split(x.size(-1) // 2, 1)
        a, b = torch.max(a, b), torch.min(a, b)
        return torch.cat([a, b], dim=-1)

def block_diagonal_init(weight_matrix, block_size=2, bound=0.5):
    """Return a matrix of Gaussian blocks along the diagonal and zeros elsewhere."""
    # Get the size of the weight matrix
    n = weight_matrix.size(0)

    weight_matrix = torch.zeros_like(weight_matrix)
    
    # Calculate the number of blocks along one dimension
    num_blocks = (n + block_size - 1) // block_size
    
    for i in range(num_blocks):
        # Calculate the actual block size for the current block
        actual_block_size = min(block_size, n - i * block_size)
        
        # Initialize with Gaussian noise
        part = torch.randn(actual_block_size, actual_block_size) * bound
                
        # Place the block on the diagonal of the weight matrix
        start_row = i * block_size
        end_row = start_row + actual_block_size
        weight_matrix[start_row:end_row, start_row:end_row] = part
    
    return weight_matrix


class UniConvDecoder(Decoder):
    """Decoder passing messages through orthogonal graph convolutions, which
    propagate features by a truncated matrix exponential.
    """
    def __init__(
        self,
        protein_graph: pyg.data.Data,
        latent_dim: int,
        feature_dim: int,
        layer_specs: list,
        embed_layers: int,
        act_name: str,
        dropout: float,
    ):
        super().__init__(protein_graph, latent_dim, act_name, dropout)
        # num_edges = protein_graph.num_edges
        self.register_buffer(
            "edge_index", 
            protein_graph['edge_index'],
        )
        
        self.embed_list = nn.ModuleList([])
        for n in range(embed_layers):
            self.embed_list.append(nn.Linear(latent_dim, latent_dim))

        self.embed_final = nn.Linear(latent_dim, self.num_residues*feature_dim)
        self.unflatten = nn.Unflatten(1, (self.num_residues, feature_dim))
        
        self.messpass_list = nn.ModuleList([])
        node_channels = feature_dim
        for i in range(len(layer_specs)): 
            spec = layer_specs[i]
            self.messpass_list.append(
                OrthogonalGCNConvMessage(
                    dim_in=node_channels,
                    dim_out=spec['node_out_channels'],
                    residual=spec['residual'],
                    global_bias=spec['global_bias'],
                    T=spec['T'],
                    use_hermitian=spec['use_hermitian'],
                    act=self.act,
                )
            )
            node_channels = spec['node_out_channels']
        self.unbed = nn.Linear(node_channels, 3)

        self.reset_parameters()
    
    def forward(
        self,
        latents: Float[Tensor, "B latent_dim"]
    ) -> Float[Tensor, "B num_nodes 3"]:
        
        z = latents
            
        for layer in self.embed_list:
            z = layer(z)
            z = self.act(z)
        
        h = self.embed_final(z)
        h = self.unflatten(h)

        for n in range(len(self.messpass_list)):
            h = self.messpass_list[n](h, edge_index=self.edge_index)

        output = self.unbed(h) + self.base_coords

        return output

    def reset_parameters(self):
        for layer in self.embed_list:
            layer.reset_parameters()
        self.embed_final.reset_parameters()
        # for layer in self.messpass_list:
        #     layer.reset_parameters()
        self.unbed.reset_parameters()

