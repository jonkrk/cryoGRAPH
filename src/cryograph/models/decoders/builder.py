from typing import Dict, Any, Iterable, List
from torch_geometric.data import Data
from . import (
    MLPDecoder, GNNDecoder, EGCNDecoder, GNNEdgeDecoder, GINEConvDecoder,
    GCNConvDecoder, UniConvDecoder, SAGEConvDecoder, GraphConvDecoder
)

def _expand(
    specs: Iterable[Any], 
    repeat: int
) -> List[Any]:
    """Repeat a single layer specification repeat times."""
    specs = list(specs)
    if repeat <= 1:
        return specs
    if len(specs) != 1:
        raise ValueError("When 'repeat' > 1, 'layer_specs' must have length 1.")
    return specs * repeat

def build_decoder(
    cfg: Dict[str, Any], 
    protein_graph: Data):
    """Construct the decoder named by network.decoder.model."""
    dec = cfg['network']['decoder']
    latent_dim = cfg['network']['latent_dim']
    model = dec['model']
    act = dec['act_name']
    drop = dec['dropout']

    if model == "mlp":
        sc = dec['mlp']
        return MLPDecoder(
            protein_graph=protein_graph,
            latent_dim=latent_dim,
            num_layers=sc['num_layers'],
            layer_width=sc['layer_width'],
            act_name=act, 
            dropout=drop,
        )

    if model == "gnn":
        sc = dec['gnn']
        return GNNDecoder(
            protein_graph=protein_graph,
            latent_dim=latent_dim,
            feature_dim=sc['feature_dim'],
            embed_layers=sc['embed_layers'],
            messpass_layers=sc['messpass_layers'],
            aggreg_sublayers=sc['aggreg_sublayers'],
            message_sublayers=sc['message_sublayers'],
            act_name=act, 
            dropout=drop,
        )

    if model == "egcn":
        sc = dec['egcn']
        return EGCNDecoder(
            protein_graph=protein_graph,
            latent_dim=latent_dim,
            feature_dim=sc['feature_dim'],
            embed_layers=sc['embed_layers'],
            layer_specs=_expand(sc['layer_specs'], sc['repeat']),
            act_name=act, 
            dropout=drop,
        )

    if model == "edge_gnn":
        sc = dec['edge_gnn']
        return GNNEdgeDecoder(
            protein_graph=protein_graph,
            latent_dim=latent_dim,
            feature_dim=sc['feature_dim'],
            embed_layers=sc['embed_layers'],
            edge_attribute_channels=sc['edge_attribute_channels'],
            layer_specs=_expand(sc['layer_specs'], sc['repeat']),
            act_name=act, 
            dropout=drop,
        )

    if model == "gineconv":
        sc = dec['gineconv']
        return GINEConvDecoder(
            protein_graph=protein_graph,
            latent_dim=latent_dim,
            feature_dim=sc['feature_dim'],
            embed_layers=sc['embed_layers'],
            edge_attribute_channels=sc['edge_attribute_channels'],
            layer_specs=_expand(sc['layer_specs'], sc['repeat']),
            act_name=act, 
            dropout=drop,
        )

    if model == "gcnconv":
        sc = dec['gcnconv']
        return GCNConvDecoder(
            protein_graph=protein_graph,
            latent_dim=latent_dim,
            feature_dim=sc['feature_dim'],
            embed_layers=sc['embed_layers'],
            layer_specs=_expand(sc['layer_specs'], sc['repeat']),
            act_name=act, 
            dropout=drop,
        )

    if model == "uniconv":
        sc = dec['uniconv']
        return UniConvDecoder(
            protein_graph=protein_graph,
            latent_dim=latent_dim,
            feature_dim=sc['feature_dim'],
            embed_layers=sc['embed_layers'],
            layer_specs=_expand(sc['layer_specs'], sc['repeat']),
            act_name=act, 
            dropout=drop,
        )

    if model == "sageconv":
        sc = dec['sageconv']
        return SAGEConvDecoder(
            protein_graph=protein_graph,
            latent_dim=latent_dim,
            feature_dim=sc['feature_dim'],
            embed_layers=sc['embed_layers'],
            layer_specs=_expand(sc['layer_specs'], sc['repeat']),
            act_name=act, 
            dropout=drop,
        )

    if model == "graphconv":
        sc = dec['graphconv']
        return GraphConvDecoder(
            protein_graph=protein_graph,
            latent_dim=latent_dim,
            feature_dim=sc['feature_dim'],
            embed_layers=sc['embed_layers'],
            layer_specs=_expand(sc['layer_specs'], sc['repeat']),
            act_name=act, 
            dropout=drop,
        )

    raise ValueError(f"Unknown decoder model: {model!r}")
