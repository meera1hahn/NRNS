from math import pi as PI
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class AttentionConv(MessagePassing):
    def __init__(
        self,
        node_dim,
        edge_dim=16,
        hidden_dim=256,
        num_heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0,
        **kwargs
    ):
        super(AttentionConv, self).__init__(aggr="add")

        self.node_feat_size = node_dim
        self.edge_feat_size = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.concat = concat
        self.negative_slope = negative_slope

        self.att = nn.Sequential(
            nn.Linear(
                2 * self.node_feat_size + self.edge_feat_size,
                self.hidden_dim * self.num_heads,
            ),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(self.hidden_dim * self.num_heads, self.num_heads),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.value = nn.Linear(
            self.node_feat_size + self.edge_feat_size,
            self.hidden_dim * self.num_heads,
        )

        if self.concat is True:
            self.fc_post = nn.Linear(
                self.num_heads * self.hidden_dim, self.node_feat_size
            )
        else:
            self.fc_post = nn.Linear(self.hidden_dim, self.node_feat_size)

    def forward(self, x, edge_index, edge_attr):
        """
        Arguments:
            x has shape [num_nodes, node_feat_size]
            edge_index has shape [2, num_edges]
            edge_attr is [num_edges, edge_feat_size]
        """

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, edge_index_i, size_i):
        """
        Arguments:
            x_i has shape [num_edges, node_feat_size]
            x_j has shape [num_edges, node_feat_size]
            edge_attr has shape [num_edges, edge_feat_size]

        Returns:
            tensor of shape [num_edges, num_heads, hidden_dim]
        """

        alpha = self.att(torch.cat([x_i, x_j, edge_attr], dim=-1))
        alpha = softmax(alpha, index=edge_index_i, num_nodes=size_i)
        alpha = self.dropout(alpha).view(-1, self.num_heads, 1)

        value = self.value(torch.cat([x_j, edge_attr], dim=-1)).view(
            -1, self.num_heads, self.hidden_dim
        )
        return (value * alpha).squeeze()

    def update(self, aggr_out, x):
        """
        Arguments:
            aggr_out has shape [num_nodes, num_heads, hidden_dim]
                This is the result of aggregating features output by the
                `message` function from neighboring nodes. The aggregation
                function is specified in the constructor (`add` in this case).
            x has shape [num_nodes, node_feat_size]

        Returns:
            tensor of shape [num_nodes, node_feat_size]
        """
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.num_heads * self.hidden_dim)
        else:
            aggr_out = aggr_out.mean(dim=1)

        aggr_out = nn.ReLU()(x + self.fc_post(aggr_out))

        return aggr_out


from typing import Union, Tuple
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d


class CGConv(MessagePassing):
    r"""The crystal graph convolutional operator from the
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`_
    paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)}
        \sigma \left( \mathbf{z}_{i,j} \mathbf{W}_f + \mathbf{b}_f \right)
        \odot g \left( \mathbf{z}_{i,j} \mathbf{W}_s + \mathbf{b}_s  \right)
    where :math:`\mathbf{z}_{i,j} = [ \mathbf{x}_i, \mathbf{x}_j,
    \mathbf{e}_{i,j} ]` denotes the concatenation of central node features,
    neighboring node features and edge features.
    In addition, :math:`\sigma` and :math:`g` denote the sigmoid and softplus
    functions, respectively.
    Args:
        channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        dim (int, optional): Edge feature dimensionality. (default: :obj:`0`)
        aggr (string, optional): The aggregation operator to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        batch_norm (bool, optional): If set to :obj:`True`, will make use of
            batch normalization. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        channels: Union[int, Tuple[int, int]],
        dim: int = 0,
        aggr: str = "add",
        batch_norm: bool = False,
        bias: bool = True,
        **kwargs
    ):
        super(CGConv, self).__init__(aggr=aggr, **kwargs)
        self.channels = channels
        self.dim = dim
        self.batch_norm = batch_norm

        if isinstance(channels, int):
            channels = (channels, channels)

        self.lin_f = Linear(sum(channels) + dim, channels[1], bias=bias)
        self.lin_s = Linear(sum(channels) + dim, channels[1], bias=bias)
        self.bn = BatchNorm1d(channels[1])

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        self.bn.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.bn(out) if self.batch_norm else out
        out += x[1]
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        if edge_attr is None:
            z = torch.cat([x_i, x_j], dim=-1)
        else:
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))

    def __repr__(self):
        return "{}({}, dim={})".format(self.__class__.__name__, self.channels, self.dim)