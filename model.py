from torch import nn
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
	def __init__(self, feature_size: int, num_classes: int):
		super().__init__()
		
		self.conv1 = GraphConv(in_feats=feature_size, out_feats=16, activation=nn.ReLU())
		self.dropout = nn.Dropout(p=0.5)
		self.conv2 = GraphConv(in_feats=16, out_feats=num_classes)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, graph, n_feats):
		
		h = self.conv1(graph, n_feats)
		x = self.dropout(h)
		x = self.conv2(graph, x)

		return h, self.softmax(x)
