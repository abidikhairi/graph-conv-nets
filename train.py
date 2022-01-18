import torch
import torchmetrics.functional as metrics
from dgl.data import CoraGraphDataset
from model import GCN

if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	epochs = 100
	dataset = CoraGraphDataset()
	
	graph = dataset[0]

	train_idx = torch.nonzero(graph.ndata['train_mask'], as_tuple=False).flatten()
	test_idx = torch.nonzero(graph.ndata['test_mask'], as_tuple=False).flatten()

	model = GCN(feature_size=1433, num_classes=7)
	criterion = torch.nn.NLLLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)

	model.to(device)
	graph.to(device)

	for epoch in range(epochs):
		model.train()
		optimizer.zero_grad()
		n_feats = graph.ndata['feat']
		labels = graph.ndata['label']

		logits = model(graph, n_feats)

		loss = criterion(logits[train_idx], labels[train_idx])

		loss.backward()
		optimizer.step()

		with torch.no_grad():
			model.eval()
			output = model(graph, n_feats)

			loss = criterion(output, labels)
			accuracy = metrics.accuracy(output, labels)

			print(f'epoch [{epoch+1}/{epochs}]\t Loss: {loss.item():.4f}\t Accuracy: {accuracy*100:.2f} %')
	
	torch.save(model.state_dict(), 'pretrained/gcn-model.pt')
