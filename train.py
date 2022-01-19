import torch
import torchmetrics.functional as metrics
from dgl.data import CoraGraphDataset, KarateClubDataset
from model import GCN

if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	epochs = 30
	dataset = KarateClubDataset()
	
	graph = dataset[0]

	train_idx = torch.arange(0, 28)
	test_idx = torch.arange(28, 34)

	model = GCN(feature_size=34, num_classes=2)
	criterion = torch.nn.NLLLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)

	model.to(device)
	graph.to(device)

	for epoch in range(epochs):
		model.train()
		optimizer.zero_grad()
		n_feats = torch.eye(34).to(device)
		labels = graph.ndata['label']

		_, logits = model(graph, n_feats)

		loss = criterion(logits[train_idx], labels[train_idx])

		loss.backward()
		optimizer.step()

		with torch.no_grad():
			model.eval()
			_, output = model(graph, n_feats)

			loss = criterion(output, labels)
			accuracy = metrics.accuracy(output, labels)

			print(f'epoch [{epoch+1}/{epochs}]\t Loss: {loss.item():.4f}\t Accuracy: {accuracy*100:.2f} %')
	
	torch.save(model.state_dict(), 'pretrained/gcn-model.pt')
