from json import load
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from dgl.data import CoraGraphDataset, KarateClub

def load_karate_club():
	dataset = KarateClub()

	graph = dataset[0]
	labels = graph.ndata['label'].numpy()

	return graph.to_networkx(), labels


selected = st.sidebar.selectbox("Navigation", [
	'Model prediction', 'Graph Visualization', 'Graph Stats'
])

if selected == 'Model prediction':
	st.warning(selected)
elif selected == 'Graph Visualization':
	graph, labels = load_karate_club()

	nodes = []
	edges = []
	node_colors = ['#DBEBC2', '#F48B94']
	
	for index, node in enumerate(graph.nodes(False)):
		nodes.append(Node(str(node), size=450, label=str(node), labelPosition="center", color=node_colors[labels[index]]))

	for (src, dst) in graph.edges():
		edges.append(Edge(source=str(src), target=str(dst), type="STRAIGHT"))

	config = Config(width=1000, 
                height=800, 
                directed=False,
                nodeHighlightBehavior=True, 
                highlightColor="#F7A7A6",
                collapsible=True,
                node={'labelProperty':'label'},
                link={'labelProperty': 'label', 'renderLabel': True}
                ) 

	return_value = agraph(nodes=nodes, 
                      edges=edges, 
                      config=config)

elif selected == 'Graph Stats':
	st.success(selected)


