from torch_geometric_signed_directed.data.general import SDSBM
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

def create_dataset(N=2000, 
                   K=4, 
                   p=0.01, 
                   F=np.array([[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0, ]]), 
                   size_ratio=1, 
                   eta=0.5, 
                   data_name="data0.csv"):
    graph, label = SDSBM(N=N, K=K, p=p, F=F, size_ratio=size_ratio, eta=eta)
    graph = graph.toarray()
    label = label.reshape(-1, 1)
    save_data = np.concatenate([graph, label], axis=1)
    df = pd.DataFrame(save_data)
    df.to_csv(data_name, index=False, header=True)

    return


def data_analysis_and_save(adjacency_matrix, node_labels, index):

    sentence = []
    labels = {i: label for i, label in enumerate(node_labels)}
    G = nx.from_numpy_matrix(adjacency_matrix, create_using=nx.DiGraph)

    # calculate the number of nodes
    num_nodes = G.number_of_nodes()
    sentence.append("the number of nodes: {num_nodes}".format(num_nodes=num_nodes))

    # calculate the number of the edges
    num_edges = G.number_of_edges()
    sentence.append("the number of edges: {num_edges}".format(num_edges=num_edges))

    # calculate the number of the positive edges
    num_positive_edges = sum(
        1 for _, _, data in G.edges(data=True) if data.get("weight", 1) > 0
    )
    sentence.append(
        "the number of positive edges: {num_positive_edges}".format(
            num_positive_edges=num_positive_edges
        )
    )

    # calculate the number of the positive edges
    num_negative_edges = sum(
        1 for _, _, data in G.edges(data=True) if data.get("weight", 1) < 0
    )
    sentence.append(
        "the number of negative edges: {num_negative_edges}".format(
            num_negative_edges=num_negative_edges
        )
    )

    # calculate the ratio of the positive edges
    ratio_positive_edges = num_positive_edges / num_edges
    sentence.append(
        "the ratio of positive edges: {ratio_positive_edges}".format(
            ratio_positive_edges=ratio_positive_edges
        )
    )

    # calculate the density of the graph
    density = num_edges / (num_nodes * (num_nodes - 1))
    sentence.append("the density: {density} %".format(density=density))

    # calculate the number of antiparallel edges
    num_antiparallel_edges = sum(1 for u, v in G.edges() if G.has_edge(v, u))
    sentence.append(
        "the number of the antiparallel edges: {num_antiparallel_edges}".format(
            num_antiparallel_edges=num_antiparallel_edges
        )
    )
    unique_elements, counts = np.unique(node_labels, return_counts=True)

    with open("data_analysis" + str(index) + ".txt", "w") as file:
        for i in range(len(sentence)):
            print(sentence[i])
            file.write(sentence[i])
            file.write("\n")

        print("the number of each label\n")

        for element, count in zip(unique_elements, counts):
            s = "the number of the label {element}: {count}".format(
                element=element, count=count
            )
            print(s)
            file.write(s)
            file.write("\n")

    return

if __name__=="__main__":
    # N = [2000, 2000, 2000, 2000]
    K = [4, 5]
    # p = [0.01, 0.01, 0.01, 0.01]
    F = [
        np.array(
            [
                [0, 1, 2, 0],
                [0, 0, 0, 1],
                [0, 2, 0, 0],
                [1, 0, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 1, 2, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 2],
                [1, 0, 0, 0, 0],
            ]
        )
    ]
    # size_ratio = [1, 1, 1, 1]
    eta = [0.3, 0.5, 0.3, 0.5]
    
    for i in range(4):
        data_name = "data" + str(i) + ".csv"
        id2 = i // 2
        
        create_dataset(
            K=K[id2],
            F=F[id2],
            eta=eta[i],
            data_name=data_name
        )
        
        df = pd.read_csv(data_name)
        A = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        data_analysis_and_save(adjacency_matrix=A, node_labels=y, index = i)
        
        
