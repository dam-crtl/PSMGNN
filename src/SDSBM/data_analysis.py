import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import networkx as nx


def data_analysis_and_save(adjacency_matrix, node_labels):

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
        "the number of positive edges: {num_negative_edges}".format(
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
    sentence.append(
        "the density: {density} %".format(
            density=density
        )
    )

    # calculate the number of antiparallel edges
    num_antiparallel_edges = sum(1 for u, v in G.edges() if G.has_edge(v, u))
    sentence.append(
        "the number of the antiparallel edges: {num_antiparallel_edges}".format(
            num_antiparallel_edges=num_antiparallel_edges
        )
    )
    unique_elements, counts = np.unique(node_labels, return_counts=True)
    
    with open("data_analysis.txt", "w") as file:
        for i in range(len(sentence)):
            print(sentence[i])
            file.write(sentence[i])
            file.write("\n")
        
        print("the number of each label\n")

        for element, count in zip(unique_elements, counts):
            s = "the number of the label {element}: {count}".format(element=element, count=count)
            print(s)
            file.write(s)
            file.write("\n")

    # plot the graph
    pos = nx.spring_layout(G)  # layout setting
    nx.draw_networkx(
        G,
        pos,
        with_labels=True,
        labels=labels,
        node_color="skyblue",
        node_size=100,
        arrows=True,
    )
    plt.title("Graph Visualization")
    plt.savefig("graph.png")


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    A = df.iloc[:, :-1].values
    label = df.iloc[:, -1].values
    data_analysis_and_save(A, label)
