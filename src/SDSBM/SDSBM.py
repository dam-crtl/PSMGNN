from torch_geometric_signed_directed.data.general import SDSBM
import numpy as np
import pandas as pd
import os

def create_dataset(N=5000, 
                   K=4, 
                   p=0.5, 
                   F=np.array([[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0, ]]), 
                   size_ratio=1, 
                   eta=0.5, 
                   path=''):
    graph, label = SDSBM(N=N, K=K, p=p, F=F, size_ratio=size_ratio, eta=eta)
    graph = graph.toarray()
    label = label.reshape(-1, 1)
    save_data = np.concatenate([graph, label], axis=1)
    df = pd.DataFrame(save_data)
    df.to_csv("data.csv", index=False, header=True)
    
    return

def save_and_output(args, results, path):
    df = pd.read_csv(path)
    log_path = os.path.join(path, "results.txt")
    score_name = ["accuracy", "f1 score", "macro f1 score", "micro f1 score", "neg_acc", "pos_acc", "time"]
    with open(log_path, 'w') as file:
        for i in range(args.num_result):
            col = results[:, i]
            sentence = "the average of {name}: {average:.4f}±{st:.4f}".format(name=score_name[i], average=np.mean(col), st=np.std(col))
            print(sentence)
            file.write(sentence)
            file.write("\n")
    return



if __name__=="__main__":
    create_dataset(
        N=2000,
        K=4,
        p=0.01,
        F=np.array(
            [
                [0, 1, 2, 0],
                [0, 0, 0, 1],
                [0, 2, 0, 0],
                [1, 0, 0, 0],
            ]
        ),
        size_ratio=1,
        eta=0.4,
        path=""
    )
