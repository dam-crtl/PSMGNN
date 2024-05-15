import os
import sys
import time
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import pickle as pk
from sklearn import metrics
from scipy.sparse import csr_matrix
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric_signed_directed import node_class_split
from torch_geometric_signed_directed.data.signed import (
    load_signed_real_data,
    SignedData,
)
from torch_geometric_signed_directed.data.directed import load_directed_real_data
from torch_geometric_signed_directed.utils import (
    node_class_split,
    in_out_degree,
    extract_network,
)
from torch_geometric_signed_directed.nn.signed import SGCN, SDGNN, SiGAT, SNEA

# laplacians
from SigMaNet.Signum import SigMaNet_node_prediction_one_laplacian
from PSM_SigMaNet.Signum import PSM_SigMaNet_node_prediction_one_laplacian
from torch_geometric_signed_directed.nn.general.MSGNN import MSGNN_node_classification
from PSM_MSGNN.PSM_MSGNN import PSM_MSGNN_node_classification
# from MSGNN.MSGNN import PSM_MSGNN_node_classification
from SigMaNet import laplacian
from PSM_SigMaNet import new_laplacian

# save data
from utils.save_settings import write_log


# select cuda device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="link prediction of SigNum")
    parser.add_argument(
        "--log_root",
        type=str,
        default="../logs/",
        help="the path saving model.t7 and the training process",
    )
    parser.add_argument(
        "--dataset", type=str, default="sdsbm", help="data set selection"
    )
    parser.add_argument(
        "--split_prob",
        type=lambda s: [float(item) for item in s.split(",")],
        default="0.05,0.15",
        help="random drop for testing/validation/training edges (for 3-class classification only)",
    )
    parser.add_argument("--task", type=str, default="node_classificaion", help="Task")
    parser.add_argument("--epochs", type=int, default=1500, help="training epochs")
    parser.add_argument("--num_filter", type=int, default=64, help="num of filters")
    parser.add_argument("--method", type=str, default="PSM_MSGNN", help="method name")
    parser.add_argument("--K", type=int, default=2, help="K for cheb series")
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="how many layers of gcn in the model, only 1 or 2 layers.",
    )
    parser.add_argument("--netflow", "-N", action="store_true", help="if use net flow")
    parser.add_argument(
        "--follow_math", "-F", action="store_true", help="if follow math"
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout prob")
    parser.add_argument("--debug", "-D", action="store_true", help="debug mode")
    parser.add_argument(
        "--num_class_node",
        type=int,
        default=4,
        help="number of classes for node classification.",
    )
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
    parser.add_argument("--l2", type=float, default=5e-4, help="l2 regularizer")
    parser.add_argument("--noisy", action="store_true")
    parser.add_argument(
        "--randomseed", type=int, default=0, help="if set random seed in training"
    )

    parser.add_argument("--trainable_q", action="store_true")
    parser.add_argument("--runs", type=int, default=10, help="number of distinct runs")
    parser.add_argument(
        "--num_result", type=int, default=7, help="number of result values"
    )
    parser.add_argument(
        "--sd_input_feat",
        type=bool,
        default=True,
        help="Whether to use both signed and directed features as input.",
    )
    parser.add_argument(
        "--num_input",
        type=int,
        default=4,
        help="The number of input.",
    )
    parser.add_argument('--normalization', type=str, default='sym')
    parser.add_argument('--sdsbm_number', type=int, default=-1, help="the index of the SDSBM graph for experiments")
    return parser.parse_args()


def neg_and_pos_acc(pred, label):
    np_pred = np.array(pred.cpu())
    np_label = np.array(label.cpu())
    neg_label_index = np.where(np_label == 0)[0]
    pos_label_index = np.where(np_label == 1)[0]
    neg_acc = np.sum(np_pred[neg_label_index] == 0) / len(neg_label_index)
    pos_acc = np.sum(np_pred[pos_label_index] == 1) / len(pos_label_index)
    return neg_acc, pos_acc


def scores(num_class_node, pred, label):
    pred_cpu = pred.cpu()
    label_cpu = label.cpu()
    accuracy = metrics.accuracy_score(pred_cpu, label_cpu)
    if num_class_node == 2:
        f1 = metrics.f1_score(pred_cpu, label_cpu)
    else:
        f1 = 0
    f1_macro = metrics.f1_score(pred_cpu, label_cpu, average="macro")
    f1_micro = metrics.f1_score(pred_cpu, label_cpu, average="micro")
    return accuracy, f1, f1_macro, f1_micro


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.cofollow_mathl)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def main(args):
    global_start_time = time.time()
    random.seed(args.randomseed)
    torch.manual_seed(args.randomseed)
    np.random.seed(args.randomseed)

    date_time = datetime.now().strftime("%m-%d-%H:%M:%S")
    log_path = os.path.join(args.log_root, args.method, args.dataset + str(args.sdsbm_number), args.save_name, date_time)

    if os.path.isdir(log_path) == False:
        os.makedirs(log_path)

    #print(datetime.now().strftime("%m-%d-%H:%M:%S"))

    # load dataset
    dataset_name = args.dataset.split("/")
    if args.dataset in ["telegram"]:
        data = load_directed_real_data(
            dataset=dataset_name[0], name=dataset_name[0]
        ).to(device)
    elif args.dataset == "sdsbm":
        df = pd.read_csv("./SDSBM/data" + str(args.sdsbm_number) + ".csv")
        # print(df.head())
        A, labels= df.iloc[:, :-1].values, df.iloc[:, -1].values
        A = csr_matrix(A)
        A, labels = extract_network(A=A, labels=labels)
        data = SignedData(A=A, y=torch.LongTensor(labels))
        data = node_class_split(
            data,
            train_size_per_class=1 - args.split_prob[0] - args.split_prob[1],
            val_size_per_class=args.split_prob[0],
            test_size_per_class=args.split_prob[1],
            data_split=args.runs,
        )
    else:
        data = load_signed_real_data(dataset=args.dataset).to(device)

    subset = args.dataset

    dataset = data

    if not data.__contains__("edge_weight"):
        dataset.edge_weight = None
    else:
        dataset.edge_weight = torch.FloatTensor(dataset.edge_weight)

    size = dataset.y.size(-1)
    f_node, e_node = dataset.edge_index[0], dataset.edge_index[1]

    # dataset = dataset.to(device)
    edge_index = dataset.edge_index.to(device)
    edge_weight = dataset.edge_weight.to(device)

    # size = torch.max(edge_index).item() + 1
    # dataset.num_nodes = size

    label = dataset.y.data.numpy().astype("int")
    train_mask = dataset.train_mask.data.numpy().astype("bool_")
    val_mask = dataset.val_mask.data.numpy().astype("bool_")
    test_mask = dataset.test_mask.data.numpy().astype("bool_")
    # normalize label, the minimum should be 0 as class index

    args.num_class_node = int(np.amax(label) - np.amin(label) + 1)
    # print(args.num_class_node)
    _label_ = label - np.amin(label)
    label = torch.from_numpy(_label_[np.newaxis]).to(device)
    label = label.reshape(label.size()[1])
    # print(label)

    dataset = dataset.to(device)

    X_real = in_out_degree(edge_index=dataset.edge_index, size=size, signed=args.sd_input_feat, edge_weight=dataset.edge_weight).to(device)
    X_img = X_real.clone()

    # print(X_real)

    criterion = nn.NLLLoss()

    """
    if args.task in ["direction", "existence", "sign"]:
        args.num_class_link = 2
    elif args.task == "three_class_digraph":
        args.num_class_link = 3
    elif args.task == "four_class_signed_digraph":
        args.num_class_link = 4
    elif args.task == "five_class_signed_digraph":
        args.num_class_link = 5
    """

    if args.dataset == "sdsbm":
        args.sd_input_feat = True
        args.num_input = 4
    else:
        args.sd_input_feat = False
        args.num_input = 2

    # num_classes = int(dataset.y.max() - dataset.y.min() + 1)
    save_file = args.dataset + "/" + subset

    # print(datetime.now().strftime("%m-%d-%H:%M:%S"))
    # split dataset
    # dataset.node_split(
    #    train_size=1-args.split_prob[0]-args.split_prob[1],
    #    val_size=args.split_prob[0],
    #    test_size=args.split_prob[1],
    #    data_split=args.runs,
    # )

    splits = train_mask.shape[1]
    #print("split")
    #print(splits)
    if len(test_mask.shape) == 1:
        test_mask = np.repeat(test_mask[:, np.newaxis], splits, 1)
    # data = data.to(device)
    also_neg = False
    results = np.zeros((args.runs, args.num_result))

    # print(datetime.now().strftime("%m-%d-%H:%M:%S"))
    # print(device)

    for i in range(args.runs):
        log_str_full = ""

        ########################################
        # get hermitian laplacian
        ########################################

        train_index = train_mask[:,i]
        #print(train_index)
        #print(label[train_index])
        val_index = val_mask[:,i]
        test_index = test_mask[:,i]
        
        edge_index = dataset.edge_index.to(device)
        edge_weight = dataset.edge_weight.to(device)

        # edge_index = dataset[i].edge_index
        # edge_weight = dataset[i].edge_weight

        # edge_index = edge_index.to(device)
        # edge_weight = edge_weight.to(device)

        # X_real = in_out_degree(edge_index=edge_index, size=size, signed=args.sd_input_feat, edge_weight=edge_weight).to(device)
        # X_img = X_real.clone()

        ########################################
        # initialize model and load dataset
        ########################################
        if args.method == "MSGNN":
            curr_graph = SignedData(edge_index=edge_index, edge_weight=edge_weight)
            args.q = 0.5 / (curr_graph.A - curr_graph.A.transpose()).max()

            model = MSGNN_node_classification(
                q=args.q,
                K=args.K,
                num_features=args.num_input,
                hidden=args.num_filter,
                label_dim=args.num_class_node,
                trainable_q=args.trainable_q,
                layer=args.num_layers,
                dropout=args.dropout,
                normalization=args.normalization,
                cached=(not args.trainable_q),
            ).to(device)
        elif args.method == "PSM_MSGNN":
            cached = True
            model = PSM_MSGNN_node_classification(
                K=args.K,
                num_features=args.num_input,
                hidden=args.num_filter,
                label_dim=args.num_class_node,
                layer=args.num_layers,
                dropout=args.dropout,
                normalization=args.normalization,
                cached=cached
            ).to(device)
        elif args.method == "SigMaNet":
            gcn = True
            edge_index, norm_real, norm_imag = laplacian.process_magnetic_laplacian(
                edge_index=edge_index,
                gcn=gcn,
                net_flow=args.netflow,
                x_real=X_real,
                edge_weight=edge_weight,
                normalization=args.normalization,
                return_lambda_max=False,
            )
            model = SigMaNet_node_prediction_one_laplacian(
                K=args.K,
                num_features=args.num_input,
                hidden=args.num_filter,
                label_dim=args.num_class_node,
                i_complex=False,
                layer=args.num_layers,
                follow_math=args.follow_math,
                gcn=gcn,
                net_flow=args.netflow,
                unwind=True,
                edge_index=edge_index,
                norm_real=norm_real,
                norm_imag=norm_imag,
            ).to(device)
        elif args.method == "PSM_SigMaNet":
            gcn = True
            edge_index, norm_real, norm_imag = new_laplacian.process_magnetic_laplacian(
                edge_index=edge_index,
                gcn=gcn,
                net_flow=args.netflow,
                x_real=X_real,
                edge_weight=edge_weight,
                normalization=args.normalization,
                return_lambda_max=False,
            )
            model = PSM_SigMaNet_node_prediction_one_laplacian(
                K=args.K,
                num_features=args.num_input,
                hidden=args.num_filter,
                label_dim=args.num_class_node,
                i_complex=False,
                layer=args.num_layers,
                follow_math=args.follow_math,
                gcn=gcn,
                net_flow=args.netflow,
                unwind=True,
                edge_index=edge_index,
                norm_real=norm_real,
                norm_imag=norm_imag,
            ).to(device)

        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        """
        y_train = datasets[i]["train"]["label"]
        y_val = datasets[i]["val"]["label"]
        y_test = datasets[i]["test"]["label"]
        y_train = y_train.long().to(device)
        y_val = y_val.long().to(device)
        y_test = y_test.long().to(device)
        """
        # print(y_test)

        # train_index = train_mask[:,i]
        # val_index = val_mask[:,i]
        # test_index = test_mask[:,i]

        # y_test_array = y_test.cpu().numpy().reshape(1, -1)

        #################################
        # Train/Validation/Test
        #################################
        best_val_err = 1000.0
        best_val_acc = 0.0
        early_stopping = 0
        for epoch in range(args.epochs):
            start_time = time.time()
            if early_stopping > 500:
                break
            ####################
            # Train
            ####################
            train_loss, train_acc = 0.0, 0.0
            model.train()

            if args.method in ["MSGNN", "PSM_MSGNN"]:
                Z, out, _, prob = model(
                    X_real,
                    X_img,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )
            else:
                out = model(X_real, X_img)
                # out = out[2]

            # print(out)
            # print(type(out))
            # print(out.shape)
            # print(label[train_index])

            # train_loss = criterion(out[train_index], label[train_index])
            train_loss = F.nll_loss(out[train_index], label[train_index])

            # train_loss = F.nll_loss(pred_label[train_index], label[train_index])
            pred_label = out.max(dim=1)[1]
            # print(pred_label)
            # print(pred_label.shape)
            # print(y_train.shape)
            train_acc, _, _, _ = scores(args.num_class_node, pred_label[train_index], label[train_index])

            opt.zero_grad()
            train_loss.backward()
            opt.step()
            outstrtrain = "Train loss: %.6f, acc: %.3f" % (
                train_loss.detach().item(),
                train_acc,
            )

            ####################
            # Validation
            ####################
            train_loss, train_acc = 0.0, 0.0
            model.eval()

            if args.method in ["MSGNN", "PSM_MSGNN"]:
                Z, out, _, prob = model(
                    X_real,
                    X_img,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )
            else:
                out = model(X_real, X_img)
                # out = out[2]

            val_loss = F.nll_loss(out[val_index], label[val_index])
            pred_label = out.max(dim=1)[1]
            val_acc, _, _, _ = scores(
                args.num_class_node, pred_label[val_index], label[val_index]
            )

            outstrval = " Val loss: %.6f, acc: %.3f" % (
                val_loss.detach().item(),
                val_acc,
            )
            duration = "--- %.4f seconds ---" % (time.time() - start_time)
            log_str = (
                ("%d / %d epoch" % (epoch, args.epochs))
                + outstrtrain
                + outstrval
                + duration
            )
            log_str_full += log_str + "\n"
            ####################
            # Save weights
            ####################
            save_perform_err = val_loss.detach().item()
            save_perform_acc = val_acc
            if save_perform_err <= best_val_err:
                early_stopping = 0
                best_val_err = save_perform_err
                torch.save(model.state_dict(), log_path + "/model_err" + str(i) + ".t7")
            if save_perform_acc >= best_val_acc:
                best_val_acc = save_perform_acc
                torch.save(model.state_dict(), log_path + "/model_acc" + str(i) + ".t7")
            else:
                early_stopping += 1
        torch.save(model.state_dict(), log_path + "/model_latest" + str(i) + ".t7")
        write_log(vars(args), log_path)

        ####################
        # Testing
        ####################
        model.load_state_dict(torch.load(log_path + "/model_err" + str(i) + ".t7"))
        model.eval()

        if args.method in ["MSGNN", "PSM_MSGNN"]:
            Z, out, _, prob = model(
                X_real,
                X_img,
                edge_index=edge_index,
                #query_edges=test_index,
                edge_weight=edge_weight,
            )
        else:
            out = model(X_real, X_img)
            # out = out[2]

        pred_label = out.max(dim=1)[1]

        # save the test case result
        # pred_label_array = pred_label.cpu().numpy().reshape(1, -1)
        # y_test_array = np.array(y_test).reshape(1, -1)
        output_array = np.concatenate(
            [pred_label[test_index].cpu().numpy().reshape(1, -1), label[test_index].cpu().numpy().reshape(1, -1)],
            axis=1,
        )
        # output_array = np.concatenate([y_test_array, pred_label_array], axis=1)
        df_output = pd.DataFrame(output_array)
        df_output.to_csv(os.path.join(log_path, "test_output" + str(i) + ".csv"), index=False, header=False)
        # df_pred_label = pd.DataFrame(pred_label)
        # df_y_test = pd.DataFrame(y_test)
        # pred_label_csv.save(log_path)

        test_acc, f_score, macro_fscore, micro_fscore = scores(
            args.num_class_node, pred_label[test_index], label[test_index]
        )

        if args.num_class_node == 2:
            neg_acc, pos_acc = neg_and_pos_acc(pred_label[test_index], label[test_index])
        else:
            neg_acc, pos_acc = 0, 0

        print(
            "acc : {test_acc:.4f}, f1_score : {f_score:.4f}, macro_fscore : {macro_fscore:.4f}, micro_fscore : {micro_fscore:.4f}".format(
                test_acc=test_acc,
                f_score=f_score,
                macro_fscore=macro_fscore,
                micro_fscore=micro_fscore,
            )
        )
        ####################
        # Save testing results
        ####################
        total_time = time.time() - global_start_time
        results[i] = [test_acc, f_score, macro_fscore, micro_fscore, neg_acc, pos_acc, total_time]
        with open(log_path + "/log" + str(i) + ".csv", "w") as file:
            file.write(log_str_full)
            file.write("\n")
        torch.cuda.empty_cache()

    return results

def save_and_output(args, results, path):
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

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.epochs = 1
        #args.runs = 1

    save_name = (
        "lr"
        + str(int(args.lr * 1000))
        + "num_filters"
        + str(int(args.num_filter))
        + "task"
        + args.task
        + "layers"
        + str(args.num_layers)
    )
    args.save_name = save_name
    date_time = datetime.now().strftime("%m-%d-%H:%M:%S")
    
    for i in range(4):
        args.sdsbm_number = i
        
        dir_name = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../result_arrays",
            args.method,
            args.dataset + str(args.sdsbm_number),
            save_name,
            date_time
        )
        if os.path.isdir(dir_name) == False:
            try:
                os.makedirs(dir_name)
            except FileExistsError:
                print("Folder exists!")
        
        results = main(args)
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(os.path.join(dir_name, "results.csv"))
        save_and_output(args, results, dir_name)
