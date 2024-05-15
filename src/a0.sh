#!/bin/bash

python syn_edge_pred.py --method=MSGNN --task=existence

python syn_edge_pred.py --method=MSGNN --task=direction

python syn_edge_pred.py --method=MSGNN --task=sign

python syn_edge_pred.py --method=MSGNN --task=three_class_digraph

python syn_edge_pred.py --method=MSGNN --task=four_class_signed_digraph

python syn_edge_pred.py --method=MSGNN --task=five_class_signed_digraph

python syn_edge_pred.py --method=PSM_MSGNN --task=existence

python syn_edge_pred.py --method=PSM_MSGNN --task=direction

python syn_edge_pred.py --method=PSM_MSGNN --task=sign

python syn_edge_pred.py --method=PSM_MSGNN --task=three_class_digraph

python syn_edge_pred.py --method=PSM_MSGNN --task=four_class_signed_digraph

python syn_edge_pred.py --method=PSM_MSGNN --task=five_class_signed_digraph

python syn_edge_pred.py --method=SigMaNet --task=existence

python syn_edge_pred.py --method=SigMaNet --task=direction

python syn_edge_pred.py --method=SigMaNet --task=sign

python syn_edge_pred.py --method=SigMaNet --task=three_class_digraph

python syn_edge_pred.py --method=SigMaNet --task=four_class_signed_digraph

python syn_edge_pred.py --method=SigMaNet --task=five_class_signed_digraph

python syn_edge_pred.py --method=PSM_SigMaNet --task=existence

python syn_edge_pred.py --method=PSM_SigMaNet --task=direction

python syn_edge_pred.py --method=PSM_SigMaNet --task=sign

python syn_edge_pred.py --method=PSM_SigMaNet --task=three_class_digraph

python syn_edge_pred.py --method=PSM_SigMaNet --task=four_class_signed_digraph

python syn_edge_pred.py --method=PSM_SigMaNet --task=five_class_signed_digraph

python syn_node_classification.py --method=MSGNN

python syn_node_classification.py --method=PSM_MSGNN

python syn_node_classification.py --method=SigMaNet

python syn_node_classification.py --method=PSM_SigMaNet