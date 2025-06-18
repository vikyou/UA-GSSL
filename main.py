import numpy as np
from skimage.segmentation import slic
import cv2
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import math
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from g2g.utils import load_dataset, score_link_prediction, score_node_classification
from model import Graph2Gauss
import os

with open('/data/oym/Pro/RemoteSen/graph2gauss-master/data/Farmland/label_gr.txt', 'r') as f:
    z = [int(line.strip().split()[1]) for line in f.readlines()]
n = len(z)

with open('/data/oym/Pro/RemoteSen/graph2gauss-master/data/Farmland/adj1.txt', 'r') as f:
    lines = f.readlines()
matrix1 = [list(map(int, line.strip().split())) for line in lines]

with open('/data/oym/Pro/RemoteSen/graph2gauss-master/data/Farmland/adj2.txt', 'r') as f:
    lines = f.readlines()
matrix2 = [list(map(int, line.strip().split())) for line in lines]

edges = []
with open('/data/oym/Pro/RemoteSen/graph2gauss-master/data/Farmland/adj1.txt', 'r') as f:
    for line in f:
        edge = line.strip().split()
        edges.append(edge)
num_nodes = max(max(int(edge[0]), int(edge[1])) for edge in edges) + 1
adj1 = np.zeros((num_nodes, num_nodes))
for edge in edges:
    src, dst = map(int, edge[:2])
    weight = float(edge[2]) if len(edge) > 2 else 1.0
    adj1[src][dst] = weight
adj_sparse1 = sp.csr_matrix(adj1)
A1 = adj_sparse1

edges = []
with open('/data/oym/Pro/RemoteSen/graph2gauss-master/data/Farmland/adj2.txt', 'r') as f:
    for line in f:
        edge = line.strip().split()
        edges.append(edge)
num_nodes = max(max(int(edge[0]), int(edge[1])) for edge in edges) + 1
adj2 = np.zeros((num_nodes, num_nodes))
for edge in edges:
    src, dst = map(int, edge[:2])
    weight = float(edge[2]) if len(edge) > 2 else 1.0
    adj2[src][dst] = weight
adj_sparse2 = sp.csr_matrix(adj2)
A2 = adj_sparse2

with open('/data/oym/Pro/RemoteSen/graph2gauss-master/data/Farmland/f1.txt', 'r') as f:
    lines = f.readlines()
num_rows, num_cols = len(lines), len(lines[0].strip().split('\t'))
data, row_ind, col_ind = [], [], []
for i in range(num_rows):
    line = lines[i].strip().split('\t')
    for j in range(num_cols):
        value = line[j]
        if value != "0":
            data.append(float(value))
            row_ind.append(i)
            col_ind.append(j)
sparse_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(num_rows, num_cols))
X1 = sparse_matrix

with open('/data/oym/Pro/RemoteSen/graph2gauss-master/data/Farmland/f2.txt', 'r') as f:
    lines = f.readlines()
num_rows, num_cols = len(lines), len(lines[0].strip().split('\t'))
data, row_ind, col_ind = [], [], []
for i in range(num_rows):
    line = lines[i].strip().split('\t')
    for j in range(num_cols):
        value = line[j]
        if value != "0":
            data.append(float(value))
            row_ind.append(i)
            col_ind.append(j)
sparse_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(num_rows, num_cols))
X2 = sparse_matrix

g2g = Graph2Gauss(A=A1, A2=A2, X=X1, X2=X2, matrix1=matrix1, matrix2=matrix2, L=64, K=3, verbose=True,
                  p_val=0.0, p_test=0.0, p_nodes=0.0, n_hidden=None, seed=1, max_iter=8000, tolerance=2000)

sess = g2g.train(gpu_list='0')
mu1, sigma1 = sess.run([g2g.mu1, g2g.sigma1])
mu2, sigma2 = sess.run([g2g.mu2, g2g.sigma2])

mu = np.zeros(n)
for i in range(n):
    mu[i] = np.linalg.norm(mu1[i]-mu2[i])

y = np.mean(mu)

pre = np.zeros(n,dtype=int)
for i in range(n):
    if(mu[i] >= y):
        pre[i] = 1

with open('label_pre-Farmland-beijing_A.txt', 'w') as f:
    for i in range(n):
        f.write(str(pre[i])+'\n')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
accuracy = accuracy_score(z, pre)
kappa = cohen_kappa_score(z, pre)

