import numpy as np
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
from g2g.utils import *
from scipy.sparse import csr_matrix
import random
import math
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

eps = 1e-14

with open('/data/oym/Pro/RemoteSen/graph2gauss-master/data/Farmland/label_gr.txt', 'r') as f:
    z = []
    for line in f.readlines():
        data = line.strip().split()[1]
        z.append(int(data))
node = len(z)

class Graph2Gauss:

    def __init__(self, A, A2, X, X2, matrix1, matrix2, L, K=3, p_val=0.10, p_test=0.05, p_nodes=0.0,
                 n_hidden=None, max_iter=2000, tolerance=100, scale=False, seed=1, verbose=True):

        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)

        X = X.astype(np.float32)
        X2 = X2.astype(np.float32)

        if p_nodes > 0:
            A = self.__setup_inductive(A, X, p_nodes)
            A2 = self.__setup_inductive(A2, X2, p_nodes)
        else:
            self.X = tf.SparseTensor(*sparse_feeder(X))
            self.X2 = tf.SparseTensor(*sparse_feeder(X2))
            self.feed_dict = None

        self.matrix1 = matrix1
        self.matrix2 = matrix2
        self.N, self.D = X.shape
        self.L = L
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.scale = scale
        self.verbose = verbose

        if n_hidden is None:
            n_hidden = [512]
        self.n_hidden = n_hidden

        if p_val + p_test > 0:
            train_ones, val_ones, val_zeros, test_ones, test_zeros = train_val_test_split_adjacency(
                A=A, p_val=p_val, p_test=p_test, seed=seed, neg_mul=1, every_node=True, connected=False,
                undirected=(A != A.T).nnz == 0)
            A_train = edges_to_sparse(train_ones, self.N)
            hops = get_hops(A_train, K)
        else:
            hops = get_hops(A, K)

        scale_terms = {h if h != -1 else max(hops.keys()) + 1:
                           hops[h].sum(1).A1 if h != -1 else hops[1].shape[0] - hops[h].sum(1).A1
                       for h in hops}

        self.__build()
        self.__build_loss()

        if p_val > 0:
            val_edges = np.row_stack((val_ones, val_zeros))
            self.neg_val_energy = -self.energy_kl(val_edges)
            self.val_ground_truth = A[val_edges[:, 0], val_edges[:, 1]].A1
            self.val_early_stopping = True
        else:
            self.val_early_stopping = False

        if p_test > 0:
            test_edges = np.row_stack((test_ones, test_zeros))
            self.neg_test_energy = -self.energy_kl(test_edges)
            self.test_ground_truth = A[test_edges[:, 0], test_edges[:, 1]].A1

        if p_nodes > 0:
            self.neg_ind_energy = -self.energy_kl(self.ind_pairs)

    def __build(self):
        w_init = tf.keras.initializers.glorot_normal(seed=None)

        sizes = [self.D] + self.n_hidden

        for i in range(1, len(sizes)):
            W = tf.Variable(name='W{}'.format(i), initial_value=w_init([sizes[i - 1], sizes[i]]),
                            dtype=tf.float32)
            b = tf.Variable(name='b{}'.format(i), initial_value=w_init([sizes[i]]),
                            dtype=tf.float32)

            W1 = tf.Variable(name='W_{}'.format(i), initial_value=w_init([sizes[i], sizes[i]]),
                             dtype=tf.float32)
            b1 = tf.Variable(name='b_{}'.format(i), initial_value=w_init([sizes[i]]),
                             dtype=tf.float32)

            W2 = tf.Variable(name='W_{}'.format(i), initial_value=w_init([sizes[i], sizes[i]]),
                             dtype=tf.float32)
            b2 = tf.Variable(name='b_{}'.format(i), initial_value=w_init([sizes[i]]),
                             dtype=tf.float32)

            if i == 1:
                encoded1 = tf.sparse.sparse_dense_matmul(self.X, W) + b
                encoded2 = tf.sparse.sparse_dense_matmul(self.X2, W) + b
            else:
                encoded1 = tf.matmul(encoded1, W) + b
                encoded2 = tf.matmul(encoded2, W) + b
                encoded1 = tf.nn.relu(encoded1)
                encoded2 = tf.nn.relu(encoded2)
                encoded1 = tf.matmul(encoded1, W1) + b1
                encoded2 = tf.matmul(encoded2, W1) + b1
                encoded1 = tf.nn.relu(encoded1)
                encoded2 = tf.nn.relu(encoded2)
                encoded1 = tf.matmul(encoded1, W2) + b2
                encoded2 = tf.matmul(encoded2, W2) + b2

            encoded1 = tf.nn.relu(encoded1)
            encoded2 = tf.nn.relu(encoded2)

        W_mu = tf.Variable(name='W_mu', initial_value=w_init([sizes[-1], self.L]), dtype=tf.float32)
        b_mu = tf.Variable(name='b_mu', initial_value=w_init([self.L]), dtype=tf.float32)
        self.mu1 = tf.matmul(encoded1, W_mu) + b_mu
        self.mu2 = tf.matmul(encoded2, W_mu) + b_mu

        W_sigma = tf.Variable(name='W_sigma', initial_value=w_init([sizes[-1], self.L]), dtype=tf.float32)
        b_sigma = tf.Variable(name='b_sigma', initial_value=w_init([self.L]), dtype=tf.float32)
        log_sigma1 = tf.matmul(encoded1, W_sigma) + b_sigma
        log_sigma2 = tf.matmul(encoded2, W_sigma) + b_sigma
        self.sigma1 = tf.nn.elu(log_sigma1) + 1 + 1e-14
        self.sigma2 = tf.nn.elu(log_sigma2) + 1 + 1e-14

    def __build_loss(self, batch_size=512, reg_weight=1e-6):
        matrix1 = self.matrix1
        matrix2 = self.matrix2
        a = 0.5
        b = 0.5
        c = 0.5
        alpha = 0.01
        beta = 0.01

        loss1 = 0
        loss2 = 0
        n = node // batch_size + 1

        for i in range(n):
            pair_pos1 = np.array([], dtype=np.int64).reshape(0, 2)
            pair_pos2 = np.array([], dtype=np.int64).reshape(0, 2)
            pair_pos3 = np.array([], dtype=np.int64).reshape(0, 2)

            pair_neg1 = np.array([], dtype=np.int64).reshape(0, 2)
            pair_neg2 = np.array([], dtype=np.int64).reshape(0, 2)
            pair_neg3 = np.array([], dtype=np.int64).reshape(0, 2)
            pair_neg4 = np.array([], dtype=np.int64).reshape(0, 2)

            for j in range(batch_size):
                idx = i * batch_size + j
                if idx >= node:
                    break
                if matrix1[idx * 3][0] == idx:
                    pair_pos1 = np.concatenate([pair_pos1, [[idx, matrix1[idx * 3][1]]]], axis=0)
                    pair_pos2 = np.concatenate([pair_pos2, [[idx, matrix2[idx * 3][1]]]], axis=0)

                    pair_pos1 = np.concatenate([pair_pos1, [[idx, matrix1[idx * 3 + 1][1]]]], axis=0)
                    pair_pos2 = np.concatenate([pair_pos2, [[idx, matrix2[idx * 3 + 1][1]]]], axis=0)

                    pair_pos1 = np.concatenate([pair_pos1, [[idx, matrix1[idx * 3 + 2][1]]]], axis=0)
                    pair_pos2 = np.concatenate([pair_pos2, [[idx, matrix2[idx * 3 + 2][1]]]], axis=0)

                    pair_pos3 = np.concatenate([pair_pos3, [[idx, node + idx]]], axis=0)

                    p = 3
                    while p > 0:
                        jj = random.randint(0, node - 1)
                        if jj != idx and jj != matrix1[idx * 3][1] and jj != matrix1[idx * 3 + 1][1] and jj != \
                                matrix1[idx * 3 + 2][1] and jj != matrix2[idx * 3][1] and jj != matrix2[idx * 3 + 1][1] and jj != \
                                matrix2[idx * 3 + 2][1]:
                            p -= 1
                            pair_neg1 = np.concatenate([pair_neg1, [[idx, jj]]], axis=0)
                            pair_neg2 = np.concatenate([pair_neg2, [[idx, jj]]], axis=0)
                            pair_neg3 = np.concatenate([pair_neg3, [[idx, node + jj]]], axis=0)
                            pair_neg4 = np.concatenate([pair_neg4, [[jj, node + idx]]], axis=0)

            pair_pos1 = tf.convert_to_tensor(pair_pos1, dtype=tf.int64)
            pair_pos2 = tf.convert_to_tensor(pair_pos2, dtype=tf.int64)
            pair_pos3 = tf.convert_to_tensor(pair_pos3, dtype=tf.int64)
            pair_neg1 = tf.convert_to_tensor(pair_neg1, dtype=tf.int64)
            pair_neg2 = tf.convert_to_tensor(pair_neg2, dtype=tf.int64)
            pair_neg3 = tf.convert_to_tensor(pair_neg3, dtype=tf.int64)
            pair_neg4 = tf.convert_to_tensor(pair_neg4, dtype=tf.int64)

            eng_pos1 = self.energy_kl(1, pair_pos1)
            eng_pos2 = self.energy_kl(2, pair_pos2)
            energy1 = tf.reduce_sum(eng_pos1) + tf.reduce_sum(eng_pos2)

            max_val = tf.reduce_max(energy1)
            min_val = tf.reduce_min(energy1)
            scale_factor = tf.where(tf.greater(tf.abs(max_val), tf.abs(min_val)), max_val, min_val)
            energy1 = tf.divide(energy1, scale_factor)

            eng_neg1 = self.energy_kl(1, pair_neg1)
            eng_neg2 = self.energy_kl(2, pair_neg2)
            energy2 = tf.reduce_sum(eng_neg1) + tf.reduce_sum(eng_neg2)

            loss12 = energy1 / (energy2 + eps)

            reg_loss = tf.reduce_sum(tf.abs(self.sigma1)) + tf.reduce_sum(tf.abs(self.sigma2))
            eng_pos3 = self.energy_kl(3, pair_pos3)
            energy1 = tf.reduce_mean(eng_pos3)

            max_val = tf.reduce_max(energy1)
            min_val = tf.reduce_min(energy1)
            scale_factor = tf.where(tf.greater(tf.abs(max_val), tf.abs(min_val)), max_val, min_val)
            energy1 = tf.divide(energy1, scale_factor)

            eng_neg3 = self.energy_kl(3, pair_neg3)
            eng_neg4 = self.energy_kl(3, pair_neg4)
            energy2 = tf.reduce_mean(eng_neg3) + tf.reduce_mean(eng_neg4)

            loss11 = energy1 / (energy2 * (1 / 2) + eps)

            loss1 += (loss11 + b * loss12)

            reg_loss1 = tf.reduce_sum(tf.abs(self.mu1)) + tf.reduce_sum(tf.square(self.mu1))
            reg_loss2 = tf.reduce_sum(tf.abs(self.mu2)) + tf.reduce_sum(tf.square(self.mu2))

            ij_mu1 = tf.gather(self.mu1, pair_pos1)
            ij_sigma1 = tf.gather(self.sigma1, pair_pos1)
            ij_mu2 = tf.gather(self.mu2, pair_pos2)
            ij_sigma2 = tf.gather(self.sigma2, pair_pos2)

            e1 = ij_mu1[:, 0] - ij_mu1[:, 1]
            e2 = ij_mu2[:, 0] - ij_mu2[:, 1]
            e3 = ij_sigma1[:, 0] - ij_sigma1[:, 1]
            e4 = ij_sigma2[:, 0] - ij_sigma2[:, 1]

            loss21 = tf.reduce_sum(((tf.square(e1 - e2)) * e3 - (tf.square(e1 - e2)) * e4), 1)
            loss22 = tf.reduce_sum(((tf.square(e1 - e2)) * e3 - (tf.square(e1 - e2)) * e4), 1)

            max_val = tf.reduce_max(loss21)
            min_val = tf.reduce_min(loss21)
            scale_factor = tf.where(tf.greater(tf.abs(max_val), tf.abs(min_val)), max_val, min_val)
            loss21 = tf.divide(loss21, scale_factor)
            loss22 = tf.divide(loss22, scale_factor)

            loss21 = tf.reduce_mean(loss21)
            loss22 = tf.reduce_mean(loss22)

            loss2 += ((1 / 3) * loss21) + c * loss22

            l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            reg_loss = reg_weight * l2_reg

        self.loss = loss1 + a * loss2 + reg_loss
        self.loss = self.loss / n

        return self.loss