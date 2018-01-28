import numpy as np
import tensorflow as tf

def call_lk(embeddings, adjmat, lambdas):
    
    n = adjmat.shape[0]

    identity = np.eye(n)

    adjmat_k = identity
    reach_old = np.diag(np.ones(n))

    for hop_perc in lambdas:
        print('reach_old')
        print(reach_old)

        adjmat_k = np.matmul(adjmat_k, adjmat)

        print('adjmat_k')
        print(adjmat_k)

        adjmat_k_ov = adjmat_k + reach_old

        print('adjmat_k_ov')
        print(adjmat_k_ov)

        reach_k = np.greater(adjmat_k_ov, np.zeros((n, n))).astype(np.float32)

        print('reach_k')
        print(reach_k)

        k_hops = reach_k - reach_old

        print('k_hops')
        print(k_hops)

        reach_old = reach_k

        d = np.sum(k_hops, axis=1)

        d = np.array(d).squeeze()

        degree = np.diag(d, k=0)

        L_k = degree - k_hops

        R = L_k

        print(degree)

        print(np.trace(np.matmul(np.matmul(np.transpose(embeddings), R), embeddings)) / np.trace(R))

    degree = np.array(np.sum(adjmat, 1)).squeeze()

    digs = np.diag(degree)

    print('digs')
    print(digs)

    print(np.matmul(np.matmul(np.transpose(embeddings), digs), embeddings))


call_lk(np.array([[0, 2, 3], [1, 4, 5], [2, 5,6], [3, 5, 6]]), np.mat([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]), [1,1, 1, 1])

kkk = tf.diag(tf.ones(5))

print(tf.ones(5))
print(kkk)

print(tf.Session().run(kkk))