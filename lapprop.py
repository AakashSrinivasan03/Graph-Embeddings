import numpy as np
import tensorflow as tf
from utils.metrics import *

class RegularizerPos():

    def __init__(self, config):
        self.max_hop_depth = config.loss_k_depth
        self.lambdas = config.lambdas
        self.reg_weight = config.reg_weight
        self.embed_dims = config.embed_dims
        self.loss_fun_type = config.loss_fun_type
        self.normalise_pos = config.normalise_pos
        self.normalise_neg = config.normalise_neg


    def call_lk(self, inputs):

        embeddings = inputs['embeddings']
        adjmat = tf.sparse_to_dense(inputs['adjmat'].indices, inputs['adjmat'].dense_shape, inputs['adjmat'].values,
                                    validate_indices=False)
        degrees = tf.diag(inputs['degrees'])

        n = tf.shape(adjmat)[0]

        identity = tf.eye(n)

        adjmat_k = identity
        adjmat_k_ov = tf.diag(tf.ones([n]))

        loss = 0

        hop_loss = []
        hop_loss.append(0)

        for hop_perc in self.lambdas:
            adjmat_k = tf.matmul(adjmat_k, adjmat)
            adjmat_k=tf.cast(tf.greater(adjmat_k, tf.zeros((n, n))), tf.float32)
            compute_k = adjmat_k - adjmat_k_ov
            adjmat_k_ov = adjmat_k_ov + adjmat_k

            reach_k = tf.cast(tf.greater(compute_k, tf.zeros((n, n))), tf.float32)

            #k_hops = reach_k - reach_old

            #reach_old = reach_k

            d = tf.reduce_sum(reach_k, 1)

            degree = tf.diag(d)

            L_k = degree - reach_k

            R = L_k

            if self.loss_fun_type == 'exponential':
                loss = loss + 1 - tf.exp(-1 * hop_perc * (
                tf.trace(tf.matmul(tf.matmul(tf.transpose(embeddings), R), embeddings))) / tf.trace(R))

            elif self.loss_fun_type == 'linear' and self.normalise_pos == 'norm':
                loss += hop_perc * (tf.trace(tf.matmul(tf.matmul(tf.transpose(embeddings), R), embeddings))) / tf.norm(
                    R)

            elif self.loss_fun_type == 'linear' and self.normalise_pos == 'trace':
                loss += hop_perc * (tf.trace(tf.matmul(tf.matmul(tf.transpose(embeddings), R), embeddings))) / tf.trace(
                    R)

            elif self.loss_fun_type == 'linear' and self.normalise_pos == 'l1_norm':
                loss += hop_perc * (tf.trace(tf.matmul(tf.matmul(tf.transpose(embeddings), R), embeddings))) / tf.norm(
                    R, ord='1')

            elif self.loss_fun_type == 'linear' and self.normalise_pos == 'none':
                loss += hop_perc * (tf.trace(tf.matmul(tf.matmul(tf.transpose(embeddings), R), embeddings)))

            else:
                print('YOU MESSED UP')

            hop_loss.append(loss - hop_loss[-1])
            hop_loss.append(loss)

        yt_d_y = tf.matmul(tf.matmul(tf.transpose(embeddings), degrees), embeddings)

        identity = tf.eye(self.embed_dims)

        if self.loss_fun_type == 'exponential':
            reg_term = 1 - tf.exp(-1 * self.reg_weight * tf.sqrt(tf.reduce_sum(tf.square(yt_d_y - identity))))

        elif self.loss_fun_type == 'linear' and self.normalise_neg == 'norm':
            reg_term = self.reg_weight * tf.sqrt(tf.reduce_sum(tf.square(yt_d_y - identity))) / tf.norm(degrees)

        elif self.loss_fun_type == 'linear' and self.normalise_neg == 'none':
            reg_term = self.reg_weight * tf.abs(tf.norm(yt_d_y - identity))

        pos_term = loss

        loss += reg_term

        return loss, pos_term, reg_term, hop_loss

    def call_lpowerk(self, inputs):

        embeddings = inputs['embeddings']
        adjmat = tf.sparse_to_dense(inputs['adjmat'].indices, inputs['adjmat'].dense_shape, inputs['adjmat'].values, validate_indices=False)
        degrees = tf.diag(inputs['degrees'])

        L = degrees - adjmat

        identity = tf.eye(tf.shape(adjmat)[0])

        L_k = identity

        # L_k = tf.matmul(tf.pow(degrees, -0.5), tf.transpose(tf.matmul(tf.pow(degrees, -0.5), L), [1, 0]))

        loss = 0

        for hop_perc in self.lambdas:
            L_k = tf.matmul(L_k, L)

            # TODO Residual R = L + Attention matrix
            R = L_k


            if self.loss_fun_type == 'exponential':
                loss = loss + 1 - tf.exp(-1 * hop_perc*(tf.trace(tf.matmul(tf.matmul(tf.transpose(embeddings), R), embeddings)))/tf.trace(R))

            elif self.loss_fun_type == 'linear' and self.normalise_pos == 'norm':
                loss += hop_perc * (tf.trace(tf.matmul(tf.matmul(tf.transpose(embeddings), R), embeddings))) / tf.norm(R)

            elif self.loss_fun_type == 'linear' and self.normalise_pos == 'trace':
                loss += hop_perc * (tf.trace(tf.matmul(tf.matmul(tf.transpose(embeddings), R), embeddings))) / tf.trace(R)

            elif self.loss_fun_type == 'linear' and self.normalise_pos == 'l1_norm':
                loss += hop_perc * (tf.trace(tf.matmul(tf.matmul(tf.transpose(embeddings), R), embeddings))) / tf.norm(R, ord='1')

            elif self.loss_fun_type == 'linear' and self.normalise_pos == 'none':
                loss += hop_perc * (tf.trace(tf.matmul(tf.matmul(tf.transpose(embeddings), R), embeddings)))

            else:
                print('YOU MESSED UP')

        yt_d_y = tf.matmul(tf.matmul(tf.transpose(embeddings), degrees), embeddings)

        identity = tf.eye(self.embed_dims)

        if self.loss_fun_type == 'exponential':
            reg_term = 1 - tf.exp(-1 * self.reg_weight*tf.sqrt(tf.reduce_sum(tf.square(yt_d_y - identity))))

        elif self.loss_fun_type == 'linear' and self.normalise_neg == 'norm':
            reg_term = self.reg_weight * tf.sqrt(tf.reduce_sum(tf.square(yt_d_y - identity))) / tf.norm(degrees)

        elif self.loss_fun_type == 'linear' and self.normalise_neg == 'none':
            reg_term = self.reg_weight * tf.sqrt(tf.reduce_sum(tf.square(yt_d_y - identity)))



        pos_term = loss

        loss += reg_term

        return loss, pos_term, reg_term

class Propagation:
    def __init__(self, config, data):
        #
        # self.inputs = data['features']
        # self.l2 = config.l2
        # self.regularization = config.loss['regKernel']
        # self.add_labels = config.add_labels
        # self.bias = config.bias
        #
        # self.n_labels = config.n_labels
        # self.wce_val = data['wce']

        self.n_node_ids = data['n_node_ids']
        self.oe_id = data['outer_epoch']
        self.max_oe = data['max_oe']
        self.lr = data['lr']
        self.optimizer = config.opt(self.lr)
        self.learning_rate = config.learning_rate
        self.laplacian = config.laplacian

        ## self.optimizer = config.opt(learning_rate=data['lr'])

        self.density = data['batch_density']

        self.loss, self.pos_loss, self.neg_loss, self.hop_loss = self.add_regularizers(config, data)

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

        grads_and_vars = self.optimizer.compute_gradients(self.pos_loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op_pos = self.optimizer.apply_gradients(clipped_grads_and_vars)

        grads_and_vars = self.optimizer.compute_gradients(self.neg_loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0] 
        self.opt_op_neg = self.optimizer.apply_gradients(clipped_grads_and_vars)

    def add_regularizers(self, config, data):
        # if config.regKernel['metric_learning']:
        #     M = tf.get_variable(name='M', shape=(config.regKernel['dim'], config.regKernel['dim']))
        # else:
        #     M = tf.cast(tf.identity(config.regKernel['dim'], 'M'), type=tf.float32)

        self.regPos = RegularizerPos(config)

        if self.laplacian == 'lk':
            return self.regPos.call_lk(data)

        elif self.laplacian == 'lpowerk':
            return self.regPos.call_lpowerk(data)

    def _loss(self):
        self.loss = 0

        # Structure Regularization Loss
        self.loss += self.loss

        tf.summary.scalar('loss', self.loss)
