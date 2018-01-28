import os
import tensorflow as tf
import time
import threading
import numpy as np
from copy import deepcopy
from tabulate import tabulate

from dataset import Dataset
from parser import Parser
from config import Config
# from src.models.propagation import Propagation
# from src.models.propagation_label import Propagation as Propagation_label
# from src.models.gated_propagation import Propagation

from utils.metrics import *
from utils.utils import remove_directory
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class OuterPropagation(object):

    def __init__(self, dataset):
        self.config = dataset.config
        self.dataset = dataset

        # Variable initialization
        self.placeholders = {}
        self.queue_placeholders_keys = {}
        self.Q = self.enqueue_op = self.dequeue_op = None
        self.coord = tf.train.Coordinator()

        # Setup Architecture
        self.update_predictions_op = None
        self.data = self.model = self.saver = self.summary = None
        self.setup_arch()

        # Setup initializers
        self.init = tf.global_variables_initializer()
        self.init2 = tf.no_op()

    def setup_arch(self):
        # Setup place_holders
        self.placeholders = {}
        self.queue_placeholders_keys = ['mask', 'degrees', 'n_conn_nodes', 'n_node_ids', 'batch_density',
                                        'batch_ids', 'adj_indices', 'adj_data', 'adj_shape']
        self.get_placeholders()

        # Setup Input Queue
        self.Q, self.enqueue_op, self.dequeue_op = self.setup_data_queues()

        # Create model and data for the model
        self.data = self.create_tfgraph_data()

        self.model = self.config.prop_class(self.config, self.data)

        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()

        # TODO gather_nd

        '''

        if self.config.max_outer_epochs > 1:
            zero = tf.constant(0, dtype=tf.int32)
            where = tf.not_equal(self.data['label_mask'], zero)
            indices = tf.squeeze(tf.where(where), 1)
            self.update_predictions_op = tf.scatter_update(self.predictions, self.data['labeled_ids'], tf.gather(self.model.predictions, indices))
            self.update_truth_predictions_op = tf.scatter_update(self.predictions, self.data['labeled_ids'], tf.gather(self.data['targets'], indices))
            self.increment_oe = tf.assign(self.data['outer_epoch'], self.data['outer_epoch']+1)
            self.test_predictions = tf.nn.embedding_lookup(self.predictions, self.data['labeled_ids'], name='curr_labels')

        '''

    def create_tfgraph_data(self):
        data = {}
        (data['label_mask'], data['degrees'], data['n_conn_nodes'], data['n_node_ids'], data['batch_density'],
         data['batch_ids'], adj_indices, adj_data, adj_shape) = self.dequeue_op

        '''

        if self.config.sparse_features:

            data['features'] = tf.SparseTensorValue(f_indices, f_data, f_shape)
        else:
            data['features'] = f_data
        '''

        data['adjmat'] = tf.SparseTensor(indices=adj_indices, values=adj_data, dense_shape=adj_shape)

        ##data['dropout_in'] = self.placeholders['dropout_in']
        ##data['dropout_out'] = self.placeholders['dropout_out']
        ##data['dropout_conv'] = self.placeholders['dropout_conv']
        ##data['wce'] = self.dataset.wce
        ##data['node_features'] = self.config.node_features
        ##data['neighbor_features'] = self.config.neighbor_features

        data['lr'] = self.placeholders['lr']
        data['is_training'] = self.placeholders['is_training']
        data['n_nodes'] = self.config.n_nodes
        data['outer_epoch'] = tf.Variable(0, name='outer_epoch', trainable=False, dtype=tf.int32)
        data['max_oe'] = tf.constant(self.config.max_outer_epochs)


        ### ! can be in gpu

        with tf.device("/cpu:0"):
            self.embeddings = tf.Variable(name='Embeddings', initial_value=tf.random_uniform([self.config.n_nodes, self.config.embed_dims], -1.0, 1.0))

        data['embeddings'] = tf.nn.embedding_lookup(self.embeddings, data['batch_ids'], name='curr_embeddings')


        '''

        if self.config.add_labels:
            self.predictions = tf.Variable(name='predictions',
                                           initial_value=tf.zeros((self.config.n_nodes, self.config.n_labels), dtype=tf.float32), trainable=False)


            data['labels'] = tf.nn.embedding_lookup(self.predictions, data['batch_ids'], name='curr_labels')

        '''

        return data

    def create_feed_dict(self, sources):
        keys = self.queue_placeholders_keys
        feed_dict = {}
        for i, key in enumerate(keys):
            feed_dict[self.placeholders[key]] = sources[i]
        return feed_dict

    def setup_data_queues(self):

        Q = tf.FIFOQueue(capacity=self.config.queue_capacity,
                         dtypes=[tf.int32, tf.float32, tf.int64, tf.int32, tf.float32, tf.int32,  tf.int64, tf.float32, tf.int64])
        keys = self.queue_placeholders_keys
        enqueue_op = Q.enqueue([self.placeholders[key] for key in keys])
        dequeue_op = Q.dequeue()
        return Q, enqueue_op, dequeue_op

    def load_and_enqueue(self, sess, data):
        for idx, batch in enumerate(self.dataset.batch_generator(data, which_degree = self.config.which_degree)):
            feed_dict = self.create_feed_dict(batch)
            sess.run(self.enqueue_op, feed_dict=feed_dict)

    def get_queue_placeholders(self):
        with tf.variable_scope('Queue_placeholders'):
            self.placeholders['mask'] = tf.placeholder(tf.int32, name='mask', shape=None)
            self.placeholders['degrees'] = tf.placeholder(tf.float32, name='degrees', shape=None)
            self.placeholders['n_conn_nodes'] = tf.placeholder(tf.int64, name='n_conn_nodes', shape=None)
            self.placeholders['n_node_ids'] = tf.placeholder(tf.int32, name='n_node_ids', shape=None)
            self.placeholders['batch_density'] = tf.placeholder(tf.float32, name='batch_density', shape=None)

            ## self.placeholders['f_indices'] = tf.placeholder(tf.int64, name='X_indices', shape=None)
            ## self.placeholders['f_data'] = tf.placeholder(tf.float32, name='X_data', shape=None)
            ## self.placeholders['f_shape'] = tf.placeholder(tf.int64, name='X_shape', shape=None)

            ## self.placeholders['nnz_features'] = tf.placeholder(tf.int32, name='nnz_features', shape=None)
            ## self.placeholders['targets'] = tf.placeholder(tf.float32, name='Targets', shape=[None, self.config.n_labels])
            ## self.placeholders['labeled_ids'] = tf.placeholder(tf.int32, name='labeled_ids', shape=[None])

            self.placeholders['batch_ids'] = tf.placeholder(tf.int32, name='batch_ids', shape=[None])

            self.placeholders['adj_indices'] = tf.placeholder(tf.int64, name='adj_indices', shape=None)
            self.placeholders['adj_data'] = tf.placeholder(tf.float32, name='adj_data', shape=None)
            self.placeholders['adj_shape'] = tf.placeholder(tf.int64, name='adj_shape', shape=None)

    def get_placeholders(self):
        with tf.variable_scope('Placeholders'):

            ## self.placeholders['dropout_in'] = tf.placeholder_with_default(0., name='dropout_in', shape=())
            ## self.placeholders['dropout_out'] = tf.placeholder_with_default(0., name='dropout_out', shape=())
            ## self.placeholders['dropout_conv'] = tf.placeholder_with_default(0., name='dropout_conv', shape=())
            self.placeholders['lr'] = tf.placeholder_with_default(0.01, name='learning_rate', shape=())
            self.placeholders['is_training'] = tf.placeholder(tf.bool, name='is_training')
            self.get_queue_placeholders()

    def test(self, sess):
        data = 'train'
        t = threading.Thread(target=self.load_and_enqueue, args=[sess, data])
        t.daemon = True
        t.start()

        _, _, _, total_steps = self.dataset.get_data(data)
        for step in range(total_steps):
            shapes = sess.run([self.model.values])
            # shapes = sess.run([self.model.layers[1].shapes])
            # print(np.shape(shapes))
            # print(shapes[1])
            print(np.shape(shapes))
        exit()
        # var, names = [v, v.name for v in tf.trainable_variables()]
        # grads = tf.gradients(self.model.loss, self.data['embeddings'])
        # sess.run[grads]
        # print(names)
        # sess.run(tf.initialize_local_variables())

    def get_test_predictions(self, sess):
        data = 'test'
        # Start Running Queue
        t = threading.Thread(target=self.load_and_enqueue, args=[sess, data])
        t.daemon = True
        t.start()

        metrics = {}
        metrics['loss'] = metrics['mc_accuracy'] = metrics['ml_accuracy'] = metrics['micro_f1'] = metrics['macro_f1'] \
            = metrics['bae'] = metrics['k_micro_f1'] = metrics['k_macro_f1'] = 0
        _, n_nodes, _, total_steps = self.dataset.get_data(data)

        for step in range(total_steps):
            # preds, labels, bd, loss, t_metrics = sess.run([self.test_predictions, self.data['targets'], self.data['batch_density'], self.model.ce_loss, self.model.metric_values])
            bd, loss, t_metrics = sess.run([self.data['batch_density'], self.model.ce_loss, self.model.metric_values])
            # _, te_metrics = evaluate(preds, labels)
            te_metrics = {'micro_f1': 0, 'macro_f1': 0}
            contrib_ratio = bd
            metrics['loss'] += loss
            metrics['mc_accuracy'] += t_metrics['mc_accuracy'] * contrib_ratio
            metrics['ml_accuracy'] += t_metrics['ml_accuracy'] * contrib_ratio
            metrics['micro_f1'] += t_metrics['micro_f1'] * contrib_ratio
            metrics['macro_f1'] += t_metrics['macro_f1'] * contrib_ratio
            metrics['bae'] += t_metrics['bae'] * contrib_ratio
            metrics['k_micro_f1'] += te_metrics['micro_f1'] * contrib_ratio
            metrics['k_macro_f1'] += te_metrics['macro_f1'] * contrib_ratio
        return metrics

    def update_global_predictions_truth(self, sess):
        data = 'train'
        # Start Running Queue
        t = threading.Thread(target=self.load_and_enqueue, args=[sess, data])
        t.daemon = True
        t.start()

        _, _, _, total_steps = self.dataset.get_data(data)
        for step in range(total_steps):
            sess.run([self.update_truth_predictions_op])

    def update_global_predictions(self, sess):
        # TODO data = all - train
        # TODO do a residual label update or Entropy based update
        # TODO Should we try a vanilla label propagation after first level of training - NO training after OE - 1
        data = 'all'
        # Start Running Queue
        t = threading.Thread(target=self.load_and_enqueue, args=[sess, data])
        t.daemon = True
        t.start()

        _, _, _, total_steps = self.dataset.get_data(data)
        for step in range(total_steps):
            sess.run([self.update_predictions_op])


    def run_epoch(self, sess, data, learning_rate, summary_writer=None, epoch_id=0, verbose=1):
        if data == 'train':
            train_op = self.model.opt_op
            train_op_pos = self.model.opt_op_pos
            train_op_neg = self.model.opt_op_neg
            feed_dict = {self.placeholders['lr']: learning_rate,
                         self.placeholders['is_training']: True}
        else:
            train_op = tf.no_op()
            train_op_neg = tf.no_op()
            train_op_pos = tf.no_op()
            feed_dict = {self.placeholders['is_training']: False}

        # Start Running Queue
        t = threading.Thread(target=self.load_and_enqueue, args=[sess, data])
        t.daemon = True
        t.start()

        metrics = {}
        metrics['loss'] = 0
        metrics['pos_loss'] = 0
        metrics['neg_loss'] = 0
        _, _, _, total_steps = self.dataset.get_data(data)

        for step in range(total_steps):
            #preds, labels, bd, loss, t_metrics = sess.run([self.test_predictions, self.data['targets'], self.data['batch_density'], self.model.ce_loss, self.model.metric_values])
                        
            bd, loss, pos_loss, neg_loss, hop_loss, _ = sess.run([self.data['batch_density'], self.model.loss, self.model.pos_loss, self.model.neg_loss, self.model.hop_loss[1:], train_op], feed_dict=feed_dict)
            
            #print('neg_loss')
            #bd, loss, pos_loss, neg_loss, hop_loss, _ = sess.run([self.data['batch_density'], self.model.loss, self.model.pos_loss, self.model.neg_loss, self.model.hop_loss[1:], train_op_neg], feed_dict=feed_dict) 

            #bd, loss, pos_loss, neg_loss, hop_loss, _ = sess.run([self.data['batch_density'], self.model.loss, self.model.pos_loss, self.model.neg_loss, self.model.hop_loss[1:], train_op], feed_dict=feed_dict)
            #print('aaaay')
            #_ = sess.run([train_op], feed_dict=feed_dict)
            #_ = sess.run([train_op], feed_dict=feed_dict)
            contrib_ratio = bd
            metrics['loss'] += loss
            metrics['pos_loss'] += pos_loss
            metrics['neg_loss'] += neg_loss
            metrics['hop_loss'] = hop_loss

        t.join()

        metrics['loss'] /= total_steps
        metrics['pos_loss'] /= total_steps
        metrics['neg_loss'] /= total_steps
        return metrics

    def fit(self, outer_epoch, sess, summary_writers):
        max_patience = self.config.patience
        patience = max_patience
        pat_increase = self.config.patience_increase
        pat_threshold = self.config.improvement_threshold

        lr = self.config.learning_rate
        suffix = '_' + self.config.train_percent + '_' + self.config.train_fold

        tot_val = []
        flag = False

        for epoch_id in range(self.config.max_inner_epochs):
            t_test = time.time()
            tr_metrics = self.run_epoch(sess, 'train', lr, summary_writers['train'], epoch_id=epoch_id)

            print(epoch_id, tr_metrics['loss'], tr_metrics['pos_loss'], tr_metrics['neg_loss'], tr_metrics['hop_loss'], patience, round(time.time() - t_test, 5))

            if epoch_id % self.config.val_epochs_freq == 0:
                val_metrics = self.run_epoch(sess, 'val', 0, summary_writers['val'], epoch_id=epoch_id)
                val_loss = val_metrics['loss']
                tot_val.append(val_loss)

                if epoch_id % 1 == 0:
                    print(epoch_id, val_loss, patience, round(time.time() - t_test, 5))

                if epoch_id > self.config.save_epochs_after:
                    if patience < 3:
                        print("aaay, possible infinte loop __main__ line 307")
                        self.saver.restore(sess, self.config.paths['ckpt' + suffix] + 'inner-last-best')
                        break
                    if not flag and tot_val[-1] > np.mean(tot_val[-(int(patience/2)):]):
                        self.saver.save(sess, self.config.paths['ckpt' + suffix] + 'inner-last-best')
                        flag = True
                    elif flag and tot_val[-1] > np.mean(tot_val[-patience:]):
                        lr = round(lr/2, 4)
                        patience = int(patience/3)

                        self.saver.restore(sess, self.config.paths['ckpt' + suffix] + 'inner-last-best')
                        flag = False

                    else:
                        print("aaay, __main__ condition doesnt check out line 307")

        tr_metrics['final_embeddings'] = sess.run(self.embeddings)

        np.savetxt('embedz', tr_metrics['final_embeddings'])

        return epoch_id, tr_metrics,  val_metrics

    def fit_outer(self, sess, summary_writers):
        max_o_epochs = self.config.max_outer_epochs

        sess.run(tf.local_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)
        epoch_id, tr_metrics, val_metrics, te_metrics = [{}]*max_o_epochs, [{}]*max_o_epochs, [{}]*max_o_epochs, [{}]*max_o_epochs

        with sess.as_default():
            # self.test(sess)
            for id in range(1):
                epoch_id[id], tr_metrics[id], val_metrics[id] = self.fit(id, sess, summary_writers)

        self.coord.request_stop()
        self.coord.join(threads)

        return epoch_id, tr_metrics, val_metrics, te_metrics

    def add_summaries(self, sess):
        # Instantiate a SummaryWriter to output summaries and the Graph.
        suffix = '_' + self.config.train_percent + '_' + self.config.train_fold
        summary_writer_train = tf.summary.FileWriter(self.config.paths['logs' + suffix] + "train", sess.graph)
        summary_writer_val = tf.summary.FileWriter(self.config.paths['logs' + suffix] + "validation", sess.graph)
        summary_writer_test = tf.summary.FileWriter(self.config.paths['logs' + suffix] + "test", sess.graph)
        summary_writers = {'train': summary_writer_train, 'val': summary_writer_val, 'test': summary_writer_test}
        return summary_writers

def init_model(config):
    tf.reset_default_graph()
    tf.set_random_seed(1234)
    np.random.seed(1234)

    # Load data
    dataset = Dataset(config)

    with tf.variable_scope('Graph_Convolutional_Network', reuse=None):
        model = OuterPropagation(dataset)

    # configure GPU usage
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    tf_config.inter_op_parallelism_threads = 32  # #CPUs - how many ops to run in parallel [0 - Default]
    tf_config.intra_op_parallelism_threads = 1  # how many threads each op gets
    sm = tf.train.SessionManager()

    if config.retrain:
        print("Loading model from checkpoint")
        load_ckpt_dir = config.ckpt_dir
    else:
        # print("No model loaded from checkpoint")
        load_ckpt_dir = ''
    sess = sm.prepare_session("", init_op=model.init, saver=model.saver, checkpoint_dir=load_ckpt_dir, config=tf_config)
    return model, sess


def dump_results(config, i_epoch, tr_metrics, val_metrics, te_metrics):
    headers = ['O_EPOCH', 'I_EPOCH', 'TR_F1', 'VAL_LOSS']
    values = []

    max_oe = config.max_outer_epochs
    if max_oe > 1:
        max_oe += 2
    for i in range(max_oe):
        values.append(np.around([i, i_epoch[i], tr_metrics[i]['loss'], val_metrics[i]['loss']], decimals=5))

    stats = tabulate(values, headers)
    print(stats)
    suffix = '_' + config.train_percent + '_' + config.train_fold
    file_name = config.paths['results'+suffix]+'metrics.txt'
    np.savetxt(file_name, values, header=str(headers), comments='', fmt='%1.5f')

    return values


def train_model(cfg):
    config = deepcopy(cfg)
    model, sess = init_model(config)
    summary_writers = model.add_summaries(sess)
    i_epochs, tr_metrics, val_metrics, te_metrics = model.fit_outer(sess, summary_writers)
    return dump_results(config, i_epochs, tr_metrics, val_metrics, te_metrics)

def main():

    args = Parser().get_parser().parse_args()
    print("=====Configurations=====\n", args)
    config = Config(args)

    start = time.time()

    outer_tracking = {}
    # TODO Loading the data and graph everytime is a total waste
    headers = ['O_EPOCH', 'I_EPOCH', 'TR_F1', 'VAL_LOSS', 'VAL_F1', 'k-MICRO-F1', 'k-MACRO-F1', 'MICRO-F1', 'MACRO-F1', 'MC_ACC', 'ML_ACC', 'BAE']
    perc_results = [[]]*len(config.train_percents)
    for perc_id, train_percent in enumerate(config.train_percents):
        print('\n\n############################  Percentage: ', train_percent, '#####################################')
        config.train_percent = train_percent
        fold_results = [[]]*len(config.train_folds)
        for fold_id, fold in enumerate(config.train_folds):
            print('\n------- Fold: ', fold)
            config.train_fold = fold
            values = train_model(config)
            outer_tracking[fold_id] = values
            fold_results[fold_id] = values[-1]
            if not config.save_model:
                remove_directory(config.paths['perc_' + train_percent] + '_' + fold)

        fold_results = np.vstack(fold_results)
        file_name = os.path.join(config.paths['perc_' + train_percent], 'metrics.txt')
        np.savetxt(file_name, fold_results, header=str(headers), comments='', fmt='%1.5f')

        perc_results[perc_id] = np.mean(fold_results, axis=0)
        if not config.save_model:
            remove_directory(config.paths['perc_' + train_percent])

    results = np.vstack(perc_results)
    file_name = os.path.join(config.paths['experiment'], 'metrics.txt')
    np.savetxt(file_name, results, header=str(headers), comments='', fmt='%1.5f')

    from os import path
    np.save(path.join(config.paths['experiment'], config.dataset_name+str(config.max_depth)+'_batch_results.npy'), outer_tracking)

    # TODO code inference - Load model and run test
    print('Time taken:', time.time() - start)

if __name__ == "__main__":
    main()
