import tensorflow as tf
import sys
from yaml import dump
from os import path
from utils import utils
import numpy as np
import importlib


class Config(object):
    def __init__(self, args):

        # SET UP PATHS
        self.paths = dict()
        self.embed_dims = args.embed_dims
        self.paths['root'] = '.' #changed this from ../

        self.paths['datasets'] = path.join('../', 'Datasets') #changed from root
        self.paths['experiments'] = path.join('../', 'Experiments') #changed from root
        self.dataset_name = args.dataset
        self.paths['experiment'] = path.join(self.paths['experiments'], args.timestamp, self.dataset_name, args.folder_suffix)
        # Parse training percentages and folds
        self.train_percents = args.percents.split(',')
        self.train_folds = args.folds.split(',')
        self.lambdas = [np.float32(x) for x in args.lambdas.split(',')]
        self.loss_k_depth = args.loss_k_depth
        self.reg_weight = args.reg_weight
        self.which_degree = args.which_degree
        self.loss_fun_type = args.loss_fun_type
        self.visualisation_directory = args.visualisation_directory
        self.embeddings_path = args.embeddings_path
        self.deep_embeddings_path = args.deep_embeddings_path
        self.normalise_pos = args.normalise_pos
        self.normalise_neg = args.normalise_neg
        self.laplacian = args.laplacian

        for perc in self.train_percents:
            self.paths['perc' + '_' + perc] = path.join(self.paths['experiment'], perc)
            for fold in self.train_folds:
                suffix = '_' + perc + '_' + fold
                path_prefix = [self.paths['experiment'], perc, fold, ]
                self.paths['logs' + suffix] = path.join(*(path_prefix + ['Logs']))
                self.paths['ckpt' + suffix] = path.join(*(path_prefix + ['Checkpoints']))
                self.paths['embed' + suffix] = path.join(*(path_prefix + ['Embeddings']))
                self.paths['results' + suffix] = path.join(*(path_prefix + ['Results']))

        # Create directories
        for (key, val) in self.paths.items():
            if key not in ['root', 'experiments', 'datasets']:

                ### !!! removing [:-1]
                utils.create_directory_tree(str.split(val, sep='/'))

        dump(args.__dict__, open(path.join(self.paths['experiment'], 'args.yaml'), 'w'), default_flow_style=False, explicit_start=True)

        self.paths['data'] = path.join(self.paths['datasets'], self.dataset_name)
        #self.paths['labels'] = path.join(path.join(self.paths['data'], 'labels.npy'))
        #self.paths['features'] = path.join(path.join(self.paths['data'], 'features.npy'))
        self.paths['adjmat'] = path.join(path.join(self.paths['data'], 'adjmat.mat'))

        # -------------------------------------------------------------------------------------------------------------

        # Hidden dimensions
        ##self.dims = list(map(int, args.dims.split(',')[:args.max_depth]))
        ##if len(self.dims) < args.max_depth:
        ##    sys.exit('#Hidden dimensions should match the max depth')

        # Propogation Depth
        self.max_depth = args.max_depth

        # Drop nodes or edges
        self.drop_edges = args.drop_edges

        # Subset of neighbors to consider at max
        self.neighbors = np.array(args.neighbors.split(','), dtype=int)

        if self.neighbors.shape[0] < args.max_depth:
            #Extend as -1 is no information provided, i.e take all neighbors at that depth
            diff = args.max_depth - self.neighbors.shape[0]
            self.neighbors = np.hstack((self.neighbors, [-1]*diff))
            #sys.exit('Neighbors argument should match max depth: ex: -1,1 or -1, 32')


        # ASK PRIYESH

        kflag = 0

        for i in self.neighbors:
            if i != -1:
                kflag = 1

        if args.drop_edges != 0 and kflag == 1:
            sys.exit('Can not have drop edges and neighbors flag set at the same time')

        # GPU
        self.gpu = args.gpu

        # Data sets
        ##self.label_type = args.labels

        # Weighed cross entropy loss
        ##self.wce = args.wce

        # Retrain
        self.retrain = args.retrain

        # Metrics to compute
        ##self.metric_keys = ['accuracy', 'micro_f1', 'macro_f1', 'bae']

        # Batch size
        if args.batch_size == -1:
            self.queue_capacity = 1
        else:
            self.queue_capacity = args.qcap

        self.batch_size = args.batch_size

        # Dropouts
        ##self.drop_in = args.drop_in
        # self.drop_out = args.drop_out
        # self.drop_conv = args.drop_conv
        ##self.drop_conv = self.drop_in
        ##self.drop_out = self.drop_conv

        # Data pertubation
        ##self.drop_features = args.drop_features
        ##self.add_noise = args.add_noise

        # Number of steps to run trainer
        self.max_outer_epochs = args.max_outer
        self.max_inner_epochs = args.max_inner
        ##self.cautious_updates = args.cautious_updates

        # Save summaries
        self.summaries = args.summaries

        # Validation frequence
        self.val_epochs_freq = args.val_freq  # 1

        # Model save frequency
        self.save_epochs_after = args.save_after  # 0

        # early stopping hyper parametrs
        self.patience = args.pat  # look as this many epochs regardless
        self.patience_increase = args.pat_inc  # wait this much longer when a new best is found
        self.improvement_threshold = args.pat_improve  # a relative improvement of this much is considered significant

        self.learning_rate = args.lr
        ##self.label_update_rate = args.lu

        # optimizer
        ##self.l2 = args.l2
        ##self.l2 = args.l2

        if args.opt == 'adam':
            self.opt = tf.train.AdamOptimizer
        elif args.opt == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer
        elif args.opt == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer
        else:
            raise ValueError('Undefined type of optmizer')

        # Set Model
        ##self.kernel_class = getattr(importlib.import_module("src.layers.graph_convolutions."+args.aggKernel+"_kernel"), "Kernel")

        self.prop_class = getattr(importlib.import_module('lapprop'), "Propagation")

        # Sparse Feature settings
        ##self.sparse_features = args.sparse_features
        ##if not self.sparse_features and self.dataset_name in ['cora', 'citeseer', 'amazon', 'facebook', 'cora_multi', 'movielens',
        ##                            'ppi_sg', 'blogcatalog', 'genes_fn', 'mlgene', 'ppi_gs']:
        ##    self.sparse_features = True
        ##    print('Sparse Features turned on forcibly!')
        ##elif self.dataset_name in ['wiki', 'reddit']:
        ##    self.sparse_features = False

        # Node features
        '''self.features = ['x', 'h'] ###############
        if args.node_features == '-':
            self.n_node_features = 0
            self.node_features = ''
        else:
            self.node_features = args.node_features.split(',')
            self.n_node_features = len(self.node_features)

        if args.neighbor_features == '-':
            self.n_neigh_features = 0
            self.neighbor_features = ''
        else:
            self.neighbor_features = args.neighbor_features.split(',')
            self.n_neigh_features = len(self.neighbor_features)

        if self.n_node_features == 0 and self.n_neigh_features == 0:
            sys.exit("Both node and neigh features can't be empty")
        else:
            if self.n_node_features != 0 and np.count_nonzero(np.in1d(self.node_features, self.features)) != self.n_node_features:
                sys.exit('Invalid node features. small case \'x and \'h are only valid')
            if self.n_neigh_features != 0 and np.count_nonzero(np.in1d(self.neighbor_features, self.features)) != self.n_neigh_features:
                sys.exit('Invalid neighbor features. small case \'x and \'h are only valid')
			'''      ####################
        # Loss terms
        self.loss = {}
        ##self.loss['label'] = args.label_loss
        ##self.loss['l2'] = args.l2
        ##self.loss['regKernel'] = args.regKernel_loss

        # Regularizer Kernel
        '''self.regKernel = {}   ##############
        self.regKernel['var'] = args.regKernel_var
        self.regKernel['order'] = args.regKernel_order
        self.regKernel['dim'] = args.regKernel_dim
        self.regKernel['save_emb'] = args.save_embeddings
        self.regKernel['residual'] = args.regKernel_residual
        self.regKernel['metric_learning'] = args.regKernel_metric_learning'''###############

        self.featureless = args.featureless
        ##self.skip_connections = args.skip_connections
        ##if args.shared_weights == 1:
        ##    self.shared_weights = True
        ##else:
        ##    self.shared_weights = False
        ## self.bias = args.bias

        ##self.add_labels = False
        ##if self.max_outer_epochs > 1:
        ##    self.add_labels = True

        self.save_model = args.save_model

        ##self.transductive = args.transductive