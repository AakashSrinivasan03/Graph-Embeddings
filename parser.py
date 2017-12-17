
import argparse
import numpy as np
from datetime import datetime


# TODO # Negative Contraints
class Parser(object):  #
    def __init__(self):
        parser = argparse.ArgumentParser()

        ##parser.add_argument("--label_loss", default=1, help="Supervised learning weightage", type=self.str2bool)
        ##parser.add_argument("--transductive", default=False, help="Transductive learning", type=self.str2bool)

        # Graph strucuture regularizer

        ##parser.add_argument("--regKernel_loss", default=0, help="weightage", type=int)
        ##parser.add_argument("--regKernel_var", default=1, help="H.T(var.L).H", type=float)
        ##parser.add_argument("--regKernel_order", default=2, help="pow(R,?)", type=int)
        ##parser.add_argument("--regKernel_dim", default=64, type=int)
        ##parser.add_argument("--regKernel_residual", default=False, type=self.str2bool)
        ##parser.add_argument("--regKernel_metric_learning", default=False, type=self.str2bool)  # DNT
        parser.add_argument("--save_embeddings", default=False, help="save embeddings for unsupervised model", type=self.str2bool)
        parser.add_argument("--loss_k_depth", default=5, help="number of hops to consider in loss term", type=int)
        parser.add_argument("--reg_weight", default=10, help="weight to regulariser", type=np.float32)
        parser.add_argument("--max_depth", default=5, help="Maximum path depth", type=int)
        parser.add_argument("--lambdas", default='0.2, 0.2, 0.2, 0.2, 0.2', help="Attention for each hop:Comma seperated")
        parser.add_argument("--loss_fun_type", default='linear', help="linear or exponential loss function ?", choices=['linear', 'exponential'])
        parser.add_argument("--visualisation_directory", default='tensorboard_vis', help='directory to store data for visualisation')
        parser.add_argument("--embeddings_path", default='embedz', help='path to file with embeddings')
        parser.add_argument("--deep_embeddings_path", default='embedz_deep', help='path to file to store embeddings in deep embeddings format')
        parser.add_argument("--normalise_pos", default='trace', help='normalise positive loss or not ?')
        parser.add_argument("--normalise_neg", default='none', help='normalise negative loss or not ?')
        parser.add_argument('--laplacian', default='lk', help = 'lk or lpowerk')

        # Node attribute Aggregator
        ##parser.add_argument("--propModel", default='propagation_gated', help='propagation model names',
        ##                    choices=['propagation', 'propagation_gated', 'propagation_label'])
        ##parser.add_argument("--aggKernel", default='simple', help="kerel names",
        ##                    choices=['kipf', 'simple', 'attention1', 'mul_attention', 'add_attention', 'embmul_attention',
        ##                             'keyval_attention', 'muladd_attention', 'maxpool', 'mul_attention2', 'mul_attention3'])
        parser.add_argument("--featureless", default=True, help="Non-attributed graphs", type=self.str2bool)
        ##parser.add_argument("--node_features", default='-', help="x,h")
        ##parser.add_argument("--neighbor_features", default='h', help="x,h")
        ##parser.add_argument("--dims", default='128,128,128,128', help="Dimensions of hidden layers: comma separated")
        ##parser.add_argument("--skip_connections", default=True, help="also l2 norm on h_T and output layer added", type=self.str2bool)
        ##parser.add_argument("--shared_weights", default=0, type=int)
        #parser.add_argument("--bias", default=False, type=self.str2bool)
        ##parser.add_argument("--sparse_features", default=True, help="For current datasets - manually set in config.py", type=self.str2bool)

        # Node attributes pertubation
        ##parser.add_argument("--add_noise", default=0, help="Add noise to input attributes", type=float, choices=np.round(np.arange(0, 1, 0.1),1))
        ##parser.add_argument("--drop_features", default=0, help="Range 0-1", type=float, choices=np.round(np.arange(0, 1, 0.1),1))

        # Structure pertubation
        parser.add_argument("--neighbors", default='-1', help="Number of neighbors at each depth; comma separated")
        parser.add_argument("--drop_edges", default=0., help="Randomly drop edges at each depth", type=float, choices=np.round(np.arange(0, 1, 0.1), 1))

        # Dataset Details
        parser.add_argument("--dataset", default='cora', help="Dataset to evluate | Check Datasets folder",
                            choices=['cora', 'citeseer', 'wiki', 'amazon', 'facebook', 'cora_multi', 'movielens',
                                    'ppi_sg', 'blogcatalog', 'genes_fn', 'mlgene', 'ppi_gs', 'reddit', 'reddit_ind'])
        #parser.add_argument("--labels", default='labels_random', help="Label Sampling Type")
        parser.add_argument("--percents", default='10', help="Training percent comma separated, ex:5,10,20")
        parser.add_argument("--folds", default='1', help="Training folds comma separated")
        parser.add_argument("--qcap", default='5', help="queue size", type=int)
        parser.add_argument("--embed_dims", default='256', help="dimensions in which to embed", type=int)

        # NN Hyper parameters
        parser.add_argument("--batch_size", default=-1, help="Batch size", type=int)
        #parser.add_argument("--wce", default=True, help="Weighted cross entropy", type=self.str2bool)
        parser.add_argument("--lr", default=0.05, help="Learning rate", type=float)
        ##parser.add_argument("--l2", default=1e-6, help="L2 loss", type=float)
        parser.add_argument("--opt", default='adam', help="Optimizer type", choices=['adam', 'sgd', 'rmsprop'])
        parser.add_argument("--which_degree", default='batch_adj_mat', help = "use degree of which matrix ?", choices=['global_adj_mat', 'batch_adj_mat'])
        ##parser.add_argument("--drop_in", default=0., help="Dropout for input", type=float, choices=np.round(np.arange(0, 1, 0.05),2))
        ##parser.add_argument("--drop_conv", default=0., help="Dropout for output - currently set to drop_conv in config.py", type=float, choices=np.round(np.arange(0, 1, 0.05),2))
        ##parser.add_argument("--drop_out", default=0., help="Dropout for output - currently set to drop_conv in config.py", type=float, choices=np.round(np.arange(0, 1, 0.05),2))


        # Training parameters
        parser.add_argument("--retrain", default=False, type=self.str2bool, help="Retrain flag")
        parser.add_argument("--gpu", default=0, help="GPU BUS ID ", type=int)
        parser.add_argument("--verbose", default=0, help="Verbose mode", type=int, choices=[0, 1, 2])
        parser.add_argument("--save_model", default=False, type=self.str2bool)

        ##parser.add_argument("--cautious_updates", default=True, type=self.str2bool)
        parser.add_argument("--max_outer", default=1, help="Maximum outer ", type=int)
        parser.add_argument("--max_inner", default=500, help="Maximum inner epoch", type=int)
        ##parser.add_argument("--lu", default=0.8, help="Label update rate", type=float)

        parser.add_argument("--pat", default=15, help="Patience", type=int)
        parser.add_argument("--pat_inc", default=2, help="Patience Increase", type=int)
        parser.add_argument("--pat_improve", default=.9999, help="Improvement threshold for patience", type=float)
        parser.add_argument("--save_after", default=30, help="Save after epochs", type=int)
        parser.add_argument("--val_freq", default=10, help="Validation frequency", type=int)
        parser.add_argument("--summaries", default=True, help="Save summaries after each epoch", type=self.str2bool)

        now = datetime.now()
        timestamp = str(now.month) + '|' + str(now.day) + '|' + str(now.hour) + ':' + str(now.minute) + ':' + str(
            now.second)
        parser.add_argument("--timestamp", default=timestamp, help="Timestamp to prefix experiment dumps")
        parser.add_argument("--folder_suffix", default='Default', help="folder name suffix")

        # TODO Load saved model and saved argparse
        self.parser = parser


    def str2bool(self, text):
        if text == 'True':
            arg = True
        elif text == 'False':
            arg = False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        return arg


    def get_parser(self):
        return self.parser
