import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.tensorboard.plugins import projector

from parser import Parser
from config import Config

args = Parser().get_parser().parse_args()
config = Config(args)

embeddings_path = config.embeddings_path
visualisation_directory = config.visualisation_directory

embeddings = np.genfromtxt(embeddings_path, dtype=np.float32, delimiter=' ')
labels = np.reshape(np.argmax(np.load('labels.npy'), axis=1), (2708, 1))

tf_embeddings = tf.Variable(embeddings, name='embeddings')

os.system("rm -r " + visualisation_directory + ";" + " mkdir " + visualisation_directory)
os.system("touch " + os.path.join(visualisation_directory, 'metafile.tsv'))
os.system("rm " + os.path.join(visualisation_directory, 'metafile.tsv'))

with open(os.path.join(visualisation_directory, 'metafile.tsv'), 'w') as metafile:
    for i in labels:
        metafile.write(str(i)[1] + '\n')


with tf.Session() as sess:
    saver = tf.train.Saver([tf_embeddings])

    sess.run(tf_embeddings.initializer)

    config = projector.ProjectorConfig()

    to_vis = config.embeddings.add()

    to_vis.tensor_name = tf_embeddings.name

    to_vis.metadata_path = os.path.join(visualisation_directory, 'metafile.tsv')

    saver.save(sess, os.path.join(visualisation_directory, 'embeddings.ckpt'))

    projector.visualize_embeddings(tf.summary.FileWriter(visualisation_directory), config)

print('embedding visualisation done')


