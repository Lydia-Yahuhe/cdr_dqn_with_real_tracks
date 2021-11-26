import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

from featurizers import TDCFeaturizer


def visualize_embeddings(embeddings, experiment_name='default'):
    """Save the embeddings to be visualised using t-sne on TensorBoard
    Based on https://medium.com/@vegi/visualizing-higher-dimensional-data-using-t-sne-on-tensorboard-7dbf22682cf2
    """
    tf_embeddings = tf.Variable(np.concatenate(embeddings, 0))

    # Generate metadata
    metadata = 'video_index\tframe_index\n'
    for video_index in range(len(embeddings)):
        for frame_index in range(embeddings[video_index].shape[0]):
            metadata += '{}\t{}\n'.format(video_index, frame_index)

    metadata_path = 'embeddings/{}/labels.tsv'.format(experiment_name)
    with open(metadata_path, 'w') as metadata_file:
        metadata_file.write(metadata)

    with tf.Session() as sess:
        saver = tf.train.Saver([tf_embeddings])
        sess.run(tf_embeddings.initializer)
        saver.save(sess, 'embeddings/{}/embeddings.ckpt'.format(experiment_name))
        config = projector.ProjectorConfig()

        embedding = config.embeddings.add()

        embedding.tensor_name = tf_embeddings.name
        embedding.metadata_path = metadata_path.split('/')[-1]

        projector.visualize_embeddings(tf.summary.FileWriter('embeddings/{}'.format(experiment_name)), config)


def visualize(dataset, load_path, feature_class=TDCFeaturizer):
    initial_width = 92
    initial_height = 92
    desired_width = 84
    desired_height = 84
    learning_rate = 0.0001
    featurizer = feature_class(initial_width, initial_height, desired_width, desired_height,
                               feature_vector_size=1024, learning_rate=learning_rate, experiment_name='default')
    featurizer.load(load_path)
    features_all = [featurizer.featurize(data) for data in dataset]
    visualize_embeddings(features_all)
