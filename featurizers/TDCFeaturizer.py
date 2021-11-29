import tensorflow as tf
import numpy as np
import datetime
import random

from .BaseFeaturizer import BaseFeaturizer
from .residual_block import residual_block


class TDCFeaturizer(BaseFeaturizer):
    """Temporal Distance Classification featurizers

    Reference: "Playing hard exploration games by watching YouTube"
    The unsupervised task consists of presenting the network with 2 frames separated by n timesteps,
    and making it classify the distance between the frames.

    We use the same network architecture as the paper:
        3 convolutional layers, followed by 3 residual blocks,
        followed by 2 fully connected layers for the encoder.

        For the classifier, we do a multiplication between both feature vectors
        followed by a fully connected layer.
    """

    def __init__(self, initial_width, initial_height, desired_width, desired_height, feature_vector_size=1024,
                 learning_rate=0.0001, experiment_name='default'):
        super().__init__()
        print("Starting featurizers initialization")
        self.sess = tf.Session()
        self.graph = self._generate_featurizer(initial_width, initial_height, desired_width, desired_height,
                                               feature_vector_size, learning_rate)
        self.saver = tf.train.Saver()
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.summary_writer = tf.summary.FileWriter('./summaries/{}/{}/'.format(experiment_name, timestamp),
                                                    tf.get_default_graph())
        self.summary_writer.flush()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def _generate_featurizer(self, initial_width, initial_height, desired_width, desired_height, feature_vector_size,
                             learning_rate):
        """Builds the TensorFlow graph for the featurizers

        Args:
            initial_width: The images' width before cropping.
            initial_height: The images' height before cropping.
            desired_width: Target width after cropping.
            desired_height: Target height after cropping.
            feature_vector_size: Length of the feature vector.
            learning_rate: Step size for the learning algorithm.
        Returns:
            graph object that contains the public nodes of the model.
        """
        is_training = tf.placeholder(dtype=tf.bool, shape=(), name="is_training")
        stacked_state = tf.placeholder(dtype=tf.float32, shape=(None, 2, initial_height, initial_width, 3),
                                       name="stacked_state")
        labels = tf.placeholder(dtype=tf.float32, shape=(None, 6), name="labels")  # There are 6 possible labels

        state = tf.reshape(stacked_state, (-1, initial_height, initial_width, 3))
        state = tf.random_crop(state, (tf.shape(state)[0], desired_height, desired_width, 3))

        with tf.variable_scope('Encoder'):
            x = state
            for filters, strides in [(8, 2)]:
                x = self._convolutional_layer(x, filters, strides, is_training)
            for i in range(1):
                x = residual_block(x, 8, 8, 1, is_training)

            x = tf.layers.flatten(x)
            x = tf.layers.dense(
                x,
                feature_vector_size,
                tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                name="fc1"
            )
            # Compose feature vector
            feature_vector = tf.layers.dense(
                x,
                feature_vector_size,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                name="fv"
            )
            feature_vector = tf.nn.l2_normalize(feature_vector, -1)
            feature_vector_stack = tf.reshape(feature_vector, (-1, 2, feature_vector_size))

        with tf.variable_scope('Classifier'):
            combined_embeddings = tf.multiply(feature_vector_stack[:, 0, :], feature_vector_stack[:, 1, :])

            x = tf.layers.dense(
                combined_embeddings,
                feature_vector_size,
                tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                name="fc1"
            )

            prediction = tf.layers.dense(
                x,
                6,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                name="prediction"
            )

        # Losses and optimizers
        with tf.variable_scope('losses', reuse=False):
            classifier_loss = tf.losses.softmax_cross_entropy(labels, prediction)
            accuracy, accuracy_update_op = tf.metrics.accuracy(
                labels=tf.argmax(labels, -1),
                predictions=tf.argmax(prediction, -1)
            )

            # We need to add the batch normalization update ops as dependencies
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(classifier_loss)

        classifier_loss_summary = tf.summary.scalar('classifier_loss', classifier_loss)
        accuracy_summary = tf.summary.scalar('accuracy', accuracy_update_op)
        summaries = tf.summary.merge([classifier_loss_summary, accuracy_summary])

        return {'is_training': is_training,
                'state': state,
                'labels': labels,
                'stacked_state': stacked_state,
                'feature_vector': feature_vector,
                'prediction': prediction,
                'loss': classifier_loss,
                'train_op': train_op,
                'summaries': summaries,
                'accuracy_update_op': accuracy_update_op}

    def _convolutional_layer(self, inputs, filters, strides, is_training):
        """Constructs a conv2d layer followed by batch normalization, and max pooling"""
        x = tf.layers.conv2d(
            inputs,
            filters=filters,
            kernel_size=(3, 3),
            strides=strides,
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001)
        )
        x = tf.layers.batch_normalization(x, training=is_training)
        return tf.layers.max_pooling2d(x, 2, 2)

    def train(self, dataset, epochs, batch_size):
        """Runs the training algorithm"""
        from tqdm import tqdm
        print("Starting training procedure")
        for epoch in tqdm(range(1, epochs+1)):
            # frames == state
            frames, labels = self._generate_training_data(dataset, batch_size)
            print(frames.shape, labels.shape)

            _, train_summary, accuracy = self.sess.run(
                [self.graph['train_op'], self.graph['summaries'], self.graph['accuracy_update_op']],
                feed_dict={self.graph['is_training']: True, self.graph['stacked_state']: frames,
                           self.graph['labels']: labels})
            if epoch % 1 == 0:
                print('Epoch: {}/{}'.format(epoch, epochs))
                self.summary_writer.add_summary(train_summary, epoch)
        return True

    def _generate_training_data(self, videos, number_of_samples):
        """
        Constructs the unsupervised task
        
        Input radarTracks: two frames that are a number of time steps apart
        Output: classify how many frames there are between the frames.
        """
        video = videos[random.randint(0, len(videos) - 1)]

        [video_len, *video_shape] = video.shape

        frames = np.empty((number_of_samples, 2, *video_shape))
        labels = np.zeros((number_of_samples, 6))

        for i in range(number_of_samples):
            interval = random.randint(0, 5)
            if interval == 0:
                possible_frames_start = 0
                possible_frames_end = 0
            elif interval == 1:
                possible_frames_start = 1
                possible_frames_end = 1
            elif interval == 2:
                possible_frames_start = 2
                possible_frames_end = 2
            elif interval == 3:
                possible_frames_start = 3
                possible_frames_end = 4
            elif interval == 4:
                possible_frames_start = 5
                possible_frames_end = 20
            else:
                possible_frames_start = 21
                possible_frames_end = 60

            first_frame_index = random.randint(0, video_len - possible_frames_end - 1)
            second_frame_index = random.randint(possible_frames_start, possible_frames_end)

            frames[i, 0] = video[first_frame_index] / 255  # To normalize radarTracks
            frames[i, 1] = video[first_frame_index + second_frame_index] / 255
            labels[i, interval] = 1.

        return frames, labels
