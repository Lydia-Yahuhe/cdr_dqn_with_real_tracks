import datetime
import time

import tensorflow as tf
import numpy as np

from .BaseFeaturizer import BaseFeaturizer


class VAEFeaturizer(BaseFeaturizer):
    # Variational Auto Encoder featurizers
    def __init__(self, initial_width, initial_height, learning_rate=0.0001):
        super().__init__()
        self.feature_vector_size = 64
        self.hidden_size = 64

        print("Starting featurizers initialization")
        self.sess = tf.Session()
        self.lr = learning_rate
        self.graph = self._generate_featurizer(initial_width, initial_height)
        self.saver = tf.train.Saver()

        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.summary_writer = tf.summary.FileWriter('.\\dataset\\summaries\\{}\\'.format(timestamp),
                                                    tf.get_default_graph())
        self.summary_writer.flush()

        print("About to initialize vars")
        self.sess.run(tf.global_variables_initializer())

    def _generate_featurizer(self, width, height):
        # VAE based in https://github.com/shaohua0116/VAE-Tensorflow/blob/master/demo.py
        state = tf.placeholder(dtype=tf.float32, shape=(None, width, height), name="state")
        feature_vector_size = self.feature_vector_size

        # Encoder
        with tf.variable_scope('Encoder', reuse=False):
            x = state
            flattened_features = tf.layers.flatten(x)

            with tf.variable_scope('FF_encoder'):
                fc1_en = tf.layers.dense(
                    flattened_features,
                    feature_vector_size,
                    tf.nn.relu,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                    name="fc1"
                )
                feature_vector_mean = tf.layers.dense(
                    fc1_en,
                    feature_vector_size,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                    name="feature_vector_mean"
                )
                feature_vector_log_std = tf.layers.dense(
                    fc1_en,
                    feature_vector_size,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                    name="feature_vector_log_std"
                )

            normal_dist = tf.random_normal(shape=tf.shape(feature_vector_log_std), mean=0, stddev=1, dtype=tf.float32)
            feature_vector = feature_vector_mean + tf.exp(feature_vector_log_std) * normal_dist

        # Decoder
        with tf.variable_scope('Decoder', reuse=False):
            decoder_input = feature_vector

            with tf.variable_scope('FF_decoder'):
                fc1_de = tf.layers.dense(
                    decoder_input,
                    self.hidden_size,
                    tf.nn.relu,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                    name="fc1"
                )
                fc2 = tf.layers.dense(
                    fc1_de,
                    self.hidden_size,
                    tf.nn.relu,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                    name="fc2"
                )
                fc3 = tf.layers.dense(
                    fc2,
                    width * height,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                    name="fc3"
                )

            decoder_output = tf.reshape(fc3, (-1, width, height))

        # Losses and optimizers
        with tf.variable_scope('losses', reuse=False):
            reconstruction_loss = tf.squared_difference(state, decoder_output, name='reconstruction_loss')
            reconstruction_loss = tf.reduce_mean(reconstruction_loss)

            kl_loss = -0.5 * tf.reduce_sum(
                1 + feature_vector_log_std - tf.square(feature_vector_mean) - tf.exp(feature_vector_log_std),
                axis=-1
            )
            kl_loss = tf.reduce_mean(kl_loss)
            total_loss = reconstruction_loss + kl_loss

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(total_loss)

        reconstruction_loss_summary = tf.summary.scalar('Reconstruction_loss', reconstruction_loss)
        kl_loss_summary = tf.summary.scalar('KL_loss', kl_loss)
        summaries = tf.summary.merge([reconstruction_loss_summary, kl_loss_summary])

        self.loss = [total_loss, kl_loss]
        graph = {
            'state': state,
            'feature_vector': feature_vector,
            'decoder_output': decoder_output,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss,
            'train_op': train_op,
            'summaries': summaries
        }

        return graph

    def train(self, dataset, epochs, batch_size):
        train_data, test_data = dataset
        print("Starting training procedure")

        for epoch in range(epochs):
            train_summary, loss = None, None
            start_time = time.time()

            for batch_index in range(train_data.shape[0] // batch_size):
                batch = train_data[batch_index * batch_size: (batch_index + 1) * batch_size, :, :]

                _, train_summary, *loss = self.sess.run([self.graph['train_op'], self.graph['summaries']] + self.loss,
                                                        {self.graph['state']: batch})
            end_time = time.time()
            print('Epoch: {}/{}, Time: {}, Total: {}, KL: {}'.format(epoch, epochs, end_time - start_time, loss[0],
                                                                     loss[1]), end='\t')
            self.summary_writer.add_summary(train_summary, epoch)

            if epoch % 100 == 0:
                test_loss = None
                for batch_index in range(test_data.shape[0] // batch_size):
                    batch = test_data[batch_index * batch_size: (batch_index + 1) * batch_size, :, :]

                    _, *test_loss = self.sess.run([self.graph['train_op']] + self.loss,
                                                  {self.graph['state']: batch})
                print('Total: {}, KL: {}'.format(test_loss[0], test_loss[1]))
            else:
                print()
        return True

    def save(self, save_path='default.ckpt'):
        return self.saver.save(self.sess, save_path)

    def load(self, load_path='default.ckpt'):
        self.saver.restore(self.sess, load_path)

    def featurize(self, data, batch_size=32):
        split_data = np.array_split(data, max(data.shape[0] // batch_size, 1))
        feature_vectors = []
        for batch in split_data:
            feature_vector = self.sess.run(self.graph['feature_vector'], feed_dict={self.graph['state']: batch})
            feature_vectors.append(feature_vector)
        feature_vectors = np.concatenate(feature_vectors)
        return feature_vectors

    def defeaturize(self, features):
        # Features: ndarray shape = (None, feature_vector_size)
        return self.sess.run(self.graph['decoder_output'], {self.graph['feature_vector']: features})

    def reconstruct(self, data):
        # From images to reconstructed images
        return self.sess.run(self.graph['decoder_output'], {self.graph['state']: data})
