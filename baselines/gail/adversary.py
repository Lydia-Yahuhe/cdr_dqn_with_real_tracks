"""
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
"""
import tensorflow as tf
import numpy as np

from baselines import logger
from baselines.common.mpi_adam import MpiAdam
from baselines.common import tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd


def logsigmoid(a):
    """Equivalent to tf.log(tf.sigmoid(a))"""
    return -tf.nn.softplus(-a)


"""
Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51
"""


def logit_bernoulli_entropy(logits):
    ent = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent


class TransitionClassifier(object):
    def __init__(self, env, hidden_size, entcoeff=0.001, scope="adversary"):
        self.scope = scope

        self.generator_obs_ph = tf.placeholder(tf.float32, (None, 256), name="observations_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None, 256), name="expert_observations_ph")

        # Build graph
        generator_logits = self.__build_graph(self.generator_obs_ph, hidden_size=hidden_size)
        expert_logits = self.__build_graph(self.expert_obs_ph, hidden_size=hidden_size, reuse=True)

        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) <= 0.001))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) >= 0.999))

        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits, labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)

        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff * entropy

        # Loss + Accuracy terms
        self.total_loss = generator_loss + expert_loss + entropy_loss
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc, self.total_loss]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc", "total_loss"]

        # Build Reward for policy
        # self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
        self.reward_op = -(1 - tf.nn.sigmoid(generator_logits))

        var_list = self.get_trainable_variables()
        self.adam = MpiAdam(var_list, epsilon=1e-5)
        self.lossandgrad = U.function(
            [self.generator_obs_ph, self.expert_obs_ph],
            self.losses + [U.flatgrad(self.total_loss, var_list)])

    def __build_graph(self, obs_ph, hidden_size=128, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            x = obs_ph
            x = tf.contrib.layers.fully_connected(x, hidden_size, activation_fn=tf.nn.relu)
            x = tf.contrib.layers.fully_connected(x, hidden_size, activation_fn=tf.nn.relu)
            logits = tf.contrib.layers.fully_connected(x, 1, activation_fn=tf.identity)
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_reward(self, obs):
        sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        feed_dict = {self.generator_obs_ph: obs}
        reward = sess.run(self.reward_op, feed_dict)
        return reward


# if __name__ == '__main__':
#     from tqdm import tqdm
#
#     from fltenv.env import ConflictEnv
#     from baselines.gail.dataset import generate_dataset
#
#     env = ConflictEnv(limit=0)
#
#     print(env.picture_size)
#     dataset = generate_dataset('../../dataset/env_train.avi', env.picture_size)
#
#     reward_giver = TransitionClassifier(env, 128, entcoeff=0.1)
#     adam = reward_giver.adam
#
#     U.initialize()
#     adam.sync()
#
#     num_steps = int(2e5)
#     num_actions = env.action_space.n
#     actions = list(range(num_actions))
#
#     col = []
#     for iter_so_far in tqdm(range(1, num_steps+1)):
#         ob_expert = dataset.get_next_batch(32)
#         ob_batch = ob_expert[:, :]
#         *train_loss, g = reward_giver.lossandgrad(ob_batch, ob_expert)
#         adam.update(g, 1e-4)
#
#         # print(train_loss)
#         col.append(train_loss)
#
#         if iter_so_far % 100 == 0:
#             train_loss = np.mean(col, axis=0)
#             print(train_loss.shape)
#             logger.record_tabular("step", iter_so_far)
#             logger.record_tabular("g_loss", train_loss[0])
#             logger.record_tabular("e_loss", train_loss[1])
#             logger.record_tabular("entropy", train_loss[2])
#             logger.record_tabular("entropy_loss", train_loss[3])
#             logger.record_tabular("g_acc", train_loss[4])
#             logger.record_tabular("e_acc", train_loss[5])
#             logger.record_tabular("t_loss", train_loss[6])
#
#             ob_expert = dataset.get_next_batch(-1)
#             ob_batch = ob_expert[:, :]
#
#             rewards_expert = reward_giver.get_reward(ob_expert)
#             logger.record_tabular("rewards_expert", np.mean(rewards_expert))
#
#             rewards_batch = reward_giver.get_reward(ob_batch)
#             logger.record_tabular("rewards_batch", np.mean(rewards_batch))
#
#             print(rewards_expert.shape, rewards_batch.shape)
#
#             logger.dump_tabular()
#
#             if iter_so_far % 10000 == 0:
#                 U.save_variables('discriminator_{}'.format(num_steps), variables=reward_giver.get_variables())
