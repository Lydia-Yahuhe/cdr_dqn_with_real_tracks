import os
import random
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)


def iterbatches(array, *, batch_size=None, shuffle=True, include_final_partial_batch=True):
    n = len(array)
    inds = np.arange(n)
    if shuffle:
        np.random.shuffle(inds)

    sections = np.arange(0, n, batch_size)[1:]
    for batch_inds in np.array_split(inds, sections):
        if include_final_partial_batch or len(batch_inds) == batch_size:
            yield array[batch_inds]


def learn(env,
          network,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          learning_starts=1000,
          gamma=1.0,
          reward_giver=None,
          expert_dataset=None,
          featurizer=None,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          load_path=None,
          **network_kwargs):
    # Create all the functions necessary to train the model

    sess = get_session()
    set_global_seeds(seed)

    q_func = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space
    num_actions = env.action_space.n

    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=num_actions,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': num_actions,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    # U.load_variables('checkpoint\\discriminator_200000_125', variables=reward_giver.get_variables())
    # print('load variables successfully!')
    update_target()

    episode_rewards, true_rewards = [0.0], [0.0]
    losses, d_losses = [], []

    obs = env.reset()
    reset = True

    if load_path is not None:
        load_variables(load_path)
        logger.log('Loaded model from {}'.format(load_path))
        return act

    for t in range(1, total_timesteps + 1):
        # Take action and update exploration to the newest value
        kwargs = {}
        if not param_noise:
            update_eps = exploration.value(t)
            update_param_noise_threshold = 0.
        else:
            update_eps = 0.
            # Compute the threshold such that the KL divergence between perturbed and non-perturbed
            # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
            # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
            # for detailed explanation.
            update_param_noise_threshold = -np.log(
                1. - exploration.value(t) + exploration.value(t) / float(num_actions))
            kwargs['reset'] = reset
            kwargs['update_param_noise_threshold'] = update_param_noise_threshold
            kwargs['update_param_noise_scale'] = True

        action = act(obs, update_eps=update_eps, **kwargs)[0]
        env_action = action
        reset = False

        new_obs, true_rew, done, _ = env.step(env_action)

        # Store transition in the replay buffer.
        rew = reward_giver.get_reward(featurizer.featurize(new_obs))[0][0]
        data_set = expert_dataset[0].get_next_batch(batch_size=64)
        rew_e = reward_giver.get_reward(featurizer.featurize(data_set))
        print('{:>+7.4f}, {:>+7.4f}, {:>4d}'.format(round(rew, 4), round(np.mean(rew_e), 4), env_action), end=', ')
        replay_buffer.add(obs, action, rew, new_obs, float(done))
        obs = new_obs

        episode_rewards[-1] += rew
        true_rewards[-1] += true_rew
        num_episodes = len(episode_rewards)
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)
            true_rewards.append(0.0)
            print('episode: {}'.format(num_episodes))
            reset = True

        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if prioritized_replay:
                experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
            else:
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
            [td_errors, error] = train(obses_t, actions, rewards, obses_tp1, dones, weights)
            losses.append(error)
            if prioritized_replay:
                new_priorities = np.abs(td_errors) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)

            d_step = 3
            experience = replay_buffer.sample(batch_size*d_step)
            (_, _, _, obses_tp1, *_) = experience
            data_set = expert_dataset[random.randint(0, len(expert_dataset) - 1)]

            for ob_batch in iterbatches(obses_tp1, batch_size=len(obses_tp1) // d_step):
                real_batch_size = len(ob_batch)

                obs_e = data_set.get_next_batch(batch_size=real_batch_size)
                obs_e = featurizer.featurize(obs_e)
                ob_batch = featurizer.featurize(ob_batch)
                *newlosses, g = reward_giver.lossandgrad(ob_batch, obs_e)
                reward_giver.adam.update(g, 1e-4)
                d_losses.append(newlosses)

        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            update_target()

            if t % 5000 == 0:
                act.save(".\\dataset\\my_model_{}.pkl".format(t))

        if done and num_episodes % print_freq == 0:
            if len(d_losses) > 0:
                for (name, loss) in zip(reward_giver.loss_name, np.mean(d_losses, axis=0)):
                    logger.record_tabular(name, loss)

            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 100 episode reward", np.mean(episode_rewards[-101:-1]))
            logger.record_tabular("mean 100 true reward", np.mean(true_rewards[-101:-1]))
            logger.record_tabular("mean 100 loss", np.mean(losses))
            logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
            logger.dump_tabular()

            losses, d_losses = [], []

    return act
