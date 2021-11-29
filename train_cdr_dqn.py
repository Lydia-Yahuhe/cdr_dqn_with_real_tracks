from baselines import deepq
from baselines.common import models
from baselines.gail.adversary import TransitionClassifier
from baselines.gail.dataset.mujoco_dset import Dset

from fltenv.env import ConflictEnv

from train_featurizer import train_featurizer

root = ".\\dataset\\my_model"


def train(test=False, path='dqn_policy'):
    env = ConflictEnv(limit=0, reverse='evaluate' in path)
    featurizer, dataset = train_featurizer(restore_model=True)
    reward_giver = TransitionClassifier(env, 64, entcoeff=1e-3)

    network = models.mlp(num_hidden=64, num_layers=2, layer_norm=True)
    if not test:
        act = deepq.learn(
            env,
            network=network,  # 隐藏节点，隐藏层数
            lr=1e-3,
            batch_size=16,
            total_timesteps=100000,
            buffer_size=1000,

            learning_starts=100,

            reward_giver=reward_giver,
            expert_dataset=[Dset(frames, randomize=True) for frames in dataset],
            featurizer=featurizer

            # prioritized_replay=True
        )
        print('Save model to my_model.pkl')
        act.save(root+'.pkl')
    else:
        act = deepq.learn(env,
                          network=network,
                          total_timesteps=0,
                          load_path=root+"_{}.pkl".format(path.split('_')[-1]))
        env.evaluate(act, save_path=path)
    env.close()


if __name__ == '__main__':
    train()
    # train(test=True, path='dqn_policy_evaluate_20000')
    # train(test=True, path='dqn_policy_test_20000')
