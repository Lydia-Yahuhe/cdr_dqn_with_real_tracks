from abc import ABC

import gym
from gym import spaces

from fltenv.scene import ConflictScene
from fltenv.cmd import CmdCount, reward_for_cmd

from fltsim.load import load_and_split_data
from fltsim.visual import *


def calc_reward(solved, cmd_info):
    if not solved:  # failed
        reward = -5.0
    else:  # solved
        rew = reward_for_cmd(cmd_info)
        reward = 0.5+min(rew, 0)

    print('{:>+4.2f}'.format(reward), end=', ')
    return reward


class ConflictEnv(gym.Env, ABC):
    def __init__(self, limit=30, act='Discrete', reverse=False):
        self.limit = limit

        if reverse:
            self.test, self.train = load_and_split_data('scenarios_gail_final', split_ratio=0.8)
        else:
            self.train, self.test = load_and_split_data('scenarios_gail_final', split_ratio=0.8)

        if act == 'Discrete':
            self.action_space = spaces.Discrete(CmdCount)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1, ), dtype=np.float64)

        self.picture_size = (width, height, channel) = (670, 450, 3)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(height, width, channel), dtype=np.float64)

        print('----------env----------')
        print('    train size: {:>6}'.format(len(self.train)))
        print(' validate size: {:>6}'.format(len(self.test)))
        if act == 'Discrete':
            print('  action shape: {}'.format((self.action_space.n,)))
        else:
            print('  action shape: {}'.format(self.action_space.shape))
        print('   state shape: {}'.format(self.observation_space.shape))
        print('-----------------------')

        self.scene = None
        self.video_out = None
        self.result = None

    def test_over(self):
        return len(self.test) <= 0

    def shuffle_data(self):
        np.random.shuffle(self.train)

    def reset(self, test=False):
        if not test:
            info = self.train.pop(0)
            self.scene = ConflictScene(info, limit=self.limit)
            self.train.append(info)
        else:
            info = self.test.pop(0)
            self.scene = ConflictScene(info, limit=self.limit)

        return self.scene.get_states(*self.picture_size)

    def step(self, action, scene=None):
        if scene is None:
            scene = self.scene

        solved, cmd_info = scene.do_step(action)
        rewards = calc_reward(solved, cmd_info)
        states = scene.get_states(*self.picture_size)
        self.result = solved
        return states, rewards, True, {'result': solved}

    def evaluate(self, act, save_path='policy', **kwargs):
        num_array = []
        obs_array = []
        act_array = []
        rew_array = []
        n_obs_array = []
        indexes = []

        size = len(self.test)

        episode = 0
        while not self.test_over():
            print(episode, size)
            obs_collected = {'num': [], 'obs': [], 'act': [], 'rew': [], 'n_obs': []}

            obs, done = self.reset(test=True), False
            result = {'result': True}
            count = 0
            while not done:
                if 'gail' in save_path:
                    action, _ = act(kwargs['stochastic'], obs)
                    action = np.argmax(action)
                    print('gail', action)
                    # action = int(action[0]*54+54)
                elif 'dqn' in save_path:
                    action = act(np.array(obs)[None])[0]
                    print('dqn', action)
                else:
                    action = np.random.randint(0, CmdCount)
                    print('random', action)
                next_obs, rew, done, result = self.step(action)

                obs_collected['num'].append(self.scene.info.id)
                obs_collected['obs'].append(obs)
                obs_collected['act'].append(action)
                obs_collected['rew'].append(rew)
                obs_collected['n_obs'].append(next_obs)
                obs = next_obs
                count += 1

            if result['result']:
                # self.render()
                num_array += obs_collected['num']
                obs_array += obs_collected['obs']
                act_array += obs_collected['act']
                rew_array += obs_collected['rew']
                n_obs_array += obs_collected['n_obs']
                indexes.append(count)

            episode += 1

        num_array = np.array(num_array)
        obs_array = np.array(obs_array, dtype=np.float64)
        act_array = np.array(act_array, dtype=np.float64)
        rew_array = np.array(rew_array, dtype=np.float64)
        n_obs_array = np.array(n_obs_array, dtype=np.float64)
        indexes = np.array(indexes, dtype=np.int8)

        print('Success Rate is {}%'.format(len(indexes) * 100.0 / size))
        print(obs_array.shape, act_array.shape, rew_array.shape, n_obs_array.shape)
        np.savez(save_path+'.npz',
                 num=num_array, obs=obs_array, acs=act_array,
                 rews=rew_array, n_obs=n_obs_array, indexes=indexes)

    def render(self, mode='human', **kwargs):
        picture_size = (670, 450)
        frames = []
        if self.video_out is None:
            self.video_out = cv2.VideoWriter(kwargs['name'], cv2.VideoWriter_fourcc(*'MJPG'), 20.0, picture_size)

        kwargs = dict(border=[109.3, 116, 29, 33.5], scale=100)

        info = self.scene.info
        agent_set = self.scene.agentSet
        conflict_ac, clock = info.conflict_ac, info.time

        # 轨迹点
        points_dict = {}
        for key, agent in agent_set.agents.items():
            for clock_, track in agent.tracks.items():
                if clock_ in points_dict.keys():
                    points_dict[clock_].append([key, key in conflict_ac] + track)
                else:
                    points_dict[clock_] = [[key, key in conflict_ac]+track]

        for t in range(clock-300+self.limit, clock+300):
            # 武汉扇区的底图（有航路）
            base_img = cv2.imread('dataset/wuhan_base.jpg', cv2.IMREAD_COLOR)

            if t not in points_dict.keys():
                continue

            points = points_dict[t]

            # 将当前时刻所有航空器的位置和状态放在图上
            frame, points_just_coord = add_points_on_base_map(points, base_img, **kwargs)
            frame = cv2.resize(frame, picture_size)
            frames.append(frame)

            self.video_out.write(frame)
        return np.stack(frames)

    def close(self):
        if self.video_out is not None:
            self.video_out.release()
            cv2.waitKey(1) & 0xFF
            cv2.destroyAllWindows()
            self.video_out = None

