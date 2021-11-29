import cv2
import time
from contextlib import contextmanager

import numpy as np

from baselines.common import colorize

from fltenv.agent_Set import AircraftAgentSet
from fltenv.cmd import int_2_atc_cmd, check_cmd
from fltsim.visual import add_points_on_base_map


@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'), end=' ')
    tstart = time.time()
    yield
    print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))


def read_from_csv(file_name, limit):
    if file_name is None:
        return [{}, None]

    file_name = 'C:\\Users\\lydia\\Desktop\\Workspace\\data set\\Trajectories\\No.{}.csv'.format(file_name)
    with open(file_name, 'r', newline='') as f:
        ret = {}
        for line in f.readlines():
            [fpl_id, time_, *line] = line.strip('\r\n').split(',')
            if fpl_id in limit:
                continue

            if fpl_id in ret.keys():
                ret[fpl_id][int(time_)] = [fpl_id] + [float(x) for x in line]
            else:
                ret[fpl_id] = {int(time_): [fpl_id] + [float(x) for x in line]}

    return [ret, limit]


class ConflictScene:
    def __init__(self, info, limit=0, read=True):
        self.info = info

        self.conflict_ac, self.clock = info.conflict_ac, info.time

        if read:
            self.agentSet = AircraftAgentSet(fpl_list=info.fpl_list, start=info.start,
                                             supply=read_from_csv(info.id, self.conflict_ac))
        else:
            self.agentSet = AircraftAgentSet(fpl_list=info.fpl_list, start=info.start)

        self.agentSet.do_step(self.clock - 300 + limit, basic=True)
        self.conflict_pos = info.other[0]

        # print('\nNew scenario--------------------------------')
        # print(' Conflict Info: ', self.conflict_ac, self.clock, self.agentSet.time, len(info.fpl_list))

        self.cmd_check_dict = {ac: {'HDG': [], 'ALT': [], 'SPD': []} for ac in self.conflict_ac}
        self.cmd_info = {}

    def now(self):
        return self.agentSet.time

    def get_states(self, width, height, channel, show=False):
        kwargs = dict(border=[109.3, 116, 29, 33.5], scale=100)

        # 轨迹点
        points = []
        for [agent, *state] in self.agentSet.agent_en_:
            points.append([agent, agent in self.conflict_ac] + state)

        # 武汉扇区的底图（有航路）
        base_img = cv2.imread('dataset\\wuhan_base.jpg', cv2.IMREAD_COLOR)

        # 将当前时刻所有航空器的位置和状态放在图上
        frame, _ = add_points_on_base_map(points, base_img, **kwargs)
        frame = cv2.resize(frame, (width, height))

        if show:
            cv2.imshow('image', frame)
            cv2.waitKey(100)
        return frame

    def do_step(self, action):
        agent_id, idx = self.conflict_ac[0], action

        # 指令解析
        now = self.now()
        agent = self.agentSet.agents[agent_id]
        [hold, *cmd_list] = int_2_atc_cmd(now + 1, idx, agent)
        # print('{:>4d}, {:>4d}'.format(idx, hold), end=', ')

        # 执行hold，并探测冲突
        while self.now() < now + hold:
            self.agentSet.do_step(duration=15)
            conflicts = self.agentSet.detect_conflict_list(search=self.conflict_ac)
            if len(conflicts) > 0:
                return False, None  # solved, done, cmd

        # 分配动作
        for cmd in cmd_list:
            cmd.ok, reason = check_cmd(cmd, agent, self.cmd_check_dict[agent_id])
            # print(now, hold, cmd.assignTime, self.now())
            # print('{:>+5d}, {}'.format(int(cmd.delta), int(cmd.ok)), end=', ')
            agent.assign_cmd(cmd)
        cmd_info = {'agent': agent_id, 'cmd': cmd_list, 'hold': hold}
        self.cmd_info[now] = cmd_info

        # 执行动作并探测冲突
        has_conflict = self.__do_step(self.clock + 300, duration=15)

        return not has_conflict, cmd_info  # solved, done, cmd

    def __do_step(self, end_time, duration):
        while self.now() < end_time:
            self.agentSet.do_step(duration=duration)
            conflicts = self.agentSet.detect_conflict_list(self.conflict_ac)
            if len(conflicts) > 0:
                return True
        return False

