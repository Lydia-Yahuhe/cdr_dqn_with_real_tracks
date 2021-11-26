import time

import numpy as np
import matplotlib.pyplot as plt

from .agent_real import Single

from ..utils.functions import build_rt_index
from ..utils.computation import make_bbox, distance

from fltsim.visual import save_to_kml


KM2M = 1000.0


class AgentSetReal:
    def __init__(self, fpl_list, starts):
        self.agents = {fpl.id: Single(fpl) for fpl in fpl_list}
        self.time, self.end = min(starts) - 1, max(starts)
        print("\n>>> Start:", self.time, "end:", self.end, "agent:", len(fpl_list))

        self.around_agents = {fpl.id: {} for fpl in fpl_list}
        self.flow = []  # 统计每个时刻在航路上飞的航班（流量）
        self.start = time.time()

    @property
    def all_done(self):
        return self.time >= self.end and len(self.ac_en) <= 1

    def __prepare(self):
        self.time += 1
        self.ac_en = []

    def do_step(self, cue=True, cv=True):
        self.__prepare()
        now = self.time

        states = {}
        for key, agent in self.agents.items():
            agent.do_step(now)
            if agent.is_enroute():
                self.ac_en.append(agent)
                if cv:
                    states[key] = agent.position

        if cue and now % 1200 == 0:
            end = time.time()
            print(now, self.end, len(self.ac_en), round(end - self.start, 2))

        self.flow.append([now, len(self.ac_en)])
        return states

    def detection_conflicts(self, ext=(0.75, 0.75, 900)):
        if len(self.ac_en) <= 1:
            return []

        link_two = []
        idx = build_rt_index(self.ac_en)
        check_list = []
        for a0 in self.ac_en:
            a0_id = a0.id
            pos0 = a0.position[1:4]
            bbox = make_bbox(pos0, ext=ext)

            a0_around = self.around_agents[a0_id]
            for i in idx.intersection(bbox):
                a1 = self.ac_en[i]
                a1_id = a1.id
                if a0_id == a1_id or a0_id+'-'+a1_id in check_list:
                    continue

                pos1 = a1.position[1:4]
                h_dist = distance(pos0, pos1) / KM2M
                v_dist = abs(pos0[2]-pos1[2])
                if h_dist <= 20 or v_dist <= 300:
                    link_two.append([pos0, pos1, h_dist, v_dist])

                if a1_id in a0_around.keys():
                    a0_around[a1_id].append([self.time, h_dist, v_dist])
                else:
                    a0_around[a1_id] = [[self.time, h_dist, v_dist]]
                check_list += [a0_id+'-'+a1_id, a1_id+'-'+a0_id]
            self.around_agents[a0.id] = a0_around
        return link_two

    # 画图的时候加了placemark
    def visual(self, save_path='AgentSet', limit=None):
        tracks_real = {}
        tracks_plan = {}
        for a_id, agent in self.agents.items():
            if limit is not None and a_id not in limit:
                continue

            tracks_real[a_id] = [(point[1], point[2], point[3]) for point in agent.points]
            tracks_plan[a_id] = [(t[0], t[1], t[2]) for t in agent.plan.values()]
        save_to_kml(tracks_real, tracks_plan, save_path=save_path)

    def flow_visual(self, show=True):
        # 可视化流量
        flows = np.array(self.flow)
        x, y = list(flows[:, 0]), list(flows[:, 1])
        plt.plot(x, y)

        if show:
            print('flow visual:')
            plt.savefig('flow.png')
            plt.show()
