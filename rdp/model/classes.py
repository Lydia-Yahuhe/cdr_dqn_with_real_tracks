from dataclasses import dataclass

from enum import Enum
from typing import List, Dict


@dataclass
class FlightPlan:
    id: str
    ac: dict
    from_to: dict
    plan_tracks: Dict[str, List[float]]
    real_tracks: List[object]

    def tostring(self):
        return self.id, self.ac['id'], self.ac['type'], self.from_to, len(self.plan_tracks), len(self.real_tracks)


ALT = Enum('ALT', ('Cruise', 'Descent', 'Climb'))
SPD = Enum('SPD', ('Accel', 'Decel', 'Uniform'))
HDG = Enum('HDG', ('Right', 'Left', 'Straight'))


@dataclass
class State:
    alt: Enum
    hdg: Enum
    spd: Enum
    chg: str

    def tostring(self):
        return self.alt, self.hdg, self.spd, self.chg


def agent_info(agent, agent_pos):
    fpl = agent.fpl.tostring()
    state = agent.state.tostring()
    return "\n\t    fpl:{}\n\t  state:{}\n\tpos_ori:{}\n\tpos_aft:{}\n".format(
        fpl, state, agent.position[:], agent_pos[:])


@dataclass
class Conflict:
    def __init__(self, c_id, time, hDist, vDist, c_type, a0, a1, a0_pos, a1_pos):
        self.id = c_id
        self.time = time
        self.hDist = hDist
        self.vDist = vDist
        self.type = c_type
        self.c_points = [a0.position, a1.position, a0_pos, a1_pos]
        self.a0_info = agent_info(a0, a0_pos)
        self.a1_info = agent_info(a1, a1_pos)
        self.other = []

    def dump(self):
        print("-------------------------------")
        print(" c_info:", self.conflict_info())
        print("a0_info:", self.a0_info)
        print("a1_info:", self.a1_info)
        print("c_other:", self.other)
        print("------------------------------")

    def conflict_info(self):
        return "\n\t   id:{}\n\t time:{}\n\thDist:{}\n\tvDist:{}\n\t type:{}".format(
            self.id, self.time, self.hDist, self.vDist, self.type)

    def set_other(self, delta_hs, delta_vs):
        if len(delta_hs) > 0:
            # 找出水平最小的水平距离和垂直距离
            min_h = min(delta_hs)
            idx = delta_hs.index(min_h)
            self.other += [min_h, delta_vs[idx]]

            # 找出垂直最小的水平距离和垂直距离
            min_v = min(delta_vs)
            idx = delta_vs.index(min_v)
            self.other += [delta_hs[idx], min_v]
