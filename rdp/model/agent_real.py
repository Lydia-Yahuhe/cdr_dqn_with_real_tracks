from .classes import State, SPD, ALT, HDG


class Single:
    def __init__(self, fpl):
        self.fpl = fpl
        self.idx = -1
        self.tracks_length = len(self.points)-1
        self.state = State(ALT.Cruise, HDG.Straight, SPD.Uniform, 'no')

    @property
    def plan(self):
        return self.fpl.plan_tracks

    @property
    def points(self):
        return self.fpl.real_tracks

    @property
    def id(self):
        return self.fpl.id

    def do_step(self, time):
        if time == self.next[0]:
            self.idx += 1

    def is_enroute(self):
        return self.tracks_length > self.idx >= 0

    def is_finished(self):
        return self.idx >= self.tracks_length

    @property
    def position(self):
        return self.points[self.idx]

    @property
    def next(self, n=1):
        return self.points[min(self.idx+n, self.tracks_length)]
