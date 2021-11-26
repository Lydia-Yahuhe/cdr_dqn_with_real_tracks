import time
from enum import Enum
from contextlib import contextmanager

import os
import random

from rtree import index

from .computation import make_bbox, bearing, delta_hdg, high, coord_conversing, distance
from .load_from_db import atsRoute


# -------------
# File Processing
# -------------
def get_file_list(root, limit=None, shuffle=True):
    file_list = []

    for dir_or_file in os.listdir(root):
        if limit is not None and dir_or_file not in limit:
            continue

        dir_or_file = os.path.join(root, dir_or_file)
        if os.path.isfile(dir_or_file) and dir_or_file.endswith('.xlsx'):
            file_list.append(dir_or_file)

    if shuffle:
        random.shuffle(file_list)

    return file_list


def get_dir_files_dict(data_path, shuffle=True):
    print(data_path)
    # 是否存在data文件夹
    assert os.path.exists(data_path)

    file_dict = {}
    for dir_name in os.listdir(data_path):
        dir_path = os.path.join(data_path, dir_name)
        if os.path.isfile(dir_path):
            continue
        file_dict[dir_name] = get_file_list(dir_path, shuffle=shuffle)
    return file_dict


# -----------------
# Common Functions
# -----------------
@contextmanager
def timed(msg):
    print(msg, end='\t')
    start = time.time()
    yield
    print("done in %.3f seconds" % (time.time() - start))


# --------------------
# Geometric Functions
# --------------------
def build_rt_index(ac_en):
    p = index.Property()
    p.dimension = 3
    idx = index.Index(properties=p)
    for i, a in enumerate(ac_en):
        idx.insert(i, make_bbox(a.position[1:4]))
    return idx


def data_cleaning(points):
    time_diff_list = []
    for i, p in enumerate(points[1:]):
        last_p = points[i]
        time_diff = p[0] - last_p[0]
        dist = distance(p[1:3], last_p[1:3])
        print(last_p[0], p[0], time_diff, dist, dist/time_diff, p[3]-last_p[3])

        time_diff_list.append(time_diff)
    print(time_diff_list)
    return points


conflictType = Enum('conflictType', ('SameTrack', 'Cross', 'Opposite', 'Parallel'))


def check_track_type(p1, p2):
    h1, h2 = p1[-1], p2[-1]
    delta = (h1-h2+360) % 360

    angle = abs(delta_hdg(bearing(p1[1:3], p2[1:3]), h1))
    if delta <= 45 or delta >= 315:
        if 165 >= angle >= 15:
            return conflictType.Parallel, 20000
        else:
            return conflictType.SameTrack, 20000

    if 225 >= delta >= 135:
        if 165 >= angle >= 15:
            return conflictType.Parallel, 20000
        else:
            return conflictType.Opposite, 50000

    return conflictType.Cross, 30000


def check_in_bbox(bbox_, point_, start, end):
    lngs, lats = [start[0], end[0]], [start[1], end[1]]
    minLng, maxLng = min(lngs), max(lngs)
    minLat, maxLat = min(lats), max(lats)

    if maxLng < bbox_[0] or minLng > bbox_[3]:
        return False
    elif maxLat < bbox_[1] or minLat > bbox_[4]:
        return False
    elif high(point_, start, end) >= 100000:
        return False
    return True


def around_point(point):
    segments = []
    bbox = make_bbox(point, ext=(1, 1, 3000))
    for route in atsRoute:
        if check_in_bbox(bbox, point, route[1], route[2]):
            segments.append(route)
    return segments


# 平行航路的方向和距离
def parallel_dist_hdg(pos, start, end):
    dist = high(pos, start, end)
    hdg = bearing(start, end)
    delta = delta_hdg(hdg, bearing(pos, end))

    if delta < 0:
        hdg = (hdg + 90) % 360
    else:
        hdg = (hdg - 90) % 360

    return dist, hdg


# 点判断是否在一条线上
def online_or_not(pos, start, end):
    h11 = bearing(start, end)
    h12 = bearing(start, pos)
    delta1 = abs(delta_hdg(h11, h12))

    h21 = bearing(end, start)
    h22 = bearing(end, pos)
    delta2 = abs(delta_hdg(h21, h22))

    return delta1 <= 90 and delta2 <= 90


# 得到坐标系转换的旋转角度和坐标
def make_conversion_coord(points):
    heading = bearing(points[0], points[1])
    theta = delta_hdg(90, heading)
    if theta > 0:
        theta -= 360
    theta = abs(theta)

    tmp = []
    for p in points:
        tmp.append(coord_conversing(p[0], p[1], theta)[0])
    return tmp, theta


# 判断在计划的哪个航段上
def which_segment(self):
    pos = self.position[1:4]
    lng = coord_conversing(pos[0], pos[1], self.theta)[0]

    for i, start in enumerate(self.lngs_[:-1]):
        check = [start, self.lngs_[i+1]]
        if lng < min(check):
            return -1
        if max(check) >= lng >= min(check):
            return i
    return -100

