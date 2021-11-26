import os

import matplotlib.pyplot as plt
import numpy as np
import xlrd

from .model import FlightPlan
from .utils.computation import clock2time


def visual_flight_vertical(points, fpl_id):
    points_arr = np.array(points, dtype=np.float64)

    x_list = list(points_arr[:, 0])
    y_list = list(points_arr[:, 3])
    plt.plot(x_list, y_list)

    plt.title('The altitude changes of {}'.format(fpl_id))
    plt.xlabel('Time Line/s')
    plt.ylabel('Altitude/m')
    plt.yticks(list(range(0, 8600, 300)))

    min_xtick = int(min(x_list) // 300 * 300) - 100
    max_xtick = int((max(x_list) // 300 + 1) * 300) + 100
    plt.xticks(list(range(min_xtick, max_xtick, 300)))

    plt.grid()
    plt.show()


def get_fpl_list(alt_limit=0, v_visual=False, number=-1):
    data_path = os.path.abspath(".\\rdp\\0501-0510.xlsx")
    sheet = xlrd.open_workbook(data_path).sheets()[0]

    fpl_list, starts = [], []
    for row in range(1, sheet.nrows):
        [_, fpl_id, dep, arr, _, _, tracks, *_] = sheet.row_values(row)
        tracks = tracks.split('LA')[1:]
        from_to = dep + '-' + arr

        points, date = [], None
        for track in tracks:
            [position, state] = track.split('V')
            lat, lng, alt = float(position[:9]), float(position[11:21]), float(position[22:]) * 10
            spd, hdg, timestamp = float(state[:4]), float(state[5:8]), state[10:]
            clock = clock2time(timestamp, day=True)
            # print(lat, lng, alt, spd, hdg, timestamp)
            if date is None:
                date = timestamp[:8]

            if alt < alt_limit:
                continue
            points.append([clock, lng, lat, alt, spd, hdg])

        if v_visual:
            visual_flight_vertical(points, fpl_id)

        if len(points) < 10:
            continue

        print(row, fpl_id, dep, arr, date, len(points))
        starts.append(points[0][0])
        fpl = FlightPlan(id=fpl_id + '#' + date + '#' + from_to, ac=fpl_id,
                         from_to=from_to, plan_tracks={}, real_tracks=points)
        fpl_list.append(fpl)
        if 0 < number <= len(fpl_list):
            break

    return fpl_list, starts
