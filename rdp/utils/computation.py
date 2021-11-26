import math


R = 6371393.0
RADIAN_DEG = 180.0 / math.pi
G0 = 0.982
KM2M = 1000.0
M2KM = 1.0 / KM2M

parameters_time = [1, 60, 60*60, 24*60*60]


def reach2level(alt):
    alt_ = alt//300*300
    if alt_ < 8400.0 and alt-alt_ >= 150.0:
        alt_ += 300.0
    elif alt_ >= 8700.0:
        alt_ += 200.0
    return alt_


def clock2time(clock, day=False):
    clock_str = str(int(clock))
    day = int(clock_str[6:8])-1 if day else 0
    clock = clock_str[8:]
    hour = int(clock[:2])
    minute = int(clock[2:4])
    second = int(clock[4:])
    return sum([e*parameters_time[i] for i, e in enumerate([second, minute, hour, day])])


def str2coord(coord):
    coord = str(int(coord*10))
    second = int(coord[-3:])
    minute = int(coord[-5:-3])
    hour = int(coord[-8:-5])

    return hour + minute/60 + second/36000


# 计算两个坐标点的距离
def distance(d0, d1):
    lng0 = math.radians(d0[0])
    lat0 = math.radians(d0[1])
    lng1 = math.radians(d1[0])
    lat1 = math.radians(d1[1])
    dlng = lng0 - lng1
    dlat = lat0 - lat1
    tmp1 = math.sin(dlat/2)
    tmp2 = math.sin(dlng/2)
    a = tmp1 * tmp1 + math.cos(lat0) * math.cos(lat1) * tmp2 * tmp2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


# 两个heading之间的正负角度，用来判断左转还是右转
def delta_hdg(h1, h2):
    if h1 > 180:
        h11 = h1-180
        if h2 >= h11:
            delta = h2-h1
        else:
            delta = 180-h11+h2
    else:
        h11 = h1+180
        if h2 <= h11:
            delta = h2-h1
        else:
            delta = h2-h11-180

    return delta


# 计算两个坐标点的航向
def bearing(d0, d1):
    lng0 = math.radians(d0[0])
    lat0 = math.radians(d0[1])
    lng1 = math.radians(d1[0])
    lat1 = math.radians(d1[1])
    dlng = lng1 - lng0
    coslat1 = math.cos(lat1)
    tmp1 = math.sin(dlng) * coslat1
    tmp2 = math.cos(lat0) * math.sin(lat1) - math.sin(lat0) * coslat1 * math.cos(dlng)

    return (math.atan2(tmp1, tmp2) * RADIAN_DEG)%360


# 根据距离和航向得出下一个点的坐标
def move_point2d(src, course, dist):
    lng1 = math.radians(src[0])
    lat1 = math.radians(src[1])
    r = dist / R
    course = math.radians(course)
    cosR = math.cos(r)
    sinR = math.sin(r)
    sinLat1 = math.sin(lat1)
    cosLat1 = math.cos(lat1)

    lat2 = math.asin(sinLat1 * cosR + cosLat1 * sinR * math.cos(course))
    lng2 = lng1 + math.atan2(math.sin(course) * sinR * cosLat1, cosR - sinLat1 * math.sin(lat2))

    return [math.degrees(lng2), math.degrees(lat2)]


def make_bbox(pos, ext=(0, 0, 0)):
    return (pos[0] - ext[0], pos[1] - ext[1], pos[2]*10 - ext[2],
            pos[0] + ext[0], pos[1] + ext[1], pos[2]*10 + ext[2])


# 已知三个点的坐标，计算其三角形面积
def area(a, b, c):
    ab = distance(a, b)
    ac = distance(a, c)
    bc = distance(b, c)
    p = (ab+ac+bc) / 2
    S = math.sqrt(abs(p*(p-ab)*(p-ac)*(p-bc)))

    return S


# 已知三个点的坐标，计算a点到bc的距离
def high(a, b, c):
    S = area(a, b, c)
    return 2*S/distance(b, c)


# 坐标系转换
# x'=x·cos(θ)+y·sin(θ) y'=y·cos(θ)-x·sin(θ)
def coord_conversing(lng, lat, theta):
    theta = math.radians(theta)

    lng_ = lng*math.cos(theta) + lat*math.sin(theta)
    lat_ = lat*math.cos(theta) - lng*math.sin(theta)
    return lng_, lat_
