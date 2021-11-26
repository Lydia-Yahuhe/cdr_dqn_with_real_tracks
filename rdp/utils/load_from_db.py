import pymongo


def load_waypoint():
    wpts = {}
    cursor = db['Waypoint'].find()
    for pt in cursor:
        wpts[pt['id']] = (pt['point']['lng'], pt['point']['lat'], 8100)

    # cursor = db["Airport"].find()
    # for pt in cursor:
    #     wpts[pt['id']] = [pt['location']['lng'], pt['location']['lat']]

    return wpts


def load_routing():
    ret = {}
    cursor = db['Routing'].find()
    for e in cursor:
        r_id = e['id'].split('#')[0]

        if r_id in ret.keys():
            ret[r_id].append(e['waypointList'])
        else:
            ret[r_id] = [e['waypointList']]

    return ret


def load_atsRoute():
    ret = []
    cursor = db['ATSRoute'].find()
    for e in cursor:
        for seg in e['segmentList']:
            start = seg['start']
            end = seg['end']
            ret.append([start+'-'+end, wpt_dict[start], wpt_dict[end]])

    return ret


db = pymongo.MongoClient('localhost')['admin']
wpt_dict = load_waypoint()
atsRoute = load_atsRoute()

