import csv
from math import radians, cos, sin, asin, sqrt
import datetime as dt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in km
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def isWeekDayAft(str, hour):
    """
    Determine if taxi ride took place on a weekday sometime b/w 1 
    and 2 pm
    """
    str = str.split(" ")
    dayStr = str[0]
    timeStr = str[1]
    date = dt.datetime.strptime(dayStr, "%Y-%m-%d")
    weekday = False # only set to true if day is M-Th
    time = False # only set to true if time is in "hour"
    if date.weekday() < 4: #M-Th
        weekday = True
    if int(timeStr.split(":")[0]) == hour:
        time = True
    return weekday and time

def getCoords(str):
    """
    Convert "(x,y)" to a float tuple
    """
    str = str.split(",")
    x = float(str[0].split("(")[1])
    y = float(str[1].split(")")[0])
    return x,y

def createTransitionModel(file = 'taxi_master_clean.csv', hour = 13):
    transitionModel = dict()
    with open(file, 'rb') as csvfile:
        taxireader = csv.reader(csvfile)
        taxireader.next()
        totalRides = 0.
        # obtain initial counts for actions in each state
        for line in taxireader:
            pTime = line[1] # pickup time
            if isWeekDayAft(pTime, hour):
                totalRides += 1 
                py, px = getCoords(line[3]) # pickup latitude, longitude
                dy, dx = getCoords(line[4]) # dropoff latitude, longitude
                dist = haversine(px, py, dx, dy) # dist is proxy for avg reward
                pNei = line[5] # pickup neighborhood
                dNei = line[6] # dropoff neighborhood
                
                if pNei in transitionModel:
                    if dNei in transitionModel[pNei]:
                        oCount, oDist = transitionModel[pNei][dNei]
                        nCount = oCount + 1
                        nDist = (oCount * oDist + dist)/ nCount
                        transitionModel[pNei][dNei] = (nCount, nDist)
                    else:
                        transitionModel[pNei][dNei] = (1.,dist)
                else:
                    transitionModel[pNei] = dict()
                    transitionModel[pNei][dNei] = (1.,dist)

        # normalize counts for each action for each state and obtain prob of pickup
        pickupProb = dict()
        maxProb = 0.
        for key in transitionModel:
            total = sum(item[0] for item in transitionModel[key].values())
            prob = total/totalRides
            pickupProb[key] = prob
            maxProb = max(prob, maxProb)
            for innerKey in transitionModel[key]:
                count, dist = transitionModel[key][innerKey]
                transitionModel[key][innerKey] = (count/total, dist)

        # normalize prob of pickup so maxProb location has 95% chance of pickup
        weight = 0.95/maxProb
        for key in pickupProb:
            pickupProb[key] *= weight

        return transitionModel,pickupProb
