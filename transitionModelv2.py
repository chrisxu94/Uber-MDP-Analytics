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

def isWeekDay(string):
    """
    Determine if taxi ride took place on a weekday sometime b/w 1 
    and 2 pm
    """
    string = string.split(" ")
    dayStr = string[0]
    timeStr = string[1]
    date = dt.datetime.strptime(dayStr, "%Y-%m-%d")
    weekday = False # only set to true if day is M-Th
    if date.weekday() < 4: #M-Th
        weekday = True
    hour = int(timeStr.split(":")[0])
    halfhour = int(timeStr.split(":")[1])
    half = 0
    if halfhour > 30:
        half = 1
    return weekday, 2*hour + half

def getCoords(string):
    """
    Convert "(x,y)" to a float tuple
    """
    string = string.split(",")
    x = float(string[0].split("(")[1])
    y = float(string[1].split(")")[0])
    return x,y

def createTransitionModel(File = 'taxi_master_clean.csv', hour = 13):
    transitionModel = dict()
    with open(File, 'rb') as csvfile:
        taxireader = csv.reader(csvfile)
        taxireader.next()
        totalRides = 0.
        # obtain initial counts for actions in each state
        for line in taxireader:
            pTime = line[1] # pickup time
            weekDay, time = isWeekDay(pTime)
            if weekDay:
                totalRides += 1 
                py, px = getCoords(line[3]) # pickup latitude, longitude
                dy, dx = getCoords(line[4]) # dropoff latitude, longitude
                dist = haversine(px, py, dx, dy) # dist is proxy for avg reward
                pNeighborhood = line[5] # pickup neighborhood
                dNeighborhood = line[6] # dropoff neighborhood
                pNei = (pNeighborhood, time)
                dNei = (dNeighborhood, (time + 1)%48)
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

        # normalize prob of pickup so maxProb location has 100% chance of pickup
        weight = 1./maxProb
        for key in pickupProb:
            pickupProb[key] *= weight

        return transitionModel,pickupProb
