from mdpv2 import DrivingMDP
import numpy as np
utility = dict()
mdp = DrivingMDP()
vals = mdp.value_iteration()
for loc,time in vals:
    if loc in utility:
        utility[loc].append(vals[(loc,time)])
    else:
        utility[loc] = [vals[(loc,time)]]

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

x = range(48)
y = range(len(utility.keys()))
plt.yticks(y,utility.keys())
times = []
for i in range(24):
    times.append(str(i)+":00")
    times.append(str(i)+":30")
#plt.xticks(x,times)
x, y = np.meshgrid(x,y)

intensity = np.array(utility.values())

plt.pcolormesh(x, y, intensity)
plt.colorbar()
plt.xlabel('Time (.5 h)')
plt.ylabel('Neighborhood')
plt.title('Value of Neighborhood throughout the Day')
plt.savefig('valsTime.png')
plt.show()
