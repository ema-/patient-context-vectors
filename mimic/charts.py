import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import numpy as np


clusters = ['Multi-organ\nFailure', 'Toxic\nIngestion', 'Nosocomial\nComplications', 'Vascular\nDisease\nComplications',
            'Surgical\nSource\nControl', 'Complex\nComorbidities', 'Malignancy', 'Chronic\nHepatic\nDisease', 'Neurologic\nComorbidities', 'Trauma/Fx']
deadlist = [464, 54, 457, 265, 222, 509, 227, 322, 240, 61]
alivelist = [314, 249, 193, 117, 170, 452, 24, 107, 166, 142]
percentdead=[]
for i,d in enumerate(deadlist):
    percentdead.append(deadlist[i]/float(deadlist[i]+alivelist[i]))
zipped=list(zip(clusters,deadlist,alivelist,percentdead))
sortedandzipped=sorted(zipped,key=lambda x: x[-1], reverse=True)
clusters, deadlist, alivelist, percentdead =list(zip(*sortedandzipped))

dead = np.array(deadlist)
alive = np.array(alivelist)
ind = [x for x, _ in enumerate(clusters)]

plt.bar(ind, alive, width=0.8, label='', color='grey', bottom=dead)
rects=plt.bar(ind, dead, width=0.8, label='Mortality', color='salmon')

for index,i in enumerate(rects):
    plt.text(i.get_x()+.2, i.get_height()+1,
            '%s%%'%int(percentdead[index]*100),
             ha='center', va='bottom')

plt.xticks(ind, clusters)
plt.ylabel("Number of Admissions")
plt.xlabel("Clusters")
plt.legend(loc="upper right")
plt.title("MIMIC3 ARDS Clusters")

plt.show()