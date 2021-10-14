import json
import numpy as np
import random 
data = json.load(open('gibson_dset_with_qual.json'))

quals = []
scans = []
for key,items in data.items():
    if items['qual'] == None: continue
    quals.append(items['qual'])
    if items['qual'] >=4:
        scans.append(items['id'])
    

print(quals)
unique, counts = np.unique(np.asarray(quals), return_counts=True)
print(dict(zip(unique, counts)))

random.shuffle(scans)
print(scans)
print('\nPassive-------------')
for x in scans[:46]: print(x)

print('\nActive--------------')
for x in scans[47:47+45]: print(x)

print('\nTest----------------')
for x in scans[-23:]: print(x)
