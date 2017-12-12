import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({'font.size':22})

g = [32.15, 32.87, 31.15, 29.10]
p = [20.75, 43.02, 19.95, 25.33]

inds = np.array([1,2,3,4])
w = 0.3

plt.figure(1, figsize=(16,9))
r1 = plt.bar(inds, g, w)
r2 = plt.bar(inds+w, p, w)
plt.xlabel('Selected Feature(s)')
plt.ylabel('MAE in Years')
plt.title('MAE in Years by Feature')
plt.xticks(inds + w/2, ['Unigrams', 'Trees', 'Uni+Trees', 'POS Tags'])

for r,x in zip(r1, g):
    h = r.get_height()
    plt.text(r.get_x() + r.get_width()/2, h + 0.2, str(x), ha='center', va='bottom')
    
for r,x in zip(r2, p):
    h = r.get_height()
    plt.text(r.get_x() + r.get_width()/2, h + 0.2, str(x), ha='center', va='bottom')

plt.legend(['Gutenberg', 'Proquest'])
plt.grid()
plt.show()
