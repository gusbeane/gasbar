import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import astropy.units as u

name_list = ['lvl5', 'lvl4']#, 'lvl3', 'lvl5-fg0.2']

n = 5

fig, ax = plt.subplots(1, 1)

for name in name_list:
    out = pickle.load(open('bar_angle_'+name+'.p', 'rb'))

    time = out['time']
    polyfit = out['poly_fit']
    firstkey = out['firstkey']

    true_bar_angle = out['bar_angle']

    bar_angle = np.zeros(len(time))

    mypfit = polyfit[n]
    for i in range(n):
        bangle = mypfit[i] * time**(n-i) # taking the derivative of a general polynomial
        bar_angle[firstkey:] += bangle[firstkey:] # set pattern speed to zero before firstkey

    ax.plot(time, bar_angle, label=name)
    ax.scatter(time, true_bar_angle, s=1, c='k')

ax.set_xlabel('time [Myr]')
ax.set_ylabel('bar angle')

fig.legend(frameon=False, title='resolution')

fig.tight_layout()
fig.savefig('bar_angle.pdf')
