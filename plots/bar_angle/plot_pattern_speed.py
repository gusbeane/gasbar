import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import astropy.units as u

name_list = ['lvl5']#, 'lvl4', 'lvl3', 'lvl5-fg0.2']

n = 5

fig, ax = plt.subplots(1, 1)

for name in name_list:
    out = pickle.load(open('bar_angle_'+name+'.p', 'rb'))

    time = out['time']
    polyfit = out['poly_fit']
    firstkey = out['firstkey']

    pattern_speed = np.zeros(len(time))

    mypfit = polyfit[n]
    for i in range(n):
        pspeed = (n-i) * mypfit[i] * time**(n-1-i) # taking the derivative of a general polynomial
        pattern_speed[firstkey:] += pspeed[firstkey:] # set pattern speed to zero before firstkey

    pattern_speed = pattern_speed / u.Myr
    pattern_speed = pattern_speed.to_value(u.km/u.s/u.kpc)

    ax.plot(time[firstkey:], pattern_speed[firstkey:], label=name)

ax.set_xlabel('time [Myr]')
ax.set_ylabel('pattern speed [ km/s/kpc ]')

fig.legend(frameon=False, title='resolution')

fig.tight_layout()
fig.savefig('pattern_speed.pdf')
