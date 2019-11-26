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

    pattern_speed = np.zeros(len(time))

    mypfit = polyfit[n]
    for i in range(n):
        pspeed = (n-i) * mypfit[i] * time**(n-1-i) # taking the derivative of a general polynomial
        pattern_speed[firstkey:] += pspeed[firstkey:] # set pattern speed to zero before firstkey

    pattern_speed = pattern_speed / u.Myr
    pattern_speed = pattern_speed.to_value(u.km/u.s/u.kpc)

    finite_diff = np.gradient(true_bar_angle, time)
    finite_diff = finite_diff / u.Myr
    finite_diff = finite_diff.to_value(u.km/u.s/u.kpc)

    l = ax.plot(time[firstkey:], pattern_speed[firstkey:], label=name)
    c = l[0].get_color()
    ax.scatter(time[firstkey:], finite_diff[firstkey:], c=c, alpha=0.2, s=1)

ax.set_xlabel('time [Myr]')
ax.set_ylabel('pattern speed [ km/s/kpc ]')

fig.legend(frameon=False, title='resolution')

fig.tight_layout()
fig.savefig('pattern_speed.pdf')
