import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.ndimage import gaussian_filter1d

name_list = ['lvl5', 'lvl4', 'lvl3', 'lvl5-rotbulge', 'lvl4-rotbulge']# 'lvl5-fg0.2']

n = 5

fig, ax = plt.subplots(1, 1)

for name in name_list:
    out = pickle.load(open('bar_angle_'+name+'.p', 'rb'))

    time = out['time']
    A2A0 = out['A2A0_max']
    smoothed_A2A0 = gaussian_filter1d(A2A0, 5)

    l = ax.plot(time, smoothed_A2A0, label=name)
    # c = l[0].get_color()
    # ax.scatter(time[firstkey:], finite_diff[firstkey:], c=c, alpha=0.2, s=1)

ax.set_xlabel('time [Myr]')
ax.set_ylabel('max(A2/A0)')

ax.set_ylim([0, 1])

fig.legend(frameon=False, title='resolution')

fig.tight_layout()
fig.savefig('fourier_fn_time.pdf')
