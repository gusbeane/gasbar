import sys
mach = sys.platform
if mach == 'linux':
    sys.path.append('/n/home01/abeane/mwib_analysis')
    base = '/n/scratchlfs/hernquist_lab/abeane/mwib_runs/arepo'
elif mach == 'darwin':
    sys.path.append('/Users/abeane/scratch/mwib_analysis')
    base = '/Volumes/abeaneSSD001/mwib_runs/arepo'

from mwib_analysis import mwib_io
import pickle
import numpy as np
import h5py as h5
from tqdm import tqdm
import matplotlib.pyplot as plt
import astropy.units as u

name_list = ['lvl5', 'lvl4', 'lvl3']
dt = 100 # Myr

fig, ax = plt.subplots(1, 1)

for name in name_list:
    # time in Myr, mass in Msun
    star_birthtime, star_mini = pickle.load(open('SFR_'+name+'.p', 'rb'))

    bins = np.linspace(0, np.max(star_birthtime), dt)

    digit = np.digitize(star_birthtime, bins)

    t_list = np.array([star_birthtime[digit == i].mean() for i in range(1, len(bins))])
    SFR_list, edges = np.histogram(star_birthtime, bins=bins, weights=star_mini)

    SFR_list = SFR_list * (1/dt) * u.Msun/u.Myr
    SFR_list = SFR_list.to_value(u.Msun/u.yr)

    ax.plot(t_list, SFR_list, label=name)

ax.set_xlabel('time [Myr]')
ax.set_ylabel('SFR [Msun/yr]')

ax.set_yscale('log')

fig.legend(frameon=False, title='resolution')

fig.tight_layout()
fig.savefig('SFR.pdf')
