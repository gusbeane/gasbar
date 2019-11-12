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

name_list = ['lvl5', 'lvl4']#, 'lvl3']

fig, ax = plt.subplots(1, 1)

for name in name_list:
    time_list, gas_fraction_list = pickle.load(open('gas_fraction_'+name+'.p', 'rb'))
    ax.plot(time_list, gas_fraction_list, label=name)

ax.set_xlabel('time [Myr]')
ax.set_ylabel('gas fraction')

fig.legend(frameon=False, title='resolution')

fig.tight_layout()
fig.savefig('gas_fraction.pdf')
