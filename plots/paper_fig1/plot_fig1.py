import numpy as np
import matplotlib.pyplot as plt
import arepo
import h5py as h5
import matplotlib as mpl
from astropy.io import fits
from scipy.interpolate import interp1d

from matplotlib import rc
mpl.use('Agg')

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

snap_path = '/n/holystore01/LABS/hernquist_lab/Users/abeane/starbar_runs/runs/'
bprop_path = '/n/home01/abeane/starbar/plots/bar_prop/data/'
skirt_path = '/n/holystore01/LABS/hernquist_lab/Users/abeane/starbar_runs/SKIRT/'

# names
Nbody = 'Nbody'
# phS2R35 = 'phantom-vacuum-Sg20-Rc3.5'
phS2R35 = 'smuggle'

lvl = 'lvl3'

def load_filter_data(wavelength):
    # assumes wavelength argument is in microns (SKIRT output)
    # and filter files are stored in angstrom (from http://svo2.cab.inta-csic.es/theory/fps/)

    filter_b_dat = np.genfromtxt('HST_ACS_HRC.F475W.dat')
    filter_b_dat[:,0] /= 10000
    filter_b_interp = interp1d(filter_b_dat[:,0], filter_b_dat[:,1], bounds_error=False, fill_value=0.0)

    filter_g_dat = np.genfromtxt('HST_ACS_HRC.F606W.dat')
    filter_g_dat[:,0] /= 10000
    filter_g_interp = interp1d(filter_g_dat[:,0], filter_g_dat[:,1], bounds_error=False, fill_value=0.0)

    filter_r_dat = np.genfromtxt('HST_ACS_HRC.F814W.dat')
    filter_r_dat[:,0] /= 10000
    filter_r_interp = interp1d(filter_r_dat[:,0], filter_r_dat[:,1], bounds_error=False, fill_value=0.0)

    fb = filter_b_interp(wavelength)
    fg = filter_g_interp(wavelength)
    fr = filter_r_interp(wavelength)

    return fr, fg, fb

def rgb_from_flux(flux, m, M, beta):
    num = np.arcsinh((flux - m)/beta)
    den = np.arcsinh((M-m)/beta)
    
    out = num/den
    
    out[flux < m] = 0.0
    out[flux > M] = 1.0
    
    return out

def gen_image(idx, name, lvl):
    # first define some parameters
    # just made ad hoc by fiddling with a jupyter notebook
    m_r = 8.702199520584396e-13 * 0.1
    M_r = 8.702199520584396e-13 * 35
    beta_r = 8.702199520584396e-13 * 0.631
    
    m_g = 7.477015680471923e-13 * 0.1
    M_g = 7.477015680471923e-13 * 25.1
    beta_g = 7.477015680471923e-13 * 0.631
    
    m_b = 2.585393310960858e-13 * 0.1
    M_b = 2.585393310960858e-13 * 30
    beta_b = 2.585393310960858e-13 * 0.631

    # load in data
    fname = skirt_path + 'run_' + name + '/snap' + "{:03d}".format(idx) + '/' + name + '_fo_total.fits'
    hdul = fits.open(fname)
    wavelength = hdul[1].data # in microns
    wavelength = np.array(wavelength.tolist()).reshape(len(wavelength))
    image_data = hdul[0].data
    image_data = np.array(image_data.tolist())

    # load in filters and apply it to the data
    fr, fg, fb = load_filter_data(wavelength)

    flux_r_band_applied = np.array([fr[i] * image_data[i] for i in range(len(fr))])
    flux_g_band_applied = np.array([fg[i] * image_data[i] for i in range(len(fg))])
    flux_b_band_applied = np.array([fb[i] * image_data[i] for i in range(len(fb))])

    # print(flux_r_band_applied)
    # print(flux_r_band_applied.shape)
    # print(wavelength)
    # print(wavelength.shape)
    # print('\n')
    # print(wavelength.astype(np.float64))

    rband_image = np.trapz(flux_r_band_applied, wavelength.astype(np.float64), axis=0)
    gband_image = np.trapz(flux_g_band_applied, wavelength.astype(np.float64), axis=0)
    bband_image = np.trapz(flux_b_band_applied, wavelength.astype(np.float64), axis=0)

    # convert filter valeus to rgb values
    r_value = rgb_from_flux(rband_image, m_r, M_r, beta_r)
    g_value = rgb_from_flux(gband_image, m_g, M_g, beta_g)
    b_value = rgb_from_flux(bband_image, m_b, M_b, beta_b)

    image = np.stack([r_value, g_value, b_value], axis=-1)
    return image

def run():
    nres = 256
    rng = [[-10., 10.], [-10., 10.]]

    Nbody_idx = [500, 700, 900]
    SMUGGLE_idx = [200, 400, 600]

    name_list = [Nbody, phS2R35]

    extent = [rng[0][0], rng[0][1], rng[1][0], rng[1][1]]

    cm = 1/2.54

    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(18*cm, 12*cm))

    for i in range(len(Nbody_idx)):
        # plot Nbody
        print('name=', Nbody, ' idx=', Nbody_idx[i])
        fname = 'image_'+Nbody+str(Nbody_idx[i])+'.npy'
        try:
            image = np.load(fname)
        except:
            image = gen_image(Nbody_idx[i], Nbody, lvl)
            np.save(fname, image)
        
        ax[0][i].imshow(image.swapaxes(0, 1), extent=extent)

        # plot SMUGGLE
        print('name=', phS2R35, ' idx=', SMUGGLE_idx[i])
        fname = 'image_'+phS2R35+str(SMUGGLE_idx[i])+'.npy'
        try:
            image = np.load(fname)
        except:
            image = gen_image(SMUGGLE_idx[i], phS2R35, lvl)
            np.save(fname, image)
        
        ax[1][i].imshow(image.swapaxes(0, 1), extent=extent)
    
    for x in ax.ravel():
        x.axes.xaxis.set_ticks([])
        x.axes.yaxis.set_ticks([])
    
    ax[1][2].plot([6.5, 8.5], [-8, -8], c='w', lw=2)
    ax[1][2].text(7.5, -7.5, r'$2\,\textrm{kpc}$', c='w', ha='center')

    ax[0][0].set_title(r'$t=1\,\textrm{Gyr}$')
    ax[0][1].set_title(r'$t=2\,\textrm{Gyr}$')
    ax[0][2].set_title(r'$t=3\,\textrm{Gyr}$')

    ax[0][0].set_ylabel(r'without interstellar medium')
    ax[1][0].set_ylabel(r'with interstellar medium')

    fig.tight_layout()

    fig.savefig('fig1.pdf')


if __name__ == '__main__':
    run()

