"""
Script for plotting a parameter space study of Nu v Ra.

Usage:
    parameter_space_plots.py --calculate
    parameter_space_plots.py

Options:
    --calculate     If flagged, touch dedalus output files and do time averages.  If not, use post-processed data from before
"""


import matplotlib   
matplotlib.rcParams.update({'font.size': 11})
import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter
import os
import numpy as np
import glob
from docopt import docopt
from mpi4py import MPI
comm_world = MPI.COMM_WORLD
from collections import OrderedDict
import h5py
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d


def calculate_CDF(x, pdf):
    """ Calculate the CDF of a PDF using trapezoidal rule integration """
    dx = np.diff(x)
    new_x = x[0:-1] + dx
    d_cdf   = (dx/2) * (pdf[0:-1] + pdf[1:])
    cdf = np.zeros_like(d_cdf)
    for i in range(len(cdf)):
        cdf[i] = np.sum(d_cdf[:i+1])
    return new_x, cdf

def ks_test(x1, y1, N1, x2, y2, N2, n=1000):
    x_range = [np.min(x1), np.max(x1)]
    if np.min(x2) > x_range[0]:
        x_range[0] = np.min(x2)
    if np.max(x2) < x_range[1]:
        x_range[1] = np.max(x2)

    x_points = np.linspace(*tuple(x_range), n)

    f1 = interp1d(x1, y1, bounds_error=False, assume_sorted=True)#, fill_value='extrapolate')
    f2 = interp1d(x2, y2, bounds_error=False, assume_sorted=True)#, fill_value='extrapolate')

    y1_interp = f1(x_points)
    y2_interp = f2(x_points)
    diff = np.abs(y1_interp - y2_interp)
    
    max_diff = np.max(diff)
    return max_diff, np.sqrt((N1+N2)/(N1*N2))



import dedalus.public as de

EPSILON_ORDER=[1e-7, 1e0, 1e-4, 5e-1]
FORCE_WRITE=False

COLORS=['indigo', 'orange']
MARKERS=['s', 'o', 'd', '*']
MARKERSIZE=[5,4,5,7]

MARKERS_2=['p', '^', '8']
COLORS_2=['peru', 'gold', 'teal']
MARKERSIZE_2=[5,5,5]

base_dirs_post = [
            '/home/evan/research/my_papers/bvp_initial_conditions_paper/code/runs/base_post',
            '/home/evan/research/my_papers/bvp_initial_conditions_paper/code/runs/bvp_post'
            ]
ra_runs = '1.30e8'
ra_runs = '6.01e7'

info = OrderedDict()
for a, base_dir in enumerate(base_dirs_post):
    print(base_dir)

    for i, d in enumerate(glob.glob('{:s}/*Ra{:s}*/'.format(base_dir, ra_runs))):
        ra = d.split('_Ra')[-1].split('_')[0]
        ra += '_{}'.format(a)

        info[ra] = OrderedDict()

        try:
            with h5py.File('{:s}/profile_plots/profile_info.h5'.format(d), 'r') as f:
                for k in f.keys():
                    info[ra][k+'_profile'] = f[k].value
        except:
            print('cannot find profile file in {:s}'.format(d))
        try:
            with h5py.File('{:s}/new_pdfs/slice_pdf_data.h5'.format(d), 'r') as f:
            #with h5py.File('{:s}/slice_pdfs/slice_pdf_data.h5'.format(d), 'r') as f:
                for k in f.keys():
                    info[ra][k+'_pdf'] = f[k].value
        except:
            print('cannot find pdf file in {:s}'.format(d))



print(info['{}_0'.format(ra_runs)].keys())
plt.figure(figsize=(8, 1.5))
gs     = gridspec.GridSpec(*(1000,1000))


spines = ['bottom', 'top', 'right', 'left']
axis_names   = ['x', 'y']
# Bottom three plots
#Plot 1
axes = []
gs_info = (((0,0), 1000, 267), ((0, 366), 1000, 267), ((0, 733), 1000, 267))
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[0])))
base_label = '{}_0'.format(ra_runs)
bvp_label = '{}_1'.format(ra_runs)

axes[-1].fill_between(info[base_label]['w_xs_pdf'], 0, info[base_label]['w_pdf_pdf']*info[base_label]['w_denorm_pdf'], color='blue', alpha=0.4)
axes[-1].plot(info[base_label]['w_xs_pdf'], info[base_label]['w_pdf_pdf']*info[base_label]['w_denorm_pdf'], c='blue')
axes[-1].fill_between(info[bvp_label]['w_xs_pdf'], 0, info[bvp_label]['w_pdf_pdf']*info[bvp_label]['w_denorm_pdf'], color='red', alpha=0.4)
axes[-1].plot(info[bvp_label]['w_xs_pdf'], info[bvp_label]['w_pdf_pdf']*info[bvp_label]['w_denorm_pdf'], c='red')
axes[-1].set_xlim(np.min(info[bvp_label]['w_xs_pdf']), np.max(info[bvp_label]['w_xs_pdf']))
axes[-1].set_xlabel('Vertical velocity', labelpad=-1)
axes[-1].set_ylabel('Frequency')
axes[-1].set_yscale('log')
axes[-1].annotate(r'$\mathrm{(a)}$', (-0.15, 1e8), fontsize=10)

for tick in axes[-1].get_xticklabels():
    tick.set_rotation(45)



plt.figure()
for k in ['u', 'w', 'w*T']:
    cdf_x_bvp, cdf_y_bvp = calculate_CDF(info[bvp_label]['{}_xs_pdf'.format(k)], info[bvp_label]['{}_pdf_pdf'.format(k)])
    cdf_x_base, cdf_y_base = calculate_CDF(info[base_label]['{}_xs_pdf'.format(k)], info[base_label]['{}_pdf_pdf'.format(k)])
    plt.plot(cdf_x_bvp, cdf_y_bvp)
    plt.plot(cdf_x_base, cdf_y_base)
    plt.yscale('log')

    max_diff, comp = ks_test(cdf_x_bvp, cdf_y_bvp, info[bvp_label]['u_denorm_pdf'],\
                             cdf_x_base, cdf_y_base, info[base_label]['u_denorm_pdf'])
    print('for {}, max diff is {}'.format(k, max_diff))
#    print(max_diff, comp, max_diff/comp)
    plt.show()


#Plot 2
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[1])))
axes[-1].fill_between(info[base_label]['u_xs_pdf'], 0, info[base_label]['u_pdf_pdf']*info[base_label]['u_denorm_pdf'], color='blue', alpha=0.4)
axes[-1].plot(info[base_label]['u_xs_pdf'], info[base_label]['u_pdf_pdf']*info[base_label]['u_denorm_pdf'], c='blue')
axes[-1].fill_between(info[bvp_label]['u_xs_pdf'], 0, info[bvp_label]['u_pdf_pdf']*info[bvp_label]['u_denorm_pdf'], color='red', alpha=0.4)
axes[-1].plot(info[bvp_label]['u_xs_pdf'], info[bvp_label]['u_pdf_pdf']*info[bvp_label]['u_denorm_pdf'], c='red')
axes[-1].set_xlim(np.min(info[bvp_label]['u_xs_pdf']), np.max(info[bvp_label]['u_xs_pdf']))
axes[-1].set_xlabel('Horizontal velocity', labelpad=-0.5)
axes[-1].set_yscale('log')
axes[-1].annotate(r'$\mathrm{(b)}$', (-0.175, 2e7), fontsize=10)


for tick in axes[-1].get_xticklabels():
    tick.set_rotation(45)

##Plot 3
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[2])))
axes[-1].fill_between(info[base_label]['w*T_xs_pdf'], 0, info[base_label]['w*T_pdf_pdf']*info[base_label]['w*T_denorm_pdf'], color='blue', alpha=0.4)
axes[-1].plot(info[base_label]['w*T_xs_pdf'], info[base_label]['w*T_pdf_pdf']*info[base_label]['w*T_denorm_pdf'], c='blue', label='Rundown')
axes[-1].fill_between(info[bvp_label]['w*T_xs_pdf'], 0, info[bvp_label]['w*T_pdf_pdf']*info[bvp_label]['w*T_denorm_pdf'], color='red', alpha=0.4)
axes[-1].plot(info[bvp_label]['w*T_xs_pdf'], info[bvp_label]['w*T_pdf_pdf']*info[bvp_label]['w*T_denorm_pdf'], c='red', label='BVP')
axes[-1].set_xlim(np.min(info[bvp_label]['w*T_xs_pdf']), np.max(info[bvp_label]['w*T_xs_pdf']))
plt.legend(frameon=False, fontsize=8, loc='upper right')
axes[-1].set_xlabel(r'$w(T - \bar{T})$', labelpad=-5)
axes[-1].set_yscale('log')
axes[-1].annotate(r'$\mathrm{(c)}$', (-1.2e-3, 1e8), fontsize=10)


for tick in axes[-1].get_xticklabels():
    tick.set_rotation(45)




plt.savefig('pdf_comparison.png'.format(k), bbox_inches='tight', dpi=200)
