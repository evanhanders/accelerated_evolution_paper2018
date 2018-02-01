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
            with h5py.File('{:s}/slice_pdfs/slice_pdf_data.h5'.format(d), 'r') as f:
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
axes[-1].plot(info[base_label]['z_profile'], info[base_label]['T_profile'][0,:],  c='blue', lw=1, label='Rundown')
axes[-1].plot(info[bvp_label]['z_profile'], info[bvp_label]['T_profile'][0,:], c='red', lw=1, dashes=(4,1.5), label='BVP')
plt.legend(frameon=False, loc='upper right', fontsize=10)
axes[-1].set_xlabel('z')
axes[-1].set_ylabel(r'$T$' + ' profile')
axes[-1].annotate(r'$\mathrm{(a)}$', (0.04, -0.494), fontsize=10)

y_ticks = [-0.5, -0.48, -0.46, -0.44, -0.42]
axes[-1].set_yticks(y_ticks)

#Plot 2
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[1])))

axes[-1].plot(info[base_label]['z_profile'], 100*np.abs(info[base_label]['T_profile'][0,:] - info[bvp_label]['T_profile'][0,:])/np.abs(info[base_label]['T_profile'][0,:]))
y_ticks = np.array([0, 0.1, 0.25])
axes[-1].set_yticks(y_ticks)
x_ticks = np.array([0, 0.5, 1])
axes[-1].set_xticks(x_ticks)
axes[-1].set_xlabel('z')
axes[-1].set_ylabel('% difference')
axes[-1].annotate(r'$\mathrm{(b)}$', (0.04, 0.03), fontsize=10)

##Plot 3
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[2])))
axes[-1].fill_between(info[base_label]['T_xs_pdf'], 0, info[base_label]['T_pdf_pdf']*info[base_label]['T_denorm_pdf'], color='blue', alpha=0.4)
axes[-1].plot(info[base_label]['T_xs_pdf'], info[base_label]['T_pdf_pdf']*info[base_label]['T_denorm_pdf'], c='blue')
axes[-1].fill_between(info[bvp_label]['T_xs_pdf'], 0, info[bvp_label]['T_pdf_pdf']*info[bvp_label]['T_denorm_pdf'], color='red', alpha=0.4)
axes[-1].plot(info[bvp_label]['T_xs_pdf'], info[bvp_label]['T_pdf_pdf']*info[bvp_label]['T_denorm_pdf'], c='red')
x_ticks = np.array([-0.5, -0.45, -0.40])
axes[-1].set_xticks(x_ticks)
axes[-1].set_xlabel('T')
axes[-1].set_ylabel('Frequency')
axes[-1].set_yscale('log')
axes[-1].set_xlim(-0.5, np.max(info[base_label]['T_xs_pdf']))
axes[-1].annotate(r'$\mathrm{(c)}$', (-0.495, 1.2e8), fontsize=10)




plt.savefig('temp_comparison.png'.format(k), bbox_inches='tight', dpi=200)
