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

fields = ['Re', 'Nu', 'IE']
base_dirs = [
            '/home/evan/research/my_papers/bvp_initial_conditions_paper/code/runs/base_post',
            '/home/evan/research/my_papers/bvp_initial_conditions_paper/code/runs/bvp_post'
            ]

info = OrderedDict()
for a, base_dir in enumerate(base_dirs):

    for i, d in enumerate(glob.glob('{:s}/*/'.format(base_dir))):
        ra = d.split('_Ra')[-1].split('_')[0]
        ra += '_{}'.format(a)

        info[ra] = OrderedDict()
        try:
            with h5py.File('{:s}/scalar_plots/scalar_values.h5'.format(d), 'r') as f:
                for k in fields:
                    info[ra][k] = f[k].value
                info[ra]['sim_time'] = f['sim_time'].value
        except:
            print('cannot find file in {:s}'.format(d))

print(info.keys())


plt.figure(figsize=(8, 2.5))
gs     = gridspec.GridSpec(*(1000,1000))
gs_info = (((0,0), 1000, 270), ((0, 350), 1000, 270), ((0, 700), 1000, 270))
for i,k in enumerate(fields):
    ax = plt.subplot(gs.new_subplotspec(*gs_info[i]))
    for ra_info, datum in info.items():
        try:
            good_data = datum[k]
            mean, std = np.mean(good_data), np.std(good_data)
            ra, ind = ra_info.split('_')
            if k == 'IE':
                plt.errorbar(float(ra), 0.5+mean, yerr=std,  color=COLORS[int(ind)])
                plt.scatter(float(ra), 0.5+mean, s=8*(3 - 2*int(ind)), marker='o', color=COLORS[int(ind)], alpha=0.5)
            elif k == 'Nu':
                p=2/3
                plt.errorbar(float(ra), mean/(4e-3*float(ra)**(p)), yerr=std/(4e-3*float(ra)**(p)),  color=COLORS[int(ind)])
                plt.scatter(float(ra), mean/(4e-3*float(ra)**p), s=8*(3 - 2*int(ind)), marker='o', color=COLORS[int(ind)], alpha=0.5)
            elif k == 'Re':
                plt.errorbar(float(ra), mean/float(ra)**(1/2), yerr=std/float(ra)**(1/2),  color=COLORS[int(ind)])
                plt.scatter(float(ra), mean/float(ra)**(1/2), s=8*(3 - 2*int(ind)), marker='o', color=COLORS[int(ind)], alpha=0.5)
            else:
                plt.errorbar(float(ra), mean, yerr=std,  color=COLORS[int(ind)])
                plt.scatter(float(ra), mean, s=8*(3 - 2*int(ind)), marker='o', color=COLORS[int(ind)], alpha=0.5)
        except:
            continue
    plt.xlabel('Ra')
    if k == 'IE':
        ax.set_title(r'$\langle T_1 \rangle - T_{\mathrm{top}}$', fontsize=10)
    elif k == 'Nu':
        label_end = '-{:.2g}'.format(p)
        label_end = '$\\langle\\mathrm{Nu}\\rangle\\mathrm{ Ra}^{' + label_end + '}$'
        ax.set_title(r'{}'.format(label_end) + r' / ($4\cdot 10^{-3}$)',fontsize=10)
    elif k == 'Re':
        ax.set_title(r'$\langle\mathrm{Re}\rangle \mathrm{Ra}^{-1/2}$', fontsize=10)
    else:
        ax.set_title(k)
    plt.xscale('log')
    if k == 'IE' or True:
        plt.yscale('log')
plt.savefig('parameter_space_comparison.png'.format(k), bbox_inches='tight', dpi=200)
