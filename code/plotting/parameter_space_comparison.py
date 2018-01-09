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

COLORS=['indigo', 'orange']
MARKERS=['s', 'o', 'd', '*']
MARKERSIZE=[5,4,5,7]

fields = ['Nu', 'Re', 'IE']
base_dirs = [
            '/home/evan/research/my_papers/bvp_initial_conditions_paper/code/runs/base_post',
            '/home/evan/research/my_papers/bvp_initial_conditions_paper/code/runs/bvp_post',
            '/home/evan/research/my_papers/bvp_initial_conditions_paper/code/runs/3d/base_post',
            '/home/evan/research/my_papers/bvp_initial_conditions_paper/code/runs/3d/bvp_post'
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


plt.figure(figsize=(8, 1.5))
gs     = gridspec.GridSpec(*(1000,1000))
gs_info = (((100,0), 800, 250), ((100, 370), 800, 250), ((100, 750), 800, 250))
ra_crit = 1295.78
for i,k in enumerate(fields):
    ax = plt.subplot(gs.new_subplotspec(*gs_info[i]))
    bx = ax.twiny()
    for ra_info, datum in info.items():
        try:
            good_data = datum[k]
            mean, std = np.mean(good_data), np.std(good_data)
            ra, ind = ra_info.split('_')
            ind = int(ind)
            if ind >= 2:
                mrkr = '*'
                threeD = True
            else:
                mrkr = 'o'
                threeD = False
            ind = int(ind) % 2

            if threeD:
                s = 24*(3 - 2*int(ind))
                alph = 0.7
            else:
                s = 12*(3 - 2*int(ind))
                alph = 1
            if k == 'IE':
                bx.scatter(float(ra)/ra_crit, 0.5+mean, s=0, alpha=0)
                ax.errorbar(float(ra), 0.5+mean, yerr=std,  color=COLORS[int(ind)])
                ax.scatter(float(ra), 0.5+mean, s=s, marker=mrkr, color=COLORS[int(ind)], alpha=alph)
            elif k == 'Nu':
                p=1/5#2/7#2/3
                bx.scatter(float(ra)/ra_crit, mean/float(ra)**(p), s=0, alpha=0)
                ax.errorbar(float(ra), mean/(float(ra)**(p)), yerr=std/(float(ra)**(p)),  color=COLORS[int(ind)])
                ax.scatter(float(ra), mean/(float(ra)**p), s=s, marker=mrkr, color=COLORS[int(ind)], alpha=alph)
            elif k == 'Re':
                pRe = 0.45
                bx.scatter(float(ra)/ra_crit, mean/float(ra)**(pRe), s=0, alpha=0)
                ax.errorbar(float(ra), mean/float(ra)**(pRe), yerr=std/float(ra)**(pRe),  color=COLORS[int(ind)])
                ax.scatter(float(ra), mean/float(ra)**(pRe), s=s, marker=mrkr, color=COLORS[int(ind)], alpha=alph)
            else:
                bx.scatter(float(ra)/ra_crit, mean, s=0, alpha=0)
                ax.errorbar(float(ra), mean, yerr=std,  color=COLORS[int(ind)])
                ax.scatter(float(ra), mean, s=s, marker=mrkr, color=COLORS[int(ind)], alpha=0.75)
        except:
            continue
    ax.set_xlabel(r'$\mathrm{Ra}$', labelpad=-2)
    bx.set_xlabel(r'$\mathrm{S}$', labelpad=2)

    if True and k == 'IE':
        ax.set_yscale('log')
        bx.set_yscale('log')

    if k == 'IE':
        ax.set_ylabel(r'$\langle T_1 \rangle - T_{\mathrm{top}}$', fontsize=10, labelpad=4)
        bx.set_ylabel(r'$\langle T_1 \rangle - T_{\mathrm{top}}$', fontsize=10, labelpad=4)
    elif k == 'Nu':
        label_end = '-{:.2g}'.format(p)
        label_end = '$\\langle\\mathrm{Nu}\\rangle\\mathrm{ Ra}^{' + label_end + '}$'
        ax.set_ylabel(r'{}'.format(label_end), fontsize=10, labelpad=0)
        ax.set_ylim(2e-1, 5e-1)
    elif k == 'Re':
        label_end = '-{:.3g}'.format(pRe)
        label_end = '$\\langle\\mathrm{Re}\\rangle\\mathrm{ Ra}^{' + label_end + '}$'
        ax.set_ylabel(r'{}'.format(label_end),fontsize=10, labelpad=0)
#        ax.set_title(r'$\langle\mathrm{Re}\rangle \mathrm{Ra}^{-1/2}$', fontsize=10)
        ax.set_ylim(1e-1, 3e-1)
    else:
        ax.set_title(k)

    [t.set_fontsize(10) for t in ax.get_xticklabels()]
    [t.set_fontsize(10) for t in ax.get_yticklabels()]
    [t.set_fontsize(10) for t in bx.get_xticklabels()]
    [t.set_fontsize(10) for t in bx.get_yticklabels()]
    for j,t in enumerate(ax.get_xticklabels()):
        if j % 2 == 1:
            t.set_visible(0)

    for j,t in enumerate(bx.get_xticklabels()):
        if j % 2 == 1:
            t.set_visible(0)

    ax.set_xscale('log')
    bx.set_xscale('log')

plt.savefig('parameter_space_comparison.png'.format(k), dpi=200, bbox_inches='tight')
