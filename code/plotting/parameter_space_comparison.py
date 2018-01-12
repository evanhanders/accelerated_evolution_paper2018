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
pNu = 0.2  
pRe = 0.45
pIE = -0.2
for i,k in enumerate(fields):
    ax = plt.subplot(gs.new_subplotspec(*gs_info[i]))
    bx = ax.twiny()
    for j in range(len(base_dirs)):
        ra_list = []
        mean_list = []
        max_list  = []
        min_list  = []
        m_max_list = []
        m_min_list = []
        for ra_info, datum in info.items():
            try:
                ra, ind = ra_info.split('_')
                ind = int(ind)
                if ind != j:
                    continue

                time = datum['sim_time']
                good_data = datum[k]
                mean, std = np.mean(good_data), np.std(good_data)
                d_max, d_min = mean + std, mean - std #np.min(good_data), np.max(good_data)

                line_fit = np.polyfit(time, good_data, 1)
                line = line_fit[1] + line_fit[0] * time
                m_max, m_min = np.max(line), np.min(line)

                ra_list += [ra]
                mean_list += [mean]
                max_list += [d_max]
                min_list += [d_min]
                m_max_list += [m_max]
                m_min_list += [m_min]

            except:
                continue
        ra_list = np.array(ra_list, dtype=np.float64)*(1.075-0.05*j)
        mean_list = np.array(mean_list, dtype=np.float64)
        max_list = np.array(max_list, dtype=np.float64)
        min_list = np.array(min_list, dtype=np.float64)
        m_max_list = np.array(m_max_list, dtype=np.float64)
        m_min_list = np.array(m_min_list, dtype=np.float64)



        if j >= 2:
            mrkr = '*'
            threeD = True
        else:
            mrkr = 'o'
            threeD = False
        ind = int(j) % 2

        if threeD:
            s = 20*(3 - 2*int(ind))
            alph = 0.7
            if ind == 0:
              label2='3D'
        else:
            s = 6*(3 - 2*int(ind))
            alph = 1
            if ind == 0:
                label='Rundown'
            else:
                label='BVP'
            if ind == 0:
                label2='2D'
        if k == 'IE':
            p = pIE
            mean_list += 0.5
            min_list += 0.5
            max_list += 0.5
            m_min_list += 0.5
            m_max_list += 0.5
            bx.scatter(ra_list/ra_crit, mean_list/ra_list**(p), s=0, alpha=0)
            ax.vlines(ra_list, ymin=min_list/ra_list**p, ymax=max_list/ra_list**(p), color=[COLORS[ind]]*len(min_list), zorder=j)
            ax.scatter(ra_list, mean_list/ra_list**(p), s=s, alpha=alph, color=COLORS[int(ind)], marker=mrkr, zorder=j)
            ax.errorbar(ra_list, mean_list/ra_list**p, yerr=np.array((mean_list-m_min_list, m_max_list-mean_list))/ra_list**p,  ecolor=COLORS[int(ind)], fmt='none', zorder=j)

        elif k == 'Nu':
            p = pNu
            bx.scatter(ra_list/ra_crit, mean_list/ra_list**(p), s=0, alpha=0)
            ax.vlines(ra_list, ymin=min_list/ra_list**p, ymax=max_list/ra_list**(p), color=[COLORS[ind]]*len(min_list), zorder=j)
            if ind == 0:
                ax.scatter(ra_list, mean_list/ra_list**(p), s=s, alpha=alph, color=COLORS[int(ind)], marker=mrkr, zorder=j, label=label2)
            else:
                ax.scatter(ra_list, mean_list/ra_list**(p), s=s, alpha=alph, color=COLORS[int(ind)], marker=mrkr, zorder=j)
            ax.errorbar(ra_list, mean_list/ra_list**p, yerr=np.array((mean_list-m_min_list, m_max_list-mean_list))/ra_list**p,  ecolor=COLORS[int(ind)], fmt='none', zorder=j)
        elif k == 'Re':
            p = pRe
            bx.scatter(ra_list/ra_crit, mean_list/ra_list**(p), s=0, alpha=0)
            ax.vlines(ra_list, ymin=min_list/ra_list**p, ymax=max_list/ra_list**(p), color=[COLORS[ind]]*len(min_list), zorder=j)
            if not threeD:
                ax.scatter(ra_list, mean_list/ra_list**(p), s=s, alpha=alph, color=COLORS[int(ind)], marker=mrkr, zorder=j, label=label)
            else:
                ax.scatter(ra_list, mean_list/ra_list**(p), s=s, alpha=alph, color=COLORS[int(ind)], marker=mrkr, zorder=j)
            ax.errorbar(ra_list, mean_list/ra_list**p, yerr=np.array((mean_list-m_min_list, m_max_list-mean_list))/ra_list**p,  ecolor=COLORS[int(ind)], fmt='none', zorder=j)
        else:
            bx.scatter(float(ra)/ra_crit, mean, s=0, alpha=0)
            ax.errorbar(float(ra), mean, yerr=std,  color=COLORS[int(ind)])
            ax.scatter(float(ra), mean, s=s, marker=mrkr, color=COLORS[int(ind)], alpha=0.75)
    ax.set_xlabel(r'$\mathrm{Ra}$', labelpad=-2)
    bx.set_xlabel(r'$\mathrm{S}$', labelpad=2)
    ax.legend(fontsize=8, loc='upper left', ncol=2, scatterpoints=1, handlelength=1, frameon=False)

    if k == 'IE':
        ax.annotate(r'$\mathrm{(c)}$', (1.8e3, 1.82), fontsize=10)
        label_end = '{:.2g}'.format(-pIE)
        label_end = '$(\\langle T\\rangle - T_{\mathrm{top}})\\mathrm{ Ra}^{' + label_end + '}$'
        ax.set_ylabel(r'{}'.format(label_end), fontsize=10, labelpad=4)
#        ax.set_ylabel(r'$\langle T_1 \rangle - T_{\mathrm{top}}$', fontsize=10, labelpad=4)
        ax.set_ylim(1, 2)
    elif k == 'Nu':
        ax.annotate(r'$\mathrm{(a)}$', (1.8e3, 0.225), fontsize=10)
        label_end = '-{:.2g}'.format(pNu)
        label_end = '$\\langle\\mathrm{Nu}\\rangle\\mathrm{ Ra}^{' + label_end + '}$'
        ax.set_ylabel(r'{}'.format(label_end), fontsize=10, labelpad=0)
        ax.set_ylim(2e-1, 5e-1)
    elif k == 'Re':
        ax.annotate(r'$\mathrm{(b)}$', (1e8, 0.12), fontsize=10)
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
