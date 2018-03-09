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
plt.rc('font',family='Times New Roman')

COLORS=['indigo', 'red']
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
        fancy = False
        if ra == '1.30e10':
            fancy = True
        ra += '_{}'.format(a)

        info[ra] = OrderedDict()
        try:
            with h5py.File('{:s}/scalar_plots/scalar_values.h5'.format(d), 'r') as f:
                for k in fields:
                    info[ra][k] = f[k].value
                    if 'Nu' in k:
                        print(a, 'Nu', ra, np.mean(f[k].value))
                info[ra]['sim_time'] = f['sim_time'].value
                if fancy:
                    where = info[ra]['sim_time'] > 209.8
                    for k in info[ra].keys():
                        info[ra][k] = info[ra][k][where]
        except:
            print('cannot find file in {:s}'.format(d))

print(info.keys())


plt.figure(figsize=(8, 4))
gs     = gridspec.GridSpec(*(1000,1000))
gs_info = (((100,0), 400, 250), ((100, 370), 400, 250), ((100, 750), 400, 250))
ra_crit = 1295.78
pNu = 0#0.2  
pRe = 0#0.45
pIE = 0#-0.2
for i,k in enumerate(fields):
    ax = plt.subplot(gs.new_subplotspec(*gs_info[i]))
    bx = ax.twiny()
    ax.axvline(2.79e8, ls='--', c='k')
    ax.fill_between(np.logspace(np.log10(2.79e8), 15, 2), 1e-5, 1e15, color='grey', alpha=0.4)
    ax.set_xlim(1e3, 2e10)
    for j in range(len(base_dirs)):
        if j == 0 or j == 2:
            continue
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
        ra_list = np.array(ra_list, dtype=np.float64)
        mean_list = np.array(mean_list, dtype=np.float64)
        max_list = np.array(max_list, dtype=np.float64)
        min_list = np.array(min_list, dtype=np.float64)
        m_max_list = np.array(m_max_list, dtype=np.float64)
        m_min_list = np.array(m_min_list, dtype=np.float64)



        if j >= 2:
            mrkr = '*'
            threeD = True
            label = '3D'
            color  = 'red'
            s = 20
        else:
            mrkr = 'o'
            threeD = False
            label  = '2D'
            color  = 'indigo'
            s = 10
        if k == 'IE':
            p = pIE
            mean_list += 0.5
            min_list += 0.5
            max_list += 0.5
            m_min_list += 0.5
            m_max_list += 0.5
            bx.scatter(ra_list/ra_crit, mean_list/ra_list**(p), s=0, alpha=0)
            ax.vlines(ra_list, ymin=min_list/ra_list**p, ymax=max_list/ra_list**(p), color=[color]*len(min_list), zorder=j)
            ax.scatter(ra_list, mean_list/ra_list**(p), s=s,  color=color, marker=mrkr, zorder=j)
            ax.errorbar(ra_list, mean_list/ra_list**p, yerr=np.array((mean_list-m_min_list, m_max_list-mean_list))/ra_list**p,  ecolor=color, fmt='none', zorder=j)
            ind = ra_list == 1.30e8
            ax.scatter(ra_list[ind], mean_list[ind]/ra_list[ind]**(p), s=s*1.5,  color='black', marker='s', zorder=1e10)

        elif k == 'Nu':
            p = pNu
            bx.scatter(ra_list/ra_crit, mean_list/ra_list**(p), s=0, alpha=0)
            ax.vlines(ra_list, ymin=min_list/ra_list**p, ymax=max_list/ra_list**(p), color=[color]*len(min_list), zorder=j)
            ax.scatter(ra_list, mean_list/ra_list**(p), s=s,  color=color, marker=mrkr, zorder=j, label=label)
            ax.errorbar(ra_list, mean_list/ra_list**p, yerr=np.array((mean_list-m_min_list, m_max_list-mean_list))/ra_list**p,  ecolor=color, fmt='none', zorder=j)
            ind = ra_list == 1.30e8
            ax.scatter(ra_list[ind], mean_list[ind]/ra_list[ind]**(p), s=s*1.5,  color='black', marker='s', zorder=1e10)
        elif k == 'Re':
            p = pRe
            bx.scatter(ra_list/ra_crit, mean_list/ra_list**(p), s=0, alpha=0)
            ax.vlines(ra_list, ymin=min_list/ra_list**p, ymax=max_list/ra_list**(p), color=[color]*len(min_list), zorder=j)
            ax.scatter(ra_list, mean_list/ra_list**(p), s=s,  color=color, marker=mrkr, zorder=j)
            ax.errorbar(ra_list, mean_list/ra_list**p, yerr=np.array((mean_list-m_min_list, m_max_list-mean_list))/ra_list**p,  ecolor=color, fmt='none', zorder=j)
            ind = ra_list == 1.30e8
            ax.scatter(ra_list[ind], mean_list[ind]/ra_list[ind]**(p), s=s*1.5,  color='black', marker='s', zorder=1e10)
        else:
            bx.scatter(float(ra)/ra_crit, mean, s=0, alpha=0)
            ax.errorbar(float(ra), mean, yerr=std,  color=color)
            ax.scatter(float(ra), mean, s=s, marker=mrkr, color=color, alpha=0.75)
    bx.set_xlabel('S')
    ax.legend(fontsize=8, loc='lower right', ncol=2, scatterpoints=1, handlelength=1, frameon=True)
#    ax.grid(which='major')

    if k == 'IE':
        ax.annotate(r'$\mathrm{(c)}$', (1e9, 3.5e-1), fontsize=10)
        label_end = '{:.2g}'.format(-pIE)
        label_end = '$(\\langle T\\rangle - T_{\mathrm{top}})$'#\\mathrm{ Ra}^{' + label_end + '}$'
        ax.set_ylabel(r'{}'.format(label_end))
        ax.set_ylim(1e-2, 6e-1)
#        ax.set_ylabel(r'$\langle T_1 \rangle - T_{\mathrm{top}}$', fontsize=10, labelpad=4)
#        ax.set_ylim(1, 2)
    elif k == 'Nu':
        ax.annotate(r'$\mathrm{(a)}$', (2e3, 5e1), fontsize=10)
        label_end = '-{:.2g}'.format(pNu)
        label_end = '$\\langle\\mathrm{Nu}\\rangle$'#\\mathrm{ Ra}^{' + label_end + '}$'
        ax.set_ylabel(r'{}'.format(label_end))
        ax.set_ylim(1, 1e2)
#        ax.set_ylim(2e-1, 5e-1)
    elif k == 'Re':
        ax.annotate(r'$\mathrm{(b)}$', (2e3, 5e3), fontsize=10)
        label_end = '-{:.3g}'.format(pRe)
        label_end = '$\\langle\\mathrm{Re}\\rangle$'#\\mathrm{ Ra}^{' + label_end + '}$'
        ax.set_ylabel(r'{}'.format(label_end))
        ax.set_ylim(1, 2e4)
#        ax.set_title(r'$\langle\mathrm{Re}\rangle \mathrm{Ra}^{-1/2}$', fontsize=10)
#        ax.set_ylim(1e-1, 3e-1)
    else:
        ax.set_title(k)

    [t.set_fontsize(10) for t in ax.get_xticklabels()]
    [t.set_fontsize(10) for t in ax.get_yticklabels()]
    [t.set_fontsize(10) for t in bx.get_xticklabels()]
    [t.set_fontsize(10) for t in bx.get_yticklabels()]
    for j,t in enumerate(ax.get_xticklabels()):
        t.set_visible(0)

    ax.set_xscale('log')
    bx.set_xscale('log')
    ax.set_yscale('log')
    bx.set_yscale('log')


    bx.set_xlim(1e3/ra_crit, 2e10/ra_crit)
    plt.axes(bx)
    plt.xticks(np.array((1e1, 1e3, 1e5, 1e7)))


gs_info = (((500,0), 400, 250), ((500, 370), 400, 250), ((500, 750), 400, 250))

for i,k in enumerate(fields):
    ax = plt.subplot(gs.new_subplotspec(*gs_info[i]))
    if i == 0:
        ax.annotate(r'$\mathrm{(d)}$', (1.3e3, 0.014), fontsize=10)
    if i == 1:
        ax.annotate(r'$\mathrm{(e)}$', (1.3e3, 0.014), fontsize=10)
    if i == 2:
        ax.annotate(r'$\mathrm{(f)}$', (1.3e3, 0.014), fontsize=10)
    mean_lists = []
    for j in range(len(base_dirs)):
        ra_list = []
        mean_list = []
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
        ra_list = np.array(ra_list, dtype=np.float64)
        mean_list = np.array(mean_list, dtype=np.float64)
        ra_list, mean_list =  zip(*sorted(zip(ra_list, mean_list)))
        mean_lists.append([np.array(ra_list), np.array(mean_list)])
    twoD_len  = np.min([len(mean_lists[1][1]), len(mean_lists[0][1])])
    twoD_diff = (mean_lists[1][1][:twoD_len] - mean_lists[0][1][:twoD_len])/mean_lists[0][1][:twoD_len]
    print(mean_lists[0][1][:twoD_len], mean_lists[1][1][:twoD_len])
    threeD_len  = np.min([len(mean_lists[3][1]), len(mean_lists[2][1])])
    threeD_diff = (mean_lists[3][1][:threeD_len] - mean_lists[2][1][:threeD_len])/mean_lists[2][1][:threeD_len]

    ax.axhline(0, ls='--', c='k')
    ax.axvline(2.79e8, ls='--', c='k')
    ax.fill_between(np.logspace(np.log10(2.79e8), 15, 2), -0.05, 0.05, color='grey', alpha=0.4)
    ind = mean_lists[1][0][:twoD_len] == 1.30e8
    ax.scatter(mean_lists[1][0][:twoD_len], twoD_diff, color='indigo')
    ax.scatter(mean_lists[1][0][:twoD_len][ind], twoD_diff[ind], s=s,  color='black', marker='s', zorder=1e10)
    ax.scatter(mean_lists[3][0][:threeD_len], threeD_diff, marker='*', color='red')
#    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(1e3, 2e10)
    plt.axes(ax)
    plt.xticks(np.array((1e4, 1e6, 1e8, 1e10)))
    ax.set_xlabel('Ra')
    ax.set_ylabel ('( AE - SE ) / SE')

    ax.set_ylim(-0.02, 0.02)

    ax.set_yticks([-0.01, 0, 0.01])

    [t.set_fontsize(10) for t in ax.get_xticklabels()]
    [t.set_fontsize(10) for t in ax.get_yticklabels()]

plt.savefig('parameter_space_comparison.png'.format(k), dpi=200, bbox_inches='tight')
