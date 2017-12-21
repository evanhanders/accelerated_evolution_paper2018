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

fields = ['Nu', 'Re', 'IE', 'KE', 'TE']
base_dirs_pre = [
            '/home/evan/research/my_papers/bvp_initial_conditions_paper/code/runs/base_pre',
            '/home/evan/research/my_papers/bvp_initial_conditions_paper/code/runs/bvp_pre'
            ]

base_dirs_post = [
            '/home/evan/research/my_papers/bvp_initial_conditions_paper/code/runs/base_post',
            '/home/evan/research/my_papers/bvp_initial_conditions_paper/code/runs/bvp_post'
            ]

ra_runs = '6.01e4'
ra_runs = '2.79e7'

info = OrderedDict()
for a, base_dir in enumerate(base_dirs_pre + base_dirs_post):
    print(base_dir)

    for i, d in enumerate(glob.glob('{:s}/*Ra{:s}*/'.format(base_dir, ra_runs))):
        ra = d.split('_Ra')[-1].split('_')[0]
        ra += '_{}'.format(a)

        info[ra] = OrderedDict()
        try:
            with h5py.File('{:s}/scalar_plots/scalar_values.h5'.format(d), 'r') as f:
                for k in fields:
                    info[ra][k+'_scalar'] = f[k].value
                info[ra]['sim_time'] = f['sim_time'].value
        except:
            print('cannot find file in {:s}'.format(d))

        try:
            with h5py.File('{:s}/profile_plots/profile_info.h5'.format(d), 'r') as f:
                for k in f.keys():
                    info[ra][k+'_profile'] = f[k].value
        except:
            print('cannot find profile file in {:s}'.format(d))


print(info['{}_0'.format(ra_runs)].keys())
plt.figure(figsize=(8, 5))
gs     = gridspec.GridSpec(*(1000,1000))


# Top two plots
times = info['{}_2'.format(ra_runs)]['sim_time'][-1], info['{}_3'.format(ra_runs)]['sim_time'][-1]
time_ratio = times[0]/(times[0]+times[1])

size_1 = int(1000 * time_ratio)
size_2 = int(1000 * (1 - time_ratio))

gs_info = (((0,0), 450, size_1), ((0, 1000-size_2),450, size_2))
print(size_1, size_2, gs_info)
axes = []
axes_share = []

for i in range(4):
    if i < 2:
        axes.append(plt.subplot(gs.new_subplotspec(*gs_info[i])))
        axes_share.append(axes[-1].twinx())
#Overlaying bold areas on time traces
this_label = '{}_0'.format(ra_runs, i)
t = info[this_label]['sim_time']
ke = info[this_label]['KE_scalar']
ie = info[this_label]['IE_scalar'] + 0.5
axes[0].plot(t[(t > 50)*(t <=100)], ke[(t > 50)*(t <=100)], c='orange', lw=4)
axes_share[0].plot(t[(t > 50)*(t <=100)], ie[(t > 50)*(t <=100)], c='orange', lw=4)

this_label = '{}_2'.format(ra_runs, i)
t = info[this_label]['sim_time']
ke = info[this_label]['KE_scalar']
ie = info[this_label]['IE_scalar'] + 0.5
axes[0].plot(t, ke, c='green', lw=4)
axes_share[0].plot(t, ie, c='green', lw=4)

this_label = '{}_3'.format(ra_runs, i)
t = info[this_label]['sim_time']
ke = info[this_label]['KE_scalar']
ie = info[this_label]['IE_scalar'] + 0.5
axes[1].plot(t, ke, c='green', lw=4)
axes_share[1].plot(t, ie, c='green', lw=4)


for i in range(4):
    this_label = '{}_{}'.format(ra_runs, i)
    axes[i % 2].plot(info[this_label]['sim_time'], info[this_label]['KE_scalar'], c='k')
    axes[i % 2].set_yscale('log')
    axes[i % 2].set_ylim(1e-3, 1e-1)
    if i > 1:
        axes[i % 2].axvline(info[this_label]['sim_time'][0], ls='-.')

    axes_share[i % 2].plot(info[this_label]['sim_time'], 0.5 + info[this_label]['IE_scalar'], c='red')
    axes_share[i % 2].set_yscale('log')
    axes_share[i % 2].set_ylim(1e-2, 1e0)

#Axes formatting
axes[0].set_xlim(info['{}_0'.format(ra_runs)]['sim_time'][0], info['{}_2'.format(ra_runs)]['sim_time'][-1])
axes[1].set_xlim(info['{}_1'.format(ra_runs)]['sim_time'][0], info['{}_3'.format(ra_runs)]['sim_time'][-1])
axes[0].set_ylabel('Kinetic Energy')
axes[0].set_xlabel('Simulation Time (freefall units)', labelpad=0)
x, y = axes[0].xaxis.get_label().get_position()
print(x, y)
axes[0].xaxis.get_label().set_position((x/time_ratio, y))
axes_share[1].set_ylabel(r'$T_1 - T_{\mathrm{top}}$', rotation=270, color='green')
for tick in axes[1].get_yticklabels():
    tick.set_size(0)
axes[1].yaxis.set_ticks_position('none')
for tick in axes_share[0].get_yticklabels():
    tick.set_size(0)
axes_share[0].yaxis.set_ticks_position('none')
for tick in axes_share[1].get_yticklabels():
    tick.set_color('g')


spines = ['bottom', 'top', 'right', 'left']
axis_names   = ['x', 'y']
# Bottom three plots
#Plot 1
axes = []
gs_info = (((550,0), 450, 267), ((550, 366), 450, 267), ((550, 800), 450, 200))
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[0])))
this_label = '{}_0'.format(ra_runs)
enth = info[this_label]['enth_flux_profile'][1,:]*np.sqrt(float(ra_runs))
kappa = info[this_label]['kappa_flux_profile'][1,:]*np.sqrt(float(ra_runs))
sum_f = (enth + kappa)
axes[-1].axhline(1, c='k', ls='--')
axes[-1].axhline(0, c='k')
axes[-1].plot(info[this_label]['z_profile'], enth,  c='darkblue')
axes[-1].plot(info[this_label]['z_profile'], kappa, c='darkred')
axes[-1].plot(info[this_label]['z_profile'], sum_f, color='g')
y_ticks = np.array([0, 0.5, 1, np.ceil(np.max(sum_f))])
axes[-1].set_yticks(y_ticks)
axes[-1].set_ylabel(r'$\mathrm{Flux}\cdot\sqrt{\mathrm{Ra \,Pr}}$')
x_ticks = np.array([0, 0.5, 1])
axes[-1].set_xticks(x_ticks)
axes[-1].set_xlabel('z')
[axes[-1].spines[s].set_color('orange') for s in spines]
[axes[-1].tick_params(axis=axis, colors='orange') for axis in axis_names]
[t.set_color('k') for t in axes[-1].get_xticklabels()]
[t.set_color('k') for t in axes[-1].get_yticklabels()]

#Plot 2
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[1])))
this_label = '{}_2'.format(ra_runs)
enth = info[this_label]['enth_flux_profile'][0,:]*np.sqrt(float(ra_runs))
kappa = info[this_label]['kappa_flux_profile'][0,:]*np.sqrt(float(ra_runs))
sum_f = (enth + kappa)
axes[-1].axhline(1, c='k', ls='--')
axes[-1].axhline(0, c='k')
axes[-1].plot(info[this_label]['z_profile'], enth,  c='darkblue', lw=3)
axes[-1].plot(info[this_label]['z_profile'], kappa, c='darkred', lw=3, label='Rundown')
axes[-1].plot(info[this_label]['z_profile'], sum_f, color='g', lw=3)
y_ticks = np.array([0, 0.5, 1])
axes[-1].set_yticks(y_ticks)
x_ticks = np.array([0, 0.5, 1])
axes[-1].set_xticks(x_ticks)
axes[-1].set_xlabel('z')
[axes[-1].spines[s].set_color('green') for s in spines]
[axes[-1].tick_params(axis=axis, colors='green') for axis in axis_names]
[t.set_color('k') for t in axes[-1].get_xticklabels()]
[t.set_color('k') for t in axes[-1].get_yticklabels()]

#axes.append(plt.subplot(gs.new_subplotspec(*gs_info[2])))
this_label = '{}_3'.format(ra_runs)
base_enth, base_kappa, base_sum_f = enth, kappa, sum_f
enth = info[this_label]['enth_flux_profile'][0,:]*np.sqrt(float(ra_runs))
kappa = info[this_label]['kappa_flux_profile'][0,:]*np.sqrt(float(ra_runs))
sum_f = (enth + kappa)
axes[-1].axhline(1, c='k', ls='--')
axes[-1].axhline(0, c='k')
axes[-1].plot(info[this_label]['z_profile'], enth, c='lightskyblue', lw=1, dashes= (4, 1.5))
axes[-1].plot(info[this_label]['z_profile'], kappa, c='salmon', lw=1, dashes = (4, 1.5), label='BVP')
axes[-1].plot(info[this_label]['z_profile'], sum_f, color='springgreen', lw=1, dashes=(4,1.5))
y_ticks = np.array([0, 0.5, 1])
axes[-1].set_ylabel(r'$\mathrm{Flux}\cdot\sqrt{\mathrm{Ra \,Pr}}$')
axes[-1].set_yticks(y_ticks)
x_ticks = np.array([0, 0.5, 1])
axes[-1].set_xticks(x_ticks)
axes[-1].set_xlabel('z')
plt.legend(frameon=False, loc='center', fontsize=10)

#Plot 3
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[2])))
axes[-1].axhline(0, c='k')
axes[-1].plot(info[this_label]['z_profile'], ((enth-base_enth)), c='darkblue')
axes[-1].plot(info[this_label]['z_profile'], ((kappa-base_kappa)), c='darkred')
axes[-1].plot(info[this_label]['z_profile'], ((sum_f - base_sum_f)), c='green')
axes[-1].set_ylabel(r'$\mathrm{(BVP - Rundown)}\cdot\sqrt{\mathrm{Ra \,Pr}}$')


plt.savefig('time_trace.png'.format(k), bbox_inches='tight', dpi=200)
