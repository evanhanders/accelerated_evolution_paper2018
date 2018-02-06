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
plt.rc('font',family='Times New Roman')
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

base_dirs_pre = [
            '/home/evan/research/my_papers/bvp_initial_conditions_paper/code/runs/base_pre',
            '/home/evan/research/my_papers/bvp_initial_conditions_paper/code/runs/bvp_pre'
            ]

base_dirs_post = [
            '/home/evan/research/my_papers/bvp_initial_conditions_paper/code/runs/base_post',
            '/home/evan/research/my_papers/bvp_initial_conditions_paper/code/runs/bvp_post'
            ]
fields = ['Nu', 'Re', 'IE', 'KE', 'TE']

ra_runs = '1.30e8'
ra_runs = '6.01e7'

pre_flux_file = None
##### INFO grabbing from post-plot buddy files
info = OrderedDict()
for a, base_dir in enumerate(base_dirs_pre + base_dirs_post):
    print(base_dir)

    for i, d in enumerate(glob.glob('{:s}/*Ra{:s}*/'.format(base_dir, ra_runs))):
        ra = d.split('_Ra')[-1].split('_')[0]
        ra += '_{}'.format(a)
        if a == 1 and ra_runs in d:
            pre_flux_file = h5py.File('{:s}/bvp_plots/profile_dict_file_0000.h5'.format(d), 'r')

        info[ra] = OrderedDict()
        try:
            with h5py.File('{:s}/scalar_plots/scalar_values.h5'.format(d), 'r') as f:
                for k in fields:
                    info[ra][k+'_scalar'] = f[k].value
                print('hi')
                info[ra]['sim_time'] = f['sim_time'].value
        except:
            print('cannot find scalar file in {:s}'.format(d))

        try:
            with h5py.File('{:s}/profile_plots/profile_info.h5'.format(d), 'r') as f:
                for k in f.keys():
                    info[ra][k+'_profile'] = f[k].value
        except:
            print('cannot find profile file in {:s}'.format(d))


plt.figure(figsize=(8, 5))
gs     = gridspec.GridSpec(*(1000,1000))

#####################################
# Top two plots
#####################################

times = info['{}_2'.format(ra_runs)]['sim_time'][-1], info['{}_3'.format(ra_runs)]['sim_time'][-1]
time_ratio = times[0]/(times[0]+times[1])

#Calculate how big (a) and (b) are
size_1 = int(950 * time_ratio)
size_2 = int(950 * (1 - time_ratio))

gs_info = (((0,0), 450, size_1), ((0, 950-size_2),450, size_2))
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

# Time traces
for i in range(4):
    this_label = '{}_{}'.format(ra_runs, i)
    if i == 1: #Break apart pre- and post- bvp
        t = info[this_label]['sim_time']
        f = 0.5 + info[this_label]['IE_scalar']
        df = np.abs(f[1:] - f[:-1])
        t_bvp_ind = np.argmax(df) + 1
        df[t_bvp_ind-1] = 0
        t_bvp_ind2 = np.argmax(df) + 1
        axes[0].axvline(info['{}_1'.format(ra_runs)]['sim_time'][-1], ls='--', c='k')
        axes[i % 2].plot(info[this_label]['sim_time'][:t_bvp_ind], info[this_label]['KE_scalar'][:t_bvp_ind], c='k')
        axes[i % 2].plot(info[this_label]['sim_time'][t_bvp_ind:t_bvp_ind2], info[this_label]['KE_scalar'][t_bvp_ind:t_bvp_ind2], c='k')
        axes[i % 2].plot(info[this_label]['sim_time'][t_bvp_ind2:], info[this_label]['KE_scalar'][t_bvp_ind2:], c='k')
        axes[i % 2].set_yscale('log')
        axes[i % 2].set_ylim(1e-3, 1e-1)

        axes_share[i % 2].plot(info[this_label]['sim_time'][:t_bvp_ind], 0.5 + info[this_label]['IE_scalar'][:t_bvp_ind], c='r')
        axes_share[i % 2].plot(info[this_label]['sim_time'][t_bvp_ind:t_bvp_ind2], 0.5 + info[this_label]['IE_scalar'][t_bvp_ind:t_bvp_ind2], c='r')
        axes_share[i % 2].plot(info[this_label]['sim_time'][t_bvp_ind2:], 0.5 + info[this_label]['IE_scalar'][t_bvp_ind2:], c='r')
#        axes_share[i % 2].set_yscale('log')
        axes_share[i % 2].set_ylim(1e-2, 5e-1)

    else:
        if i == 2:
            axes[0].axhline(np.mean(info[this_label]['KE_scalar']), c= 'k', ls = '--')
            axes_share[0].axhline(np.mean(info[this_label]['IE_scalar']) + 0.5, c= 'r', dashes=(2,2))
            axes[1].axhline(np.mean(info[this_label]['KE_scalar']), c= 'k', ls = '--')
            axes_share[1].axhline(np.mean(info[this_label]['IE_scalar']) + 0.5, c= 'r', dashes=(2,2))
            

        axes[i % 2].plot(info[this_label]['sim_time'], info[this_label]['KE_scalar'], c='k')
        axes[i % 2].set_yscale('log')
        axes[i % 2].set_ylim(1e-3, 1e-1)
        if i > 1:
            axes[i % 2].axvline(info[this_label]['sim_time'][0], ls='-')

        axes_share[i % 2].plot(info[this_label]['sim_time'], 0.5 + info[this_label]['IE_scalar'], c='red')
#        axes_share[i % 2].set_yscale('log')
        axes_share[i % 2].set_ylim(1e-2, 6e-1)

#Axes formatting
axes[0].set_xlim(info['{}_0'.format(ra_runs)]['sim_time'][0], info['{}_2'.format(ra_runs)]['sim_time'][-1])
axes[1].set_xlim(info['{}_1'.format(ra_runs)]['sim_time'][0], info['{}_3'.format(ra_runs)]['sim_time'][-1])
#x_ticks = [0, info['{}_3'.format(ra_runs)]['sim_time'][-1], 2000, 4000, 6000, 8000, 10000]
x_ticks = [0, info['{}_3'.format(ra_runs)]['sim_time'][-1], 2000, 4000, 6000, 7500]
axes[0].set_xticks(x_ticks)
x_ticks = [0, info['{}_3'.format(ra_runs)]['sim_time'][-1]]
axes[1].set_xticks(x_ticks)
axes[0].set_ylabel(r'$\mathrm{Kinetic\,\, Energy}$')
axes[0].set_xlabel('Simulation Time (freefall units)', labelpad=0)
x, y = axes[0].xaxis.get_label().get_position()
print(x, y)
axes[0].xaxis.get_label().set_position((x/time_ratio, y))
axes_share[1].set_ylabel(r'$T_1 - T_{\mathrm{top}}$', rotation=270, color='red', labelpad=15)
for tick in axes[1].get_yticklabels():
    tick.set_size(0)
axes[1].yaxis.set_ticks_position('none')
for tick in axes_share[0].get_yticklabels():
    tick.set_size(0)
axes_share[0].yaxis.set_ticks_position('none')
for tick in axes_share[1].get_yticklabels():
    tick.set_color('red')


spines = ['bottom', 'top', 'right', 'left']
axis_names   = ['x', 'y']


axes[0].annotate(r'$\mathrm{(a)}$', (200, 7e-2), fontsize=10)
axes[1].annotate(r'$\mathrm{(b)}$', (400, 7e-2), fontsize=10)

##################################
# Bottom three plots
###################################
#Plot 1
f_conv_color = 'blueviolet'
f_cond_color = 'firebrick'
f_sum_color  = 'black'
f_conv_color2 = 'plum'
f_cond_color2 = 'coral'
f_sum_color2  = 'darkgray'
axes = []
axes = []
gs_info = (((550,0), 450, 267), ((550, 366), 450, 267), ((550, 800), 450, 200))
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[0])))
this_label = '{}_0'.format(ra_runs)
#enth = info[this_label]['enth_flux_profile'][1,:]*np.sqrt(float(ra_runs))
#kappa = info[this_label]['kappa_flux_profile'][1,:]*np.sqrt(float(ra_runs))
#sum_f = (enth + kappa)
print([print(k) for k in pre_flux_file.keys()])
enth = pre_flux_file['enth_flux_IVP'].value*np.sqrt(float(ra_runs))
sum_f = pre_flux_file['tot_flux_IVP'].value*np.sqrt(float(ra_runs))
kappa = sum_f - enth
z = pre_flux_file['z'].value



axes[-1].axhline(1, c='k', ls='-.')
axes[-1].axhline(0, c='orange', ls='--')
axes[-1].plot(z, enth,  c=f_conv_color)
axes[-1].plot(z, kappa, c=f_cond_color)
axes[-1].plot(z, sum_f, c=f_sum_color)
y_ticks = np.array([0, 1, np.ceil(np.max(sum_f))])
axes[-1].set_yticks(y_ticks)
axes[-1].set_ylabel(r'$\mathrm{Flux}\cdot\sqrt{\mathrm{Ra \,Pr}}$')
x_ticks = np.array([0, 0.5, 1])
axes[-1].set_xticks(x_ticks)
axes[-1].set_xlabel('z')
[axes[-1].spines[s].set_color('orange') for s in spines]
[axes[-1].tick_params(axis=axis, colors='orange') for axis in axis_names]
[t.set_color('k') for t in axes[-1].get_xticklabels()]
[t.set_color('k') for t in axes[-1].get_yticklabels()]
axes[-1].annotate(r'$\mathrm{(c)}$', (0.04, 10.8), fontsize=10)

#Plot 2
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[1])))
this_label = '{}_2'.format(ra_runs)
enth = info[this_label]['enth_flux_profile'][0,:]*np.sqrt(float(ra_runs))
kappa = info[this_label]['kappa_flux_profile'][0,:]*np.sqrt(float(ra_runs))
sum_f = (enth + kappa)
axes[-1].axhline(1, c='k', ls='--')
axes[-1].axhline(0, c='k')
axes[-1].plot(info[this_label]['z_profile'], enth,  c=f_conv_color, lw=1)
axes[-1].plot(info[this_label]['z_profile'], kappa, c=f_cond_color, lw=1, label='Rundown')
axes[-1].plot(info[this_label]['z_profile'], sum_f, c=f_sum_color, lw=1)
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
axes[-1].plot(info[this_label]['z_profile'], enth, c=f_conv_color2, lw=2, dashes= (2, 1))
axes[-1].plot(info[this_label]['z_profile'], kappa, c=f_cond_color2, lw=2, dashes = (2, 1), label='BVP')
axes[-1].plot(info[this_label]['z_profile'], sum_f, color=f_sum_color2, lw=2, dashes=(2,1))
y_ticks = np.array([0, 0.5, 1])
axes[-1].set_ylabel(r'$\mathrm{Flux}\cdot\sqrt{\mathrm{Ra \,Pr}}$')
axes[-1].set_yticks(y_ticks)
x_ticks = np.array([0, 0.5, 1])
axes[-1].set_xticks(x_ticks)
axes[-1].set_xlabel('z')
plt.legend(frameon=False, loc='center', fontsize=10)
axes[-1].annotate(r'$\mathrm{(d)}$', (0.45, 0.06), fontsize=10)

#Plot 3
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[2])))
axes[-1].axhline(0, c='k')
axes[-1].plot(info[this_label]['z_profile'], ((enth-base_enth)), c=f_conv_color)
axes[-1].plot(info[this_label]['z_profile'], ((kappa-base_kappa)), c=f_cond_color)
axes[-1].plot(info[this_label]['z_profile'], ((sum_f - base_sum_f)), c=f_sum_color)
x_ticks = np.array([0, 0.5, 1])
axes[-1].set_xticks(x_ticks)
axes[-1].set_ylabel(r'$\mathrm{(BVP - Rundown)}\cdot\sqrt{\mathrm{Ra \,Pr}}$')
axes[-1].annotate(r'$\mathrm{(e)}$', (0.04, -0.028), fontsize=10)
axes[-1].set_xlabel('z')


plt.savefig('time_trace.png'.format(k), bbox_inches='tight', dpi=200)
