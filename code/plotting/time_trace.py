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
#ra_runs = '6.01e7'

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


plt.figure(figsize=(8, 7.5))
gs     = gridspec.GridSpec(*(1000,1000))

#####################################
# Top two plots
#####################################

times = info['{}_2'.format(ra_runs)]['sim_time'][-1], info['{}_3'.format(ra_runs)]['sim_time'][-1]
time_ratio = times[0]/(times[0]+times[1])

#Calculate how big (a) and (b) are
size_1 = int(950 * time_ratio)
size_2 = int(950 * (1 - time_ratio))

gs_info = (((30,0), 270, 660), ((30, 680), 270, 320), ((380, 500), 270, 500))
print(size_1, size_2, gs_info)
axes = []
axes_share = []

for i in range(3):
    axes.append(plt.subplot(gs.new_subplotspec(*gs_info[i])))
    axes_share.append(axes[-1].twinx())
#Overlaying bold areas on time traces
axes[0].fill_between(np.linspace(50, 76, 10), 1e-5, 1e15, color='orange', alpha=1)
axes[2].fill_between(np.linspace(50, 76, 10), 1e-5, 1e15, color='orange', alpha=1)

axes[1].fill_between(np.linspace(1.133e4, 1.183e4, 10), 1e-5, 1e15, color='green', alpha=0.4)
axes[2].fill_between(np.linspace(410, 910.6, 10), 1e-5, 1e15, color='green', alpha=0.4)
# Time traces

this_label = '{}_{}'.format(ra_runs, 1)
t = info[this_label]['sim_time']
f = 0.5 + info[this_label]['IE_scalar']
df = np.abs(f[1:] - f[:-1])
t_bvp_ind = np.argmax(df) + 1
df[t_bvp_ind-1] = 0
t_bvp_ind2 = np.argmax(df) + 1
df[t_bvp_ind2-1] = 0
t_bvp_ind3 = np.argmax(df) + 1

for i in [1, 3]:
    this_label = '{}_{}'.format(ra_runs, i)
    axes[0].axvline(info['{}_1'.format(ra_runs)]['sim_time'][-1], ls='--', c='k')
    axes[2].axvline(info['{}_1'.format(ra_runs)]['sim_time'][-1], ls='--', c='k')
    axes[2].plot(info[this_label]['sim_time'][:t_bvp_ind], info[this_label]['KE_scalar'][:t_bvp_ind], c='k', lw=2)
    axes[2].plot(info[this_label]['sim_time'][t_bvp_ind:t_bvp_ind2], info[this_label]['KE_scalar'][t_bvp_ind:t_bvp_ind2], c='k', lw=2)
    axes[2].plot(info[this_label]['sim_time'][t_bvp_ind2:t_bvp_ind3], info[this_label]['KE_scalar'][t_bvp_ind2:t_bvp_ind3], c='k', lw=2)
    axes[2].plot(info[this_label]['sim_time'][t_bvp_ind3:], info[this_label]['KE_scalar'][t_bvp_ind3:], c='k', lw=2)
    axes[2].set_yscale('log')
    axes[2].set_ylim(3e-3, 1e-1)

    axes_share[2].plot(info[this_label]['sim_time'][:t_bvp_ind], 0.5 + info[this_label]['IE_scalar'][:t_bvp_ind], c='b', lw=2)
    axes_share[2].plot(info[this_label]['sim_time'][t_bvp_ind:t_bvp_ind2], 0.5 + info[this_label]['IE_scalar'][t_bvp_ind:t_bvp_ind2], c='b', lw=2)
    axes_share[2].plot(info[this_label]['sim_time'][t_bvp_ind2:t_bvp_ind3], 0.5 + info[this_label]['IE_scalar'][t_bvp_ind2:t_bvp_ind3], c='b', lw=2)
    axes_share[2].plot(info[this_label]['sim_time'][t_bvp_ind3:], 0.5 + info[this_label]['IE_scalar'][t_bvp_ind3:], c='b', lw=2)
    axes_share[2].set_ylim(1e-2, 5e-1)
    axes_share[2].set_yscale('log')

for i in [0, 2]:
    this_label = '{}_{}'.format(ra_runs, i)
    if i == 2:
        axes[0].axhline(np.mean(info[this_label]['KE_scalar']), c= 'k', ls = '--')
        axes_share[0].axhline(np.mean(info[this_label]['IE_scalar']) + 0.5, c= 'b', dashes=(2,2))
        axes[1].axhline(np.mean(info[this_label]['KE_scalar']), c= 'k', ls = '--')
        axes_share[1].axhline(np.mean(info[this_label]['IE_scalar']) + 0.5, c= 'b', dashes=(2,2))
        axes[2].axhline(np.mean(info[this_label]['KE_scalar']), c= 'k', ls = '--')
        axes_share[2].axhline(np.mean(info[this_label]['IE_scalar']) + 0.5, c= 'b', dashes=(2,2))

    axes[0].plot(info[this_label]['sim_time'], info[this_label]['KE_scalar'], c='k', lw=2)
    axes[0].set_yscale('log')
    axes[0].set_ylim(3e-3, 1e-1)

    axes[1].plot(info[this_label]['sim_time'], info[this_label]['KE_scalar'], c='k', lw=2)
    axes[1].set_yscale('log')
    axes[1].set_ylim(3e-3, 1e-1)

    axes_share[0].plot(info[this_label]['sim_time'], 0.5 + info[this_label]['IE_scalar'], c='blue', lw=2)
    axes_share[0].set_yscale('log')
    axes_share[0].set_ylim(1e-2, 5e-1)

    axes_share[1].plot(info[this_label]['sim_time'], 0.5 + info[this_label]['IE_scalar'], c='blue', lw=2)
    axes_share[1].set_yscale('log')
    axes_share[1].set_ylim(1e-2, 5e-1)
axes_share[1].spines['right'].set_color('blue')
axes_share[2].spines['right'].set_color('blue')
axes_share[1].tick_params(color='b', which='both')
axes_share[2].tick_params(color='b', which='both')

#remove bad spines in middle of figure.
axes[0].spines['right'].set_visible(False)
axes_share[0].spines['right'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes_share[1].spines['left'].set_visible(False)
axes[1].axes.get_yaxis().set_visible(False)
axes[0].yaxis.tick_left()
axes_share[1].yaxis.tick_right()
axes[1].set_yticks([])

#set xlims
axes[0].set_xlim(0, 1200)
axes[1].set_xlim(1.183e4-590, 1.183e4)
axes[2].set_xlim(0, 910.6)


x_ticks = [0, 250, 500, 750, 1000]
axes[0].set_xticks(x_ticks)
x_ticks = [1.133e4, 1.133e4+250, 1.183e4]
axes[1].set_xticks(x_ticks)
x_ticks = [0, 250, 500, 750]
axes[2].set_xticks(x_ticks)
axes[0].set_ylabel(r'$\mathrm{Kinetic\,\, Energy}$', labelpad=-10)
axes[2].set_ylabel(r'$\mathrm{Kinetic\,\, Energy}$', labelpad=-1)
axes[0].set_xlabel('Simulation Time (freefall units)', labelpad=0)
axes[0].xaxis.set_label_coords(0.70, -0.125)
x, y = axes[0].xaxis.get_label().get_position()
print(x, y)
axes[0].xaxis.get_label().set_position((x/time_ratio, y))
axes_share[1].set_ylabel(r'$\langle T_1\rangle - T_{\mathrm{top}}$', rotation=270, color='blue', labelpad=15)
axes_share[2].set_ylabel(r'$\langle T_1\rangle - T_{\mathrm{top}}$', rotation=270, color='blue', labelpad=10)
#for tick in axes[2].get_yticklabels():
#    tick.set_size(0)
#axes[2].yaxis.set_ticks_position('none')
for tick in axes_share[0].get_yticklabels():
    tick.set_size(0)
for tick in axes[2].get_yticklabels():
    tick.set_size(0)
for tick in axes_share[2].get_yticklabels():
    tick.set_size(0)
axes_share[0].yaxis.set_ticks_position('none')
for tick in axes_share[1].get_yticklabels():
    tick.set_color('blue')
for tick in axes_share[2].get_yticklabels():
    tick.set_color('blue')

d = 0.02
kwargs = dict(transform=axes[0].transAxes, color='k', clip_on=False)
axes[0].plot((1,1+d/2, 1+d, 1 + 3*d/2), (0, +4*d, -4*d, 0), **kwargs)
axes[0].plot((1,1+d/2, 1+d, 1 + 3*d/2), (1, 1+4*d, 1-4*d, 1), **kwargs)


spines = ['bottom', 'top', 'right', 'left']
axis_names   = ['x', 'y']

y_ticks = np.array([1e-2, 1e-1, 5e-1])
plt.yticks(y_ticks, (r'$10^{-2}$', r'$10^{-1}$', r'$5\cdot 10^{-1}$'))
plt.axes(axes_share[1])
y_ticks = np.array([1e-2, 1e-1, 5e-1])
plt.yticks(y_ticks, ('', r'$10^{-1}$', ''))

plt.axes(axes[0])
y_ticks = np.array([3e-3, 1e-2, 1e-1])
plt.yticks(y_ticks, (r'$3\cdot 10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'))

Ra=1.30e8
t_therm = np.sqrt(Ra)
new_axis = axes[0].twiny()
plt.xlim(0, 1200/t_therm)
x_ticks = np.array([250, 500, 750, 1000])/t_therm
plt.xticks(x_ticks, (r'$2.2\cdot 10^{-2}$', r'$4.4\cdot 10^{-2}$', r'$6.6\cdot 10^{-2}$', r'$8.8 \cdot 10^{-2}$'))
plt.xlabel('Simulation Time (thermal units)')
new_axis.xaxis.set_label_coords(0.758, 1.18)
new_axis.spines['right'].set_visible(False)
new_axis = axes[1].twiny()
axes[1].set_xlim(1.183e4-590, 1.183e4)
plt.xlim((1.183e4-590)/t_therm, 1.183e4/t_therm)
x_ticks = np.array([11330, 11580, 11830])/t_therm
plt.xticks(x_ticks, ('0.99', '1.02', '1.04'))
new_axis.spines['left'].set_visible(False)

new_axis = axes[2].twiny()
new_axis.set_xlim(0, 910.6/t_therm)
x_ticks = np.array([0, 250, 500, 750])/t_therm
plt.xticks(x_ticks, ('0', '0.02', '0.04', '0.06'))
new_axis.tick_params(axis='x', which='major', pad=0)
new_axis.spines['left'].set_visible(False)


trans = axes[1].get_xaxis_transform() # x in data untis, y in axes fraction
axes[1].annotate(r'$10^{-2}$', xy=(11840, 0.03), xycoords=trans, color='blue')
axes[1].annotate(r'$5\cdot 10^{-1}$', xy=(11840, 0.94), xycoords=trans, color='blue')


trans = axes[0].get_xaxis_transform() # x in data untis, y in axes fraction
ann = axes[0].annotate('(a)', xy=(-160, 0.98 ), xycoords=trans)
#axes[0].annotate(r'$\mathrm{(a)}$', (-50, 7e-2), fontsize=10)
trans = axes[2].get_xaxis_transform() # x in data untis, y in axes fraction
axes[2].annotate('(c)', xy=(-75, 0.98), xycoords=trans)




axes[2].arrow(135, 4.8e-2, -24, 0, fc='k', ec='k', head_width=5e-3, head_length=20)
axes[2].arrow(280, 1.4e-2, 0, -2e-3, fc='k', ec='k', head_width=10, head_length=2e-3)
axes[2].arrow(365, 1.4e-2, 0, -2e-3, fc='k', ec='k', head_width=10, head_length=2e-3)
axes[2].annotate('1', xy=(150, 0.76), xycoords=trans)
axes[2].annotate('2', xy=(268, 0.48), xycoords=trans)
axes[2].annotate('3', xy=(353, 0.48), xycoords=trans)

##################################
# Bottom three plots
###################################
#Plot 1
f_conv_color = 'darkturquoise'
f_cond_color = 'red'
f_sum_color  = 'black'
f_conv_color2 = 'plum'
f_cond_color2 = 'coral'
f_sum_color2  = 'darkgray'
axes = []
axes = []
gs_info = (((380,0), 270, 400), ((730, 0), 270, 400), ((730, 550), 300, 450))
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



axes[-1].axhline(1, c='k', ls=':')
axes[-1].axhline(0, c='k')
axes[-1].plot(z, enth,  c=f_conv_color, label=r'$\mathrm{F}_{\mathrm{conv}}$', lw=2)
axes[-1].plot(z, kappa, c=f_cond_color, label=r'$\mathrm{F}_{\mathrm{cond}}$', lw=2)
axes[-1].plot(z, sum_f, c=f_sum_color, label=r'$\mathrm{F}_{\mathrm{tot}}$', lw=2)
y_ticks = np.array([0, 5, 10, 15, np.ceil(np.max(sum_f))])
axes[-1].set_yticks(y_ticks)
axes[-1].set_ylabel(r'$\mathrm{Flux}/\mathcal{P}$')
x_ticks = np.array([0, 0.5, 1])
axes[-1].set_xticks(x_ticks)
axes[-1].set_xlabel('z')
#[axes[-1].spines[s].set_color('orange') for s in spines]
#[axes[-1].tick_params(axis=axis, colors='orange') for axis in axis_names]
[t.set_color('k') for t in axes[-1].get_xticklabels()]
[t.set_color('k') for t in axes[-1].get_yticklabels()]
trans = axes[-1].get_xaxis_transform() # x in data untis, y in axes fraction
axes[-1].annotate('(b)', xy=(-0.18, 0.98), xycoords=trans)
axes[-1].annotate(r'Transient', (0.015, 17), color='orange', fontsize=14)
axes[-1].legend(loc='lower center', frameon=False, columnspacing=1, fontsize=10, ncol=3, borderpad=0)

#Plot 2
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[1])))
this_label = '{}_2'.format(ra_runs)
enth = info[this_label]['enth_flux_profile'][0,:]*np.sqrt(float(ra_runs))
kappa = info[this_label]['kappa_flux_profile'][0,:]*np.sqrt(float(ra_runs))
sum_f = (enth + kappa)
axes[-1].set_ylabel(r'$\mathrm{Flux}/\mathcal{P}$')
y_ticks = np.array([0, 0.25, 0.5, 0.75, 1])
axes[-1].set_yticks(y_ticks)
x_ticks = np.array([0, 0.5, 1])
axes[-1].set_xticks(x_ticks)
axes[-1].set_xlim(0, 1)
axes[-1].set_xlabel('z')
#[axes[-1].spines[s].set_color('green') for s in spines]
#[axes[-1].tick_params(axis=axis, colors='green') for axis in axis_names]
[t.set_color('k') for t in axes[-1].get_xticklabels()]
[t.set_color('k') for t in axes[-1].get_yticklabels()]
axes[-1].annotate(r'Evolved', (0.37, 0.075), color='green', fontsize=14)

#axes.append(plt.subplot(gs.new_subplotspec(*gs_info[2])))
this_label = '{}_3'.format(ra_runs)
base_enth, base_kappa, base_sum_f = enth, kappa, sum_f
enth = info[this_label]['enth_flux_profile'][0,:]*np.sqrt(float(ra_runs))
kappa = info[this_label]['kappa_flux_profile'][0,:]*np.sqrt(float(ra_runs))
sum_f = (enth + kappa)
axes[-1].plot(info[this_label]['z_profile'], enth,  c=f_conv_color, lw=2)
axes[-1].plot(info[this_label]['z_profile'], kappa, c=f_cond_color, lw=2, label='SE')
axes[-1].plot(info[this_label]['z_profile'], sum_f, c=f_sum_color, lw=2)
trans = axes[-1].get_xaxis_transform() # x in data untis, y in axes fraction
axes[-1].annotate('(d)', xy=(-0.225, 0.98), xycoords=trans)

#Plot 3
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[2])))
axes[-1].axhline(0, c='k')
axes[-1].plot(info[this_label]['z_profile'], ((enth-base_enth)), c=f_conv_color, lw=2)
axes[-1].plot(info[this_label]['z_profile'], ((kappa-base_kappa)), c=f_cond_color, lw=2)
axes[-1].plot(info[this_label]['z_profile'], ((sum_f - base_sum_f)), c=f_sum_color, lw=2)
x_ticks = np.array([0, 0.5, 1])
axes[-1].set_xticks(x_ticks)
y_ticks = np.array([-0.06, -0.04, -0.02, 0, 0.02])
axes[-1].set_yticks(y_ticks)
axes[-1].set_ylabel(r'$(\mathrm{Flux}_{\mathrm{AE}} - \mathrm{Flux}_{\mathrm{SE}})/\mathcal{P}$')
axes[-1].set_xlabel('z')
trans = axes[-1].get_xaxis_transform() # x in data untis, y in axes fraction
axes[-1].annotate('(e)', xy=(-0.225, 1), xycoords=trans)


plt.savefig('time_trace.png'.format(k), bbox_inches='tight', dpi=200)
