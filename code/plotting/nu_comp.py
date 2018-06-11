import matplotlib.style
import matplotlib
#matplotlib.use('Agg')
matplotlib.style.use('classic')
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
from scipy.interpolate import interp1d


import pandas as pd



base_dir = '/home/evanhanders/research/papers/accelerated_evolution/code/runs/'
subdir = 'bvp_pre/rayleigh_benard_2D_mixed_noSlip_Ra1.30e8_Pr1_a2/'
subdir2 = 'base_pre/rayleigh_benard_2D_mixed_noSlip_Ra1.30e8_Pr1_a2/'
subdir3 = 'bvp_post/rayleigh_benard_2D_mixed_noSlip_Ra1.30e8_Pr1_a2/'
full_dir = base_dir + '/' + subdir
full_dir2 = base_dir + '/' + subdir2
full_dir3 = base_dir + '/' + subdir3


fig = plt.figure(figsize=(8, 6))
gs     = gridspec.GridSpec(*(1000,1000))
gs_info = (((30,0), 450, 1000), ((550, 0), 420, 660), ((550, 680), 420, 320) )
axs = []
for i in range(len(gs_info)):
    axs.append(plt.subplot(gs.new_subplotspec(*gs_info[i])))

ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]


scalars = h5py.File('{:s}/scalar_plots/scalar_values.h5'.format(full_dir), 'r')
scalars2 = h5py.File('{:s}/scalar_plots/scalar_values.h5'.format(full_dir2), 'r')
scalars3 = h5py.File('{:s}/scalar_plots/scalar_values.h5'.format(full_dir3), 'r')

time  = scalars['sim_time'].value
nu  = scalars['Nu'].value
re  = scalars['Re'].value
T  = scalars['IE'].value

time2  = scalars2['sim_time'].value
nu2  = scalars2['Nu'].value
re2  = scalars2['Re'].value
T2  = scalars2['IE'].value

time3  = scalars3['sim_time'].value
nu3  = scalars3['Nu'].value
re3  = scalars3['Re'].value
T3  = scalars3['IE'].value

T_final = np.mean(T2[-1000:])
Nu_final = np.mean(nu2[-1000:])

time1 = np.concatenate((time, time3))
nu1 = np.concatenate((nu, nu3))
run_df = pd.DataFrame({'t':    time1[np.argmax(nu1)+50:],
                        'nu':   nu1[np.argmax(nu1)+50:]})
print(run_df)
avg_nu1 = run_df.rolling(window=500, min_periods=50).mean()
avg_nu1 = np.array(avg_nu1)[:,1]
ax1.plot(time1[nu1 > 0], nu1[nu1 > 0],   c='k')
ax1.plot(run_df['t'], avg_nu1,   c='k', ls='--', lw=3, dashes=(5,1))
#ax1.plot(time3, nu3, c='k')

run2_df = pd.DataFrame({'t':    time2[np.argmax(nu2)+5:],
                        'nu':   nu2[np.argmax(nu2)+5:]})
avg_nu2 = run2_df.rolling(window=50, min_periods=5, axis=0).mean()
avg_nu2 = np.array(avg_nu2)[:,1]

ax2.plot(time2[nu2 > 0], nu2[nu2 > 0], c='k')
ax2.plot(run2_df['t'], avg_nu2,   c='k', lw=3, dashes=(5,1))

ax3.plot(time2[nu2 > 0], nu2[nu2 > 0], c='k')
ax3.plot(run2_df['t'], avg_nu2,   c='k', lw=3, dashes=(5,1))

#ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(1e0, 1e3)
ax1.set_ylim(9e-1, 1e3)
#ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(1e0, 660)
ax2.set_ylim(9e-1, 1e3)
ax2.set_xticks([200, 400, 600])


#ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlim(11115, 1.1435e4)
ax3.set_ylim(9e-1, 1e3)
ax3.set_xticks([11235, 11435])
ax3.set_xticklabels(("11235", "11435"))

ax1.annotate("AE", xy=(950, 5e2))
ax3.annotate("SE", xy=(11385, 5e2))

ax2.set_xlabel("Time (freefall units)")
ax2.xaxis.set_label_coords(0.8, -0.125)
ax2.set_ylabel("Nu")
ax1.set_ylabel("Nu")

ax1.axvline(77 , c='k', ls='--')
ax1.axvline(278, c='k', ls='--')
ax1.axvline(360.6, c='k', ls='--')


ax1_2 = ax1.twinx()
ax2_2 = ax2.twinx()
ax3_2 = ax3.twinx()

ax1_2.plot(time,  T+0.5,   c='b', lw=2)
ax1_2.plot(time3, T3+0.5, c='b', lw=2)
ax2_2.plot(time2, T2+0.5, c='b', lw=2)
ax3_2.plot(time2, T2+0.5, c='b', lw=2)

ax1_2.set_yscale('log')
ax2_2.set_yscale('log')
ax3_2.set_yscale('log')

ax3_2.set_ylabel(r'$\langle T \rangle - T_{top}$', color='b')
ax1_2.set_ylabel(r'$\langle T \rangle - T_{top}$', color='b')
for ax in [ax1_2, ax3_2]:
    ax.spines['right'].set_color('b')
    ax.tick_params(axis='y', colors='b')


axs_share = [ax1_2, ax2_2, ax3_2]
for ax in axs_share:
    ax.set_ylim(2e-2, 1e0)

axs[1].spines['right'].set_visible(False)
axs_share[1].spines['right'].set_visible(False)
axs[2].spines['left'].set_visible(False)
axs_share[2].spines['left'].set_visible(False)
axs[2].axes.get_yaxis().set_visible(False)
axs_share[1].axes.get_yaxis().set_visible(False)
axs_share[2].yaxis.tick_right()
axs[2].set_yticks([])



for tick in axs_share[1].get_yticklabels():
    tick.set_size(0)
for tick in axs[2].get_yticklabels():
    tick.set_size(0)

# Add in "time break" lines for bottom plot
d = 0.02
kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot((1,1+d/2, 1+d, 1 + 3*d/2), (0, +4*d, -4*d, 0), **kwargs)
ax2.plot((1,1+d/2, 1+d, 1 + 3*d/2), (1, 1+4*d, 1-4*d, 1), **kwargs)

#Add in shaded box
ax1.fill_between(time1[(time1>400)*(time1<600)], 5e0, 4e1, color='black', alpha=0.25)

#Add horizontal final values
ax1_2.axhline(T_final + 0.5, ls='--', c='blue')
ax2_2.axhline(T_final + 0.5, ls='--', c='blue')
ax1.axhline(Nu_final, ls='--', c='k')
ax2.axhline(Nu_final, ls='--', c='k')




plt.savefig('nu_v_time.png', bbox_inches='tight', dpi=300)
