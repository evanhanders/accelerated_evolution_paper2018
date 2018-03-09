import matplotlib.style
import matplotlib
matplotlib.use('Agg')
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





base_dir = '/home/evan/research/my_papers/bvp_initial_conditions_paper/code/runs/bvp_post'
subdir = 'rayleigh_benard_2D_mixed_noSlip_Ra1.30e8_Pr1_a2/'
subdir2 = 'rayleigh_benard_2D_mixed_noSlip_Ra1.30e10_Pr1_a2/'
full_dir = base_dir + '/' + subdir
full_dir2 = base_dir + '/' + subdir2


plt.figure(figsize=(8, 3))
gs     = gridspec.GridSpec(*(1000,1000))

slices1 = h5py.File('{:s}/slices/slices_s81.h5' .format(full_dir), 'r')
slices2 = h5py.File('{:s}/slices/slices_s134.h5'.format(full_dir), 'r')
slices3 = h5py.File('{:s}/slices/slices_s75.h5'.format(full_dir2), 'r')
slices4 = h5py.File('{:s}/slices/slices_s81.h5'.format(full_dir2), 'r')
scalars = h5py.File('{:s}/scalar_plots/scalar_values.h5'.format(full_dir), 'r')
scalars2 = h5py.File('{:s}/scalar_plots/scalar_values.h5'.format(full_dir2), 'r')

temp1 = slices1['tasks']['T'][5,:,:]
temp2 = slices2['tasks']['T'][10,:,:]
temp3 = slices3['tasks']['T'][0,:,:]
temp4 = slices4['tasks']['T'][0,:,:]
x     = slices1['scales']['x']['1.0'].value
z     = slices1['scales']['z']['1.0'].value
x2     = slices3['scales']['x']['1.0'].value
z2     = slices3['scales']['z']['1.0'].value

zz, xx = np.meshgrid(z, x)
zz2, xx2 = np.meshgrid(z2, x2)

time  = scalars['sim_time'].value
nu  = scalars['Nu'].value
re  = scalars['Re'].value

time2  = scalars2['sim_time'].value
nu2  = scalars2['Nu'].value
re2  = scalars2['Re'].value

good = (time > time[0]+30)*(time <= time[0]+200)
good2 = time2 > time2[0]+30


axes = []
gs_info = (((0,0), 425, 400), ((525, 0), 425, 400), ((50, 450), 450, 275), ((500, 450), 450, 275),
           ((50, 725), 450, 275), ((500, 725), 450, 275))
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[0])))
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[1])))
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[2])))
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[3])))
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[4])))
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[5])))

axes[0].plot(time[good], nu[good], c='k')
axes[1].axhline(0, ls='--', c='k')
axes[1].plot(time2[good2], nu2[good2], c='k')
axes[0].set_xlim(np.min(time[good]), np.max(time[good]))
axes[1].set_xlim(np.min(time2[good2]), np.max(time2[good2]))
axes[1].set_xlabel(r'$t/t_{\mathrm{ff}}$')
axes[1].set_ylabel('Nu')
axes[0].set_ylabel('Nu')
axes[0].set_yticks((6, 10, 14, 18, 22))
axes[1].set_yticks((0, 40, 80, 120))
axes[2].pcolormesh(xx, zz, temp1, cmap='RdBu_r', vmin=-0.5, vmax=-0.432)
axes[4].pcolormesh(xx, zz, temp2, cmap='RdBu_r', vmin=-0.5, vmax=-0.432)
axes[3].pcolormesh(xx2, zz2, temp3, cmap='RdBu_r', vmin=-0.4925, vmax=-0.4810, zorder=0)
axes[5].pcolormesh(xx2, zz2, temp4, cmap='RdBu_r', vmin=-0.4925, vmax=-0.4810)
for i in range(4):
    axes[i+2].set_yticks((0, 0.5, 1))
    if i == 1:
        continue
    for t in axes[i+2].xaxis.get_ticklabels():
        t.set_visible(False)
    for t in axes[i+2].yaxis.get_ticklabels():
        t.set_visible(False)

axes[0].set_xticks((450, 500, 550, 600))

cax1 = plt.subplot(gs.new_subplotspec(*((0, 587), 50, 275)))
cbar1 = matplotlib.colorbar.ColorbarBase(cax1, cmap='RdBu_r', norm=matplotlib.colors.Normalize(vmin=-0.5, vmax=-0.432),
                                         orientation='horizontal')
cbar1.set_ticks(())    
trans = cax1.get_xaxis_transform() # x in data untis, y in axes fraction
#cax1.annotate('-0.5', xy=(-0.3, 0.15), xycoords=trans)
#cax1.annotate('-0.432', xy=(1.05, 0.15), xycoords=trans)

axes[2].annotate('-0.432', xy=(1.6, 0.87), fontsize=8)
axes[2].annotate('-0.5', xy=(1.6, 0.77), fontsize=8)
axes[4].annotate('-0.432', xy=(1.6, 0.87), fontsize=8)
axes[4].annotate('-0.5', xy=(1.6, 0.77), fontsize=8)

axes[5].annotate('-0.481', xy=(1.6, 0.87), fontsize=8)
axes[5].annotate('-0.493', xy=(1.6, 0.77), fontsize=8)
axes[3].annotate('-0.481', xy=(1.6, 0.87), fontsize=8)
axes[3].annotate('-0.493', xy=(1.6, 0.77), fontsize=8)

plt.savefig('oscillating_plumes.png', bbox_inches='tight', dpi=300)
