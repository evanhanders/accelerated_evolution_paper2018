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





base_dir = '/home/evanhanders/research/papers/accelerated_evolution/code/runs/bvp_post'
subdir = 'rayleigh_benard_2D_mixed_noSlip_Ra1.30e8_Pr1_a2/'
subdir2 = 'rayleigh_benard_2D_mixed_noSlip_Ra1.30e10_Pr1_a2/'
full_dir = base_dir + '/' + subdir
full_dir2 = base_dir + '/' + subdir2


plt.figure(figsize=(8, 3))
gs     = gridspec.GridSpec(*(1000,1000))

slices1 = h5py.File('{:s}/slices/slices_s81.h5' .format(full_dir), 'r')
slices2 = h5py.File('{:s}/slices/slices_s86.h5'.format(full_dir), 'r')
slices3 = h5py.File('{:s}/slices/slices_s75.h5'.format(full_dir2), 'r')
slices4 = h5py.File('{:s}/slices/slices_s80.h5'.format(full_dir2), 'r')
#slices4 = h5py.File('{:s}/slices/slices_s81.h5'.format(full_dir2), 'r')
scalars = h5py.File('{:s}/scalar_plots/scalar_values.h5'.format(full_dir), 'r')
scalars2 = h5py.File('{:s}/scalar_plots/scalar_values.h5'.format(full_dir2), 'r')

temp1 = slices1['tasks']['T'][10,:,:]
w1 = slices1['tasks']['w'][10,:,:]
u1 = slices1['tasks']['u'][10,:,:]
temp2 = slices2['tasks']['T'][19,:,:]
w2 = slices2['tasks']['w'][19,:,:]
u2 = slices2['tasks']['u'][19,:,:]
temp3 = slices3['tasks']['T'][4,:,:]
w3 = slices3['tasks']['w'][4,:,:]
u3 = slices3['tasks']['u'][4,:,:]
temp4 = slices4['tasks']['T'][5,:,:]
w4 = slices4['tasks']['w'][5,:,:]
u4 = slices4['tasks']['u'][5,:,:]
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

good = (time > time[0]+10)*(time <= time[0]+180)
good2 = time2 > time2[0]+30


axes = []
gs_info = (((0,0), 425, 375), ((525, 0), 425, 375), ((50, 450), 450, 275), ((500, 450), 450, 275),
           ((50, 725), 450, 275), ((500, 725), 450, 275))
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[0])))
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[1])))
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[2])))
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[3])))
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[4])))
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[5])))

axes[0].plot(time[good], nu[good], c='k')
print(time2[0])
#axes[0].axvline(573)
#axes[0].axvline(583.9)
ts = (573, 583.9)
scatter_kwargs = {'s' : 7, 'c': 'black'}
axes[0].scatter(time[(time >= ts[0]-0.025)*(time <= ts[0]+0.025)], nu[(time >= ts[0]-0.025)*(time <= ts[0]+0.025)], **scatter_kwargs)
axes[0].scatter(time[(time >= ts[1]-0.025)*(time <= ts[1]+0.025)], nu[(time >= ts[1]-0.025)*(time <= ts[1]+0.025)], **scatter_kwargs)
axes[0].annotate('Ia', xy=(562, 21.5), fontsize=10)
axes[0].annotate('Ib', xy=(573, 7), fontsize=10)
axes[1].axhline(0, ls='--', c='k')
axes[1].plot(time2[good2], nu2[good2], c='k')
#axes[1].axvline(336)#334.5
#axes[1].axvline(343.6)#344.6
ts = (336, 343.2)
axes[1].scatter(time2[(time2 >= ts[0]-0.025)*(time2 <= ts[0]+0.025)], nu2[(time2 >= ts[0]-0.025)*(time2 <= ts[0]+0.025)], **scatter_kwargs)
axes[1].scatter(time2[(time2 >= ts[1]-0.025)*(time2 <= ts[1]+0.025)], nu2[(time2 >= ts[1]-0.025)*(time2 <= ts[1]+0.025)], **scatter_kwargs)
axes[1].annotate('IIa', xy=(323, 103), fontsize=10)
axes[1].annotate('IIb', xy=(330, -15), fontsize=10)
axes[0].set_xlim(np.min(time[good]), np.max(time[good]))
axes[1].set_xlim(np.min(time2[good2]), np.max(time2[good2]))
axes[1].set_xlabel(r'$t/t_{\mathrm{ff}}$')
#axes[1].set_ylabel(r'$\langle$'+'Nu'+r'$\rangle$', labelpad=0)
#axes[0].set_ylabel(r'$\langle$'+'Nu'+r'$\rangle$')
axes[1].set_ylabel('Nu', labelpad=0)
axes[0].set_ylabel('Nu')
axes[0].set_yticks((6, 10, 14, 18, 22))
axes[1].set_yticks((0, 40, 80, 120))
axes[2].pcolormesh(xx, zz, temp1, cmap='RdBu_r', vmin=-0.5, vmax=-0.432)
xs, zs, us, ws = xx[::50, ::50], zz[::50, ::50], u1[::50, ::50], w1[::50, ::50]
conditional = (xs>0.25)*(xs<0.5)*(zs > 0.1)
axes[2].quiver(xs[conditional], zs[conditional], us[conditional], ws[conditional], units='width', alpha=0.35)
conditional = (xs>0.9)*(xs<1.2)*(zs < 0.9)
axes[2].quiver(xs[conditional], zs[conditional], us[conditional], ws[conditional], units='width', alpha=0.35)
#axes[2].quiver(xs[(xs>0.25)*(xs<1.2)], zs[(xs>0.25)*(xs<1.2)], us[(xs>0.25)*(xs<1.2)], ws[(xs>0.25)*(xs<1.2)], units='width')
axes[2].annotate('Ia', xy=(0.05, 0.05), fontsize=10)
axes[4].pcolormesh(xx, zz, temp2, cmap='RdBu_r', vmin=-0.5, vmax=-0.432)
xs, zs, us, ws = xx[::50, ::50], zz[::50, ::50], u2[::50, ::50], w2[::50, ::50]
conditional = (xs>0.25)*(xs<0.5)*(zs > 0.1)
axes[4].quiver(xs[conditional], zs[conditional], us[conditional], ws[conditional], units='width', alpha=0.35)
conditional = (xs>0.9)*(xs<1.2)*(zs < 0.9)
axes[4].quiver(xs[conditional], zs[conditional], us[conditional], ws[conditional], units='width', alpha=0.35)
axes[4].annotate('Ib', xy=(0.05, 0.05), fontsize=10)
axes[3].pcolormesh(xx2, zz2, temp3, cmap='RdBu_r', vmin=-0.4925, vmax=-0.4810)
xs, zs, us, ws = xx2[::200, ::200], zz2[::200, ::200], u3[::200, ::200], w3[::200, ::200]
conditional = (xs>0.5)*(xs<0.9)*(zs > 0.1)
axes[3].quiver(xs[conditional], zs[conditional], us[conditional], ws[conditional], units='width', alpha=0.35)
conditional = (xs>1.25)*(xs<1.6)*(zs < 0.7)
axes[3].quiver(xs[conditional], zs[conditional], us[conditional], ws[conditional], units='width', alpha=0.35)
axes[3].annotate('IIa', xy=(0.05, 0.05), fontsize=10)
axes[3].set_xlim(0, 2)
axes[5].pcolormesh(xx2, zz2, temp4, cmap='RdBu_r', vmin=-0.4925, vmax=-0.4810)
xs, zs, us, ws = xx2[::200, ::200], zz2[::200, ::200], u4[::200, ::200], w4[::200, ::200]
conditional = (xs>0.65)*(xs<0.9)*(zs > 0.3)
axes[5].quiver(xs[conditional], zs[conditional], us[conditional], ws[conditional], units='width', alpha=0.35)
conditional = (xs>1.2)*(xs<1.5)*(zs < 0.7)
axes[5].quiver(xs[conditional], zs[conditional], us[conditional], ws[conditional], units='width', alpha=0.35)
axes[5].annotate('IIb', xy=(0.05, 0.05), fontsize=10)
axes[5].set_xlim(0, 2)
for i in range(4):
    axes[i+2].set_yticks((0, 0.5, 1))
    if i == 1:
        continue
    for t in axes[i+2].xaxis.get_ticklabels():
        t.set_visible(False)
    for t in axes[i+2].yaxis.get_ticklabels():
        t.set_visible(False)

axes[0].set_xticks((450, 500, 550))

cax1 = plt.subplot(gs.new_subplotspec(*((0, 587), 50, 275)))
cbar1 = matplotlib.colorbar.ColorbarBase(cax1, cmap='RdBu_r', norm=matplotlib.colors.Normalize(vmin=-0.5, vmax=-0.432),
                                         orientation='horizontal')
cbar1.set_ticks(())    
trans = cax1.get_xaxis_transform() # x in data untis, y in axes fraction
cax1.annotate(r'$T_{\mathrm{min}}$', xy=(-0.2, 0.4), xycoords=trans)
cax1.annotate(r'$T_{\mathrm{max}}$', xy=(1.03, 0.4), xycoords=trans)

axes[2].annotate(r'$T_{\mathrm{max}} $= -0.432', xy=(1.25, 0.87), fontsize=8)
axes[2].annotate(r'$T_{\mathrm{min}} \,$= -0.5', xy=(1.25, 0.75), fontsize=8)
#axes[3].annotate(r'$T_{\mathrm{max}} $= -0.481', xy=(0.03, 0.30), fontsize=7)
#axes[3].annotate(r'$T_{\mathrm{min}} \,$= -0.493', xy=(0.03, 0.18), fontsize=7)
axes[3].annotate(r'$T_{\mathrm{max}} $= -0.481', xy=(1.25, 0.87), fontsize=8)
axes[3].annotate(r'$T_{\mathrm{min}} \,$= -0.493', xy=(1.25, 0.75), fontsize=8)

axes[3].set_xlabel('x')
axes[3].set_ylabel('z', labelpad=2)

axes[0].annotate(r'$S = 10^5$', xy=(425, 21), fontsize=10)
axes[1].annotate(r'$S = 10^7$', xy=(215, 100), fontsize=10)


plt.savefig('oscillating_plumes.png', bbox_inches='tight', dpi=400)
