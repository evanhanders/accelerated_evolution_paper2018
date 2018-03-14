"""
Script for plotting a parameter space study of Nu v Ra.

Usage:
    parameter_space_plots.py --calculate
    parameter_space_plots.py

Options:
    --calculate     If flagged, touch dedalus output files and do time averages.  If not, use post-processed data from before
"""


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



def calculate_CDF(x, pdf):
    """ Calculate the CDF of a PDF using trapezoidal rule integration """
    dx = np.diff(x)
    new_x = x[0:-1] + dx
    d_cdf   = (dx/2) * (pdf[0:-1] + pdf[1:])
    cdf = np.zeros_like(d_cdf)
    for i in range(len(cdf)):
        cdf[i] = np.sum(d_cdf[:i+1])
    return new_x, cdf

def ks_test(x1, y1, N1, x2, y2, N2, n=10000):
    x_range = [np.min(x1), np.max(x1)]
    if np.min(x2) > x_range[0]:
        x_range[0] = np.min(x2)
    if np.max(x2) < x_range[1]:
        x_range[1] = np.max(x2)

    x_points = np.linspace(*tuple(x_range), n)

    f1 = interp1d(x1, y1, bounds_error=False, assume_sorted=True)#, fill_value='extrapolate')
    f2 = interp1d(x2, y2, bounds_error=False, assume_sorted=True)#, fill_value='extrapolate')
    
    y1_interp = f1(x_points)
    y2_interp = f2(x_points)
    print(y1 - y2)
    print(y1_interp - y2_interp)
    diff_true = y1_interp - y2_interp
    diff = np.abs(diff_true)

#    plt.figure()
#    plt.plot(x_points, diff)
#    plt.plot(x1, np.abs(y1 - y2))
#    plt.scatter(x1, np.abs(y1 - y2))
#    plt.show()
    
    max_diff = np.max(diff)
    print(max_diff, diff)
    return max_diff, np.sqrt((N1+N2)/(N1*N2)), x_points, diff_true



EPSILON_ORDER=[1e-7, 1e0, 1e-4, 5e-1]
FORCE_WRITE=False

COLORS=['indigo', 'orange']
MARKERS=['s', 'o', 'd', '*']
MARKERSIZE=[5,4,5,7]

MARKERS_2=['p', '^', '8']
COLORS_2=['peru', 'gold', 'teal']
MARKERSIZE_2=[5,5,5]

base_dirs_post = [
            '/home/evan/research/my_papers/bvp_initial_conditions_paper/code/runs/base_post',
            '/home/evan/research/my_papers/bvp_initial_conditions_paper/code/runs/bvp_post'
            ]
ra_runs = '1.30e8'
#ra_runs = '6.01e7'

info = OrderedDict()
for a, base_dir in enumerate(base_dirs_post):
    print(base_dir)

    for i, d in enumerate(glob.glob('{:s}/*Ra{:s}*/'.format(base_dir, ra_runs))):
        ra = d.split('_Ra')[-1].split('_')[0]
        ra += '_{}'.format(a)

        info[ra] = OrderedDict()

        try:
            with h5py.File('{:s}/profile_plots/profile_info.h5'.format(d), 'r') as f:
                for k in f.keys():
                    info[ra][k+'_profile'] = f[k].value
        except:
            print('cannot find profile file in {:s}'.format(d))
        try:
            with h5py.File('{:s}/slice_pdfs/slice_pdf_data.h5'.format(d), 'r') as f:
                for k in f.keys():
                    info[ra][k+'_pdf'] = f[k].value
        except:
            print('cannot find pdf file in {:s}'.format(d))



print(info['{}_0'.format(ra_runs)].keys())
plt.figure(figsize=(8, 3))
gs     = gridspec.GridSpec(*(1000,1000))


spines = ['bottom', 'top', 'right', 'left']
axis_names   = ['x', 'y']
# Bottom three plots
#Plot 1
axes = []
#gs_info = (((0,0), 1000, 267), ((0, 366), 1000, 267), ((0, 733), 1000, 267))
gs_info = (((0,0), 650, 450), ((725, 0), 275, 450), ((0, 550), 650, 450), ((725, 550), 275, 450))
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[0])))
base_label = '{}_0'.format(ra_runs)
bvp_label = '{}_1'.format(ra_runs)
axes[-1].plot(info[base_label]['z_profile'], info[base_label]['T_profile'][0,:],  c='blue', lw=2, label='SE')
axes[-1].plot(info[bvp_label]['z_profile'], info[bvp_label]['T_profile'][0,:], c='red', lw=2, dashes=(4,1.5), label='AE')
plt.legend(frameon=False, loc='upper right', fontsize=10)
axes[-1].set_ylabel(r'$\langle T\,\rangle_{x,y}$')
axes[-1].annotate(r'$\mathrm{(a)}$', (0.02, -0.43), fontsize=10)

y_ticks = [-0.5, -0.48, -0.46, -0.44, -0.42]
axes[-1].set_yticks(y_ticks)
x_ticks = np.array([0, 0.5, 1])
minor_x_ticks = np.array([0.25, 0.75])
axes[-1].set_xticks(x_ticks)
axes[-1].set_xticks(minor_x_ticks, minor=True)
for t in axes[-1].get_xticklabels():
    t.set_visible(False)

#Plot 2
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[1])))

axes[-1].plot(info[base_label]['z_profile'], 100*np.abs(info[base_label]['T_profile'][0,:] - info[bvp_label]['T_profile'][0,:])/np.abs(info[base_label]['T_profile'][0,:]), c='k', lw=2)
y_ticks = np.array([0, 0.2, 0.40, 0.6])
axes[-1].set_yticks(y_ticks)
x_ticks = np.array([0, 0.5, 1])
axes[-1].set_xticks(x_ticks)
minor_x_ticks = np.array([0.25, 0.75])
axes[-1].set_xticks(minor_x_ticks, minor=True)
axes[-1].set_xlabel('z')
axes[-1].set_ylabel('% diff.')
axes[-1].annotate(r'$\mathrm{(c)}$', (0.02, 0.08), fontsize=10)

##Plot 3
axes.append(plt.subplot(gs.new_subplotspec(*gs_info[2])))
axes[-1].fill_between(info[base_label]['T_xs_pdf'], 0, info[base_label]['T_pdf_pdf'], color='blue', alpha=0.4)
axes[-1].plot(info[base_label]['T_xs_pdf'], info[base_label]['T_pdf_pdf'], c='blue', lw=2)
axes[-1].fill_between(info[bvp_label]['T_xs_pdf'], 0, info[bvp_label]['T_pdf_pdf'], color='red', alpha=0.4)
axes[-1].plot(info[bvp_label]['T_xs_pdf'], info[bvp_label]['T_pdf_pdf'], c='red', lw=2)
x_ticks = np.array([-0.5, -0.45, -0.40])
axes[-1].set_xticks(x_ticks)
minor_x_ticks = np.array([-0.475, -0.425])
axes[-1].set_xticks(minor_x_ticks, minor=True)
axes[-1].set_ylabel('PDF', labelpad=2)
axes[-1].set_yscale('log')
axes[-1].set_xlim(-0.5, -0.385)#np.max(info[base_label]['T_xs_pdf']))
axes[-1].annotate(r'$\mathrm{(b)}$', (-0.497, 3e2), fontsize=10)
axes[-1].set_ylim(1e-2, 1e3)


for t in axes[-1].get_xticklabels():
    t.set_visible(False)

T_cdf_x_bvp, T_cdf_y_bvp = calculate_CDF(info[bvp_label]['T_xs_pdf'], info[bvp_label]['T_pdf_pdf'])
T_cdf_x_base, T_cdf_y_base = calculate_CDF(info[base_label]['T_xs_pdf'], info[base_label]['T_pdf_pdf'])


share = axes[-1].twinx()
share.plot(T_cdf_x_bvp, T_cdf_y_bvp, c='darkred', dashes=(5,2), lw=2)
share.plot(T_cdf_x_base, T_cdf_y_base, c='royalblue', dashes=(4,1.5), lw=2)
axes[-1].set_xlim(-0.5, -0.385)#np.max(info[base_label]['T_xs_pdf']))
share.set_ylim(0, 1.05)
share.set_ylabel('CDF', rotation=270, labelpad=10)




max_diff, comp, diff_x, diff_y = ks_test(T_cdf_x_bvp, T_cdf_y_bvp, info[bvp_label]['T_denorm_pdf'],\
                             T_cdf_x_base, T_cdf_y_base, info[base_label]['T_denorm_pdf'])



axes.append(plt.subplot(gs.new_subplotspec(*gs_info[3])))
axes[-1].plot(diff_x[diff_y > 0], diff_y[diff_y > 0],  c='k', lw=2)
axes[-1].plot(diff_x[diff_y < 0], -diff_y[diff_y < 0], c='k', ls=':', lw=2)
axes[-1].set_xlim(-0.5, -0.385)
axes[-1].set_yscale('log')
axes[-1].set_ylim(1e-3, 1e0)
x_ticks = np.array([-0.5, -0.45, -0.40])
minor_x_ticks = np.array([-0.475, -0.425])
axes[-1].set_xticks(x_ticks)
axes[-1].set_xticks(minor_x_ticks, minor=True)
#y_ticks = np.array([1e-3, 1e-1])
#axes[-1].set_yticks(y_ticks)
axes[-1].set_ylabel(r'KS$(\,T\,)$', labelpad=2)
axes[-1].set_xlabel(r'$T$')
axes[-1].annotate(r'$\mathrm{(d)}$', (-0.497, 1.5e-1), fontsize=10)

print('for {}, max diff is {}'.format('T', max_diff))


plt.savefig('temp_comparison.png'.format(k), bbox_inches='tight', dpi=200)
