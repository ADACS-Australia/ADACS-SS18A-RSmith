#!@PYTHON@
#
# Copyright (C) 2013  Leo Singer
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
from __future__ import division
"""
Create summary plots for sky maps of found injections, binned cumulatively by
false alarm rate.
"""
__author__ = "Leo Singer <leo.singer@ligo.org>"

# Command line interface.
from optparse import Option, OptionParser
from lalinference.bayestar import command

opts, args = OptionParser(
    formatter=command.NewlinePreservingHelpFormatter(),
    description=__doc__,
    option_list=[
        Option("--cumulative", action="store_true"),
        Option("--normed", action="store_true"),
        Option("--group-by", choices=("far", "snr"), metavar="far|snr",
            default="far", help="Group plots by false alarm rate (FAR) or " +
            "signal to noise ratio (SNR) [default: %default]"),
        Option("--pp-confidence-interval", type=float, metavar="PCT",
            default=95, help="If all inputs files have the same number of "
            "samples, overlay binomial confidence bands for this percentage on "
            "the P--P plot [default: %default]")
    ]
).parse_args()

# Imports.
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib import rcParams
import scipy.stats
import os
import subprocess
import numpy as np
from glue.text_progress_bar import ProgressBar
import lalinference.plot

# Create progress bar.
pb = ProgressBar()
pb.update(-1, 'reading data')

# Read in all of the datasets listed as positional command line arguments.
datasets_ = [np.recfromtxt(arg, names=True, usemask=True) for arg in args]
dataset_names = [os.path.splitext(arg)[0] for arg in args]

# For each of the quantities that we are going to plot, find their range
# over all of the datasets.
combined = np.concatenate([dataset['searched_area'] for dataset in datasets_])
min_searched_area = np.min(combined)
max_searched_area = np.max(combined)
combined = np.concatenate([dataset['offset'] for dataset in datasets_])
min_offset = np.min(combined)
max_offset = np.max(combined)
combined = np.concatenate([dataset['runtime'] for dataset in datasets_])
if np.any(np.isfinite(combined)):
    min_runtime = np.min(combined)
    max_runtime = np.max(combined)
if opts.group_by == 'far':
    combined = np.concatenate([dataset['far'] for dataset in datasets_])
    log10_min_far = int(np.ceil(np.log10(np.min(combined))))
    log10_max_far = int(np.ceil(np.log10(np.max(combined))))
    log10_far = np.arange(log10_min_far, log10_max_far + 1)
    bin_edges = 10.**log10_far
    bin_names = ['far_1e{0}'.format(e) for e in log10_far]
    bin_titles = [r'$\mathrm{{FAR}} \leq 10^{{{0}}}$ Hz'.format(e) for e in log10_far]
elif opts.group_by == 'snr':
    combined = np.concatenate([dataset['snr'] for dataset in datasets_])
    min_snr = int(np.floor(np.min(combined)))
    max_snr = int(np.floor(np.max(combined)))
    bin_edges = np.arange(min_snr, max_snr + 1)
    bin_names = ['snr_{0}'.format(e) for e in bin_edges]
    bin_titles = [r'$\mathrm{{SNR}} \geq {0}$'.format(e) for e in bin_edges]


# Set maximum range of progress bar: one tick for each of 5 figures, for each
# false alarm rate bin.
pb.max = len(bin_edges) * 4

if opts.cumulative:
    histlabel = 'cumulative '
else:
    histlabel = ''
if opts.normed:
    histlabel += 'fraction'
else:
    histlabel += 'number'
histlabel += ' of injections'

# Loop over false alarm rate bins.
for i, (bin_edge, subdir, title) in enumerate(zip(bin_edges, bin_names, bin_titles)):
    pb.update(text=subdir)

    # Retrieve records for just those events whose false alarm rate was at most
    # the upper edge of this FAR bin.
    if opts.group_by == 'far':
        datasets = [dataset[dataset['far'] <= bin_edge] for dataset in datasets_]
    elif opts.group_by == 'snr':
        datasets = [dataset[dataset['snr'] >= bin_edge] for dataset in datasets_]
    nsamples = list(set(len(dataset) for dataset in datasets))

    # Compute titles and labels for plots.
    if rcParams['text.usetex']:
        pattern = r'\verb/{0}/'
    else:
        pattern = '{0}'
    labels = tuple(pattern.format(os.path.basename(name))
        for name in dataset_names)
    if len(args) == 1:
        title += ' ({0} events)'.format(len(datasets[0]))

    # Create and change to a subdirectory for the plots for this
    # false alarm rate bin.
    subprocess.check_call(['mkdir', '-p', subdir])
    os.chdir(subdir)

    # Set up figure 1.
    fig1 = plt.figure(figsize=(6, 6))
    ax1 = fig1.add_subplot(111, projection='pp_plot')
    fig1.subplots_adjust(bottom=0.15)
    ax1.set_xlabel('searched posterior mass')
    ax1.set_ylabel('cumulative fraction of injections')
    ax1.set_title(title)

    # Set up figure 2.
    fig2 = plt.figure(figsize=(6, 4.5))
    ax2 = fig2.add_subplot(111)
    fig2.subplots_adjust(bottom=0.15)
    ax2.set_xscale('log')
    ax2.set_xlabel('searched area (deg$^2$)')
    ax2.set_ylabel(histlabel)
    ax2.set_title(title)

    # Set up figure 3.
    fig3 = plt.figure(figsize=(6, 4.5))
    ax3 = fig3.add_subplot(111)
    ax3.set_xscale('log')
    fig3.subplots_adjust(bottom=0.15)
    ax3.set_xlabel('angle between true location and mode of posterior')
    ax3.set_ylabel(histlabel)
    ax3.set_title(title)

    # Set up figure 4.
    fig4 = plt.figure(figsize=(6, 4.5))
    ax4 = fig4.add_subplot(111)
    ax4.set_xscale('log')
    fig4.subplots_adjust(bottom=0.15)
    ax4.set_xlabel('run time (s)')
    ax4.set_ylabel(histlabel)

    # Plot a histogram from each dataset onto each of the 5 figures.
    for (data, label) in zip(datasets, labels):
        if len(data): # Skip if data is empty
            ax1.add_series(data['searched_prob'], label=label)
            ax2.hist(data['searched_area'], histtype='step', label=label, bins=np.logspace(np.log10(min_searched_area), np.log10(max_searched_area), 20), cumulative=opts.cumulative, normed=opts.normed)
            ax3.hist(data['offset'], histtype='step', label=label, bins=np.logspace(np.log10(min_offset), np.log10(max_offset), 20), cumulative=opts.cumulative, normed=opts.normed)
            if np.any(np.isfinite(data['runtime'])):
                ax4.hist(data['runtime'], histtype='step', bins=np.logspace(np.log10(min_runtime), np.log10(max_runtime), 20), cumulative=opts.cumulative, normed=opts.normed)

    # Finish and save plot 1.
    pb.update(i * 4)
    # Only plot target confidence band if all datasets have the same number
    # of samples, because the confidence band depends on the number of samples.
    ax1.add_diagonal()
    if len(nsamples) == 1:
        n, = nsamples
        ax1.add_confidence_band(n, 0.01 * opts.pp_confidence_interval)
    ax1.grid()
    if len(args) > 1:
        ax1.legend(loc='lower right')
    fig1.savefig('searched_prob.pdf')

    # Finish and save plot 2.
    pb.update(i * 4 + 1)
    ax2.grid()
    fig2.savefig('searched_area_hist.pdf')

    # Finish and save plot 3.
    pb.update(i * 4 + 2)
    ax3.grid()
    fig3.savefig('offset_hist.pdf')

    # Finish and save plot 4.
    pb.update(i * 4 + 3)
    ax4.grid()
    fig4.savefig('runtime_hist.pdf')
    plt.close()

    # Go back to starting directory.
    os.chdir(os.pardir)
