from __future__ import division
import matplotlib
matplotlib.rcdefaults()
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib.testing.compare import compare_images
import functools
import unittest
import os

import lalinference.plot


def image_comparison(testfunc, filename=None, tolerance=1):
    # Construct paths to baseline and result image directories.
    filedir = os.path.dirname(os.path.abspath(__file__))
    baseline_dir = os.path.join(filedir, 'baseline_images')
    result_dir = os.path.join(os.getcwd(), 'test_result_images')

    # Default test result filename: test function name, stripped of the
    # 'test_' prefix, and with '.png' appended
    if filename is None:
        filename = testfunc.__name__.replace('test_', '') + '.png'

    # Construct full paths to baseline and test images.
    baseline_path = os.path.join(baseline_dir, filename)
    result_path = os.path.join(result_dir, filename)

    @functools.wraps(testfunc)
    def test(*args, **kwargs):
        # Run test function
        fig = testfunc(*args, **kwargs)

        # Create directories if needed
        cbook.mkdirs(baseline_dir)
        cbook.mkdirs(result_dir)

        if os.path.exists(baseline_path):
            fig.savefig(result_path)
            msg = compare_images(baseline_path, result_path, tolerance)
            if msg is not None:
                raise AssertionError(msg)
        else:
            fig.savefig(baseline_path)
            raise unittest.SkipTest(
                "Generated baseline image, {0}".format(baseline_path))

    return test



# FIXME: remove try/except/else when we get Scipy >= 0.8 on SL clusters.
from scipy.stats import beta
try:
    beta.interval
except AttributeError:
    pass
else:
    class TestPPPlot(unittest.TestCase):

        def setUp(self):
            # Re-initialize the random seed to make the unit test repeatable
            np.random.seed(0)
            self.fig = plt.figure(figsize=(3, 3), dpi=72)
            self.ax = self.fig.add_subplot(111, projection='pp_plot')
            # self.ax = lalinference.plot.PPPlot(self.fig, [0.1, 0.1, 0.8, 0.8])
            # self.fig.add_axes(self.ax)
            self.p_values = np.arange(1, 20) / 20

        @image_comparison
        def test_pp_plot_steps(self):
            """Test P--P plot with drawstyle='steps'."""
            self.ax.add_confidence_band(len(self.p_values))
            self.ax.add_diagonal()
            self.ax.add_lightning(len(self.p_values), 20, drawstyle='steps')
            self.ax.add_series(self.p_values, drawstyle='steps')
            return self.fig

        @image_comparison
        def test_pp_plot_lines(self):
            """Test P--P plot with drawstyle='steps'."""
            self.ax.add_confidence_band(len(self.p_values))
            self.ax.add_diagonal()
            self.ax.add_lightning(len(self.p_values), 20, drawstyle='lines')
            self.ax.add_series(self.p_values, drawstyle='lines')
            self.ax.add_diagonal()
            return self.fig

        @image_comparison
        def test_pp_plot_default(self):
            """Test P--P plot with drawstyle='steps'."""
            self.ax.add_confidence_band(len(self.p_values))
            self.ax.add_diagonal()
            self.ax.add_lightning(len(self.p_values), 20)
            self.ax.add_series(self.p_values)
            return self.fig


class TestMollweideAxes(unittest.TestCase):

    @image_comparison
    def test_mollweide_axes(self):
        """Test HEALPix heat map on 'mollweide' axes"""
        fig = plt.figure(figsize=(6, 4), dpi=72)
        ax = fig.add_subplot(111, projection='mollweide')
        lalinference.plot.healpix_heatmap(np.arange(12), nest=True)
        ax.grid()
        return fig

    @image_comparison
    def test_astro_mollweide_axes(self):
        """Test HEALPix heat map on 'astro mollweide' axes"""
        fig = plt.figure(figsize=(6, 4), dpi=72)
        ax = fig.add_subplot(111, projection='astro mollweide')
        lalinference.plot.healpix_heatmap(np.arange(12), nest=True)
        ax.grid()
        return fig

if __name__ == '__main__':
    import unittest
    unittest.main()
