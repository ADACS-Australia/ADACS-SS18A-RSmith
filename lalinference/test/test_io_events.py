from __future__ import division
from __future__ import print_function
import sys
try:
    from astropy.tests.helper import pytest
except ImportError:
    print('these tests require pytest', file=sys.stderr)
    raise SystemExit(77)
import os.path
import h5py
import numpy as np
import errno
import gzip
from lalinference.io import events


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


class MockGraceDb(object):
    """Mock GraceDB client class that reads local data files."""

    def files(self, graceid, filename):
        path = os.path.join(DATA_PATH, '{}_{}'.format(graceid, filename))
        try:
            return open(path, 'rb')
        except IOError as e:
            if e.errno != errno.ENOENT:
                raise
            return gzip.GzipFile(path + '.gz', 'rb')


def test_ligolw():
    """Test reading events from LIGO-LW XML files."""
    source = events.open(os.path.join(DATA_PATH, '2016_subset.xml.gz'))
    assert len(source) == 250
    event = source[821759]
    assert len(event.singles) == 2
    assert event.singles[0].snr_series is event.singles[1].snr_series is None
    assert event.singles[0].instrument == 'H1'
    assert event.singles[0].snr == 8.5362396
    assert event.singles[0].phase == -0.81192881
    assert (event.singles[0].time == event.singles[0].zerolag_time ==
            967328914.866107842)
    psd = event.singles[0].psd
    assert psd.f0 == 0.0
    assert psd.deltaF == 0.125
    assert event.singles[1].instrument == 'L1'
    assert event.singles[1].snr == 10.36818
    assert event.singles[1].phase == 1.9740163
    assert (event.singles[1].time == event.singles[1].zerolag_time ==
            967328914.866513726)
    psd = event.singles[0].psd
    assert psd.f0 == 0.0
    assert psd.deltaF == 0.125
    assert event.template_args == {
        'mass1': 1.56687,
        'mass2': 1.474779,
        'spin1x': 0.0,
        'spin1y': 0.0,
        'spin1z': 0.0,
        'spin2x': 0.0,
        'spin2y': 0.0,
        'spin2z': 0.0}


def test_gracedb():
    """Test reading events from GraceDB records."""
    client = MockGraceDb()
    source = events.gracedb.open(client, ['G211117', 'G197392'])
    assert len(source) == 2
    for i, (event_id, event) in enumerate(source.items()):
        if i == 0:
            assert event_id == 'G211117'
            assert (event.singles[0].snr_series is event.singles[1].snr_series
                    is None)
            assert event.singles[0].instrument == 'H1'
            assert event.singles[0].snr == 9.0802174
            assert event.singles[0].phase == -0.13969257
            assert (event.singles[0].time == event.singles[0].zerolag_time ==
                    1135136350.647757924)
            psd = event.singles[0].psd
            assert psd.f0 == 0.0
            assert psd.deltaF == 0.125
            assert event.singles[1].instrument == 'L1'
            assert event.singles[1].snr == 7.3947201
            assert event.singles[1].phase == -2.7356486
            assert (event.singles[1].time == event.singles[1].zerolag_time ==
                    1135136350.646883043)
            psd = event.singles[1].psd
            assert psd.f0 == 0.0
            assert psd.deltaF == 0.125
            assert event.template_args == {
                'mass1': 19.924686,
                'mass2': 6.4254546,
                'spin1x': 0.0,
                'spin1y': 0.0,
                'spin1z': 0.33962944,
                'spin2x': 0.0,
                'spin2y': 0.0,
                'spin2z': -0.1238557}
        elif i == 1:
            assert event_id == 'G197392'
            assert (event.singles[0].snr_series is event.singles[1].snr_series
                    is None)
            assert event.singles[0].instrument == 'H1'
            assert event.singles[0].snr == 6.9068823
            assert event.singles[0].phase == 1.8298783
            assert (event.singles[0].time == event.singles[0].zerolag_time ==
                    1128678900.444335938)
            psd = event.singles[0].psd
            assert psd.f0 == 30.0
            assert psd.deltaF == 0.125
            assert event.singles[1].instrument == 'L1'
            assert event.singles[1].snr == 6.8389997
            assert event.singles[1].phase == -1.0297496
            assert (event.singles[1].time == event.singles[1].zerolag_time ==
                    1128678900.445068359)
            psd = event.singles[1].psd
            assert psd.f0 == 30.0
            assert psd.deltaF == 0.125
            assert event.template_args == {
                'mass1': 32.064007,
                'mass2': 14.607587,
                'spin1x': 0.0,
                'spin1y': 0.0,
                'spin1z': 0.34881824,
                'spin2x': 0.0,
                'spin2y': 0.0,
                'spin2z': -0.53029484}


def test_hdf(tmpdir):
    """Test reading events from HDF5 files."""

    # Create test input files
    ifos = ['L1', 'H1']
    filenames = []
    filename = str(tmpdir / 'coincs.hdf')
    filenames.append(filename)
    with h5py.File(filename, 'w') as coinc_file:
        coinc_file.attrs['timeslide_interval'] = np.e
        coinc_group = coinc_file.create_group('foreground')
        coinc_group['template_id'] = np.arange(5)
        coinc_group['timeslide_id'] = np.arange(0, 50, 10)

        filename = str(tmpdir / 'H1L1-BANK.hdf')
        filenames.append(filename)
        with h5py.File(filename, 'w') as bank_file:
            bank_file.attrs['parameters'] = []

        for i, ifo in enumerate(ifos):
            coinc_file.attrs['detector_{}'.format(i + 1)] = ifo
            coinc_group['trigger_id{}'.format(i + 1)] = np.arange(5)

            filename = str(tmpdir / (ifo + '_triggers.hdf'))
            filenames.append(filename)
            with h5py.File(filename, 'w') as trigger_file:
                trigger_group = trigger_file.create_group(ifo)
                trigger_group['snr'] = i + np.arange(5) * np.pi
                trigger_group['coa_phase'] = i + np.arange(5) * np.pi**2
                trigger_group['end_time'] = i + np.arange(5) * np.pi**3

            filename = str(tmpdir / (ifo + '_psds.hdf'))
            filenames.append(filename)
            with h5py.File(filename, 'w') as psd_file:
                psd_file.attrs['low_frequency_cutoff'] = 10.0
                psd_file.attrs['dynamic_range_factor'] = np.e
                psd_group = psd_file.create_group(ifo)
                psd_group['start_time'] = (np.arange(5) - 0.5) * np.pi**3
                psd_group['end_time'] = (np.arange(5) + 0.5) * np.pi**3
                psd_group = psd_group.create_group('psds')
                for j in range(5):
                    psd_group[str(j)] = np.concatenate(
                        (np.zeros(5), np.arange(5)**2)) * np.e**2
                    psd_group[str(j)].attrs['delta_f'] = 2.0

    # Test reading from filenames
    source = events.open(*filenames)
    assert len(source) == 5
    for coinc_id, coinc in source.items():
        for i, (ifo, single) in enumerate(zip(ifos, coinc.singles)):
            assert single.instrument == ifo
            assert single.snr == i + coinc_id * np.pi
            assert single.phase == i + coinc_id * np.pi**2
            assert single.time == (
                single.zerolag_time + coinc_id * 10 * np.e * (i == 0))
            assert single.zerolag_time == i + coinc_id * np.pi**3
            assert single.psd.f0 == 10.0
            assert single.psd.deltaF == 2.0
            assert np.all(single.psd.data.data == np.arange(5)**2)
        assert coinc.template_args == {}

    # Test reading from h5py.File instances
    events.open(*(h5py.File(filename, 'r') for filename in filenames))

    # Test reading from file-like objects
    events.open(*(open(filename, 'rb') for filename in filenames))


if __name__ == '__main__':
    raise SystemExit(pytest.main(['-vv', __file__]))
