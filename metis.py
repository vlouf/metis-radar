"""Perform routine quality checks on radar dual-polarisation moments.

.. author:: Valentin Louf <valentin.louf@monash.edu>
"""
import os
import sys
import glob
import time
import argparse
import warnings
import traceback

import pyart
import netCDF4
import numpy as np
import pandas as pd

import dask
import dask.bag as db


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    From http://stackoverflow.com/a/312464
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def qc_radar_stats(infile):
    try:
        radar = pyart.aux_io.read_odim_h5(infile, file_field_names=True)
    except(OSError, TypeError):
        print(f"Problem while reading file {infile}.")
        traceback.print_exc()
        return None

    radar_date = netCDF4.num2date(radar.time['data'][0], radar.time['units'])

    sl = slice(0, radar.sweep_end_ray_index['data'][1])

    try:
        refl = radar.fields[FIELDNAMES['DBZ']]['data'][sl]
        zdr = radar.fields[FIELDNAMES['ZDR']]['data'][sl]
        kdp = radar.fields[FIELDNAMES['RHOHV']]['data'][sl]
        rhohv = radar.fields[FIELDNAMES['PHIDP']]['data'][sl]
        phidp = radar.fields[FIELDNAMES['KDP']]['data'][sl]
    except KeyError:
        print(radar.fields.keys())
        raise

    pos = (rhohv > 0.8) & (refl >= 20) & (refl <= 28) & (phidp < 12)

    df = {'date': radar_date,
       'npoints': np.sum(pos),
       'zh_median': np.median(refl[pos]),
       'rhohv_median': np.median(rhohv[pos]),
       'kdp_median': np.median(kdp[pos]),
       'zdr_deviation': np.sum(zdr[pos] - zdr[pos].mean()) / np.sum(pos),
       'zdr_aad': np.sum(np.abs(zdr[pos] - zdr[pos].mean())) / np.sum(pos),
       'kdp': kdp[pos].std(),
       'phidp': phidp[pos].std(),
       'carrey': np.sqrt(3) * np.std(phidp[pos]) / (np.sum(pos) ** 1.5 * 0.250),
      }

    return df


def main():
    input_dir = os.path.join(INPATH, "*.h5*")
    flist = sorted(glob.glob(input_dir))
    if len(flist) == 0:
        print('No file found.')
        return None
    print(f'{len(flist)} files found')

    df_total = pd.DataFrame()
    for list_chunk in chunks(flist, 2 * NCPU):
        bag = db.from_sequence(list_chunk).map(qc_radar_stats)
        rslt = bag.compute

        data = dict()
        data['dates'] = [r['date'] for r in rslt]
        data['npoints'] = [r['npoints'] for r in rslt]
        data['zh_median'] = [r['zh_median'] for r in rslt]
        data['rhohv_median'] = [r['rhohv_median'] for r in rslt]
        data['kdp_median'] = [r['kdp_median'] for r in rslt]
        data['zdr_deviation'] = [r['zdr_deviation'] for r in rslt]
        data['zdr_aad'] = [r['zdr_aad'] for r in rslt]
        data['kdp'] = [r['kdp'] for r in rslt]
        data['phidp'] = [r['phidp'] for r in rslt]
        data['carrey'] = [r['carrey'] for r in rslt]

        df = pd.DataFrame(data).set_index('dates')
        df_total = df_total.append(df)

    dset = df_total.to_xarray()
    dset.to_netcdf(os.path.join(OUTPATH, "qccheck_output.nc"))

    return None


if __name__ == '__main__':
    """
    Global variables definition.
    """
    # Main global variables (Path directories).
    # Parse arguments
    parser_description = "Processing of radar data from level 1a to level 1b."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        '-i',
        '--input-dir',
        dest='indir',
        type=str,
        required=True,
        help='Input directory.')
    parser.add_argument(
        '-o',
        '--output-dir',
        dest='outdir',
        type=str,
        required=True,
        help='Output directory.')
    parser.add_argument(
        '-n',
        '--ncpu',
        dest='ncpu',
        default=16,
        type=int,
        help='Number of CPUs for multiprocessing.')
    parser.add_argument(
        '--dbz',
        dest='dbz',
        default="DBZH",
        type=str,
        help='Name of the reflectivity field.')
    parser.add_argument(
        '--zdr',
        dest='zdr',
        default="ZDR",
        type=str,
        help='Name of the differential reflectivity field.')
    parser.add_argument(
        '--rhohv',
        dest='rhohv',
        default="RHOHV",
        type=str,
        help='Name of the cross correlation ratio field.')
    parser.add_argument(
        '--phidp',
        dest='phidp',
        default="PHIDP",
        type=str,
        help='Name of the φdp field.')
    parser.add_argument(
        '--kdp',
        dest='kdp',
        default="KDP",
        type=str,
        help='Name of the Kdp field.')

    args = parser.parse_args()
    INPATH = args.indir
    OUTPATH = args.outdir
    NCPU = args.ncpu
    FIELDNAMES = dict()
    FIELDNAMES['DBZ'] = args.dbz
    FIELDNAMES['ZDR'] = args.zdr
    FIELDNAMES['RHOHV'] = args.rhohv
    FIELDNAMES['PHIDP'] = args.phidp
    FIELDNAMES['KDP'] = args.kdp

    print(f"The input directory is {INPATH}\nThe output directory is {OUTPATH}.")
    print(f"Metis will be looking for these fields in the radar file(s):")
    for k, v in FIELDNAMES.items():
        print(f"{k} with the name {v}.")

    sttime = time.time()

    main()

    print(f"Process completed in {time.time() - sttime:0.2f}.")