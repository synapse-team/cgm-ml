import os
import numpy as np
import h5py
import logging
import csv
import shutil


log = logging.getLogger(__name__)


class DataWriter:
    def __init__(self, config, run_id):
        base_dir = config.get('output', 'base_dir')
        self.base_dir = base_dir
        self.run_dir = os.path.join(base_dir, run_id)
        self.run_id = run_id
        self.initialize()

    def initialize(self):
        # create directory
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

    def write(self, qrcode, x_input, y_output, timestamp):
        # qr code is the name of the file
        # xinput is ndarray
        # output is the target values
        qrcode_dir = os.path.join(self.run_dir, qrcode)
        if not os.path.exists(qrcode_dir):
            os.makedirs(qrcode_dir)
        subdir = os.path.join(qrcode_dir, str(timestamp))
        if not os.path.exists(subdir):
            os.makedirs(subdir)

        x_filename = os.path.join(subdir, 'data.npy')
        x_input.tofile(x_filename)

        # target filename
        targetfilename = os.path.join(subdir,'target.txt')
        with open(targetfilename, "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(y_output)

    def wrapup(self):
        # write the readme file
        # zip and create simlink
        zipfilename = self.run_id
        zipfile = os.path.join(self.base_dir, zipfilename)
        shutil.make_archive(zipfile, 'zip', self.run_dir)

        # create a simlink
        os.symlink(zipfile, os.path.join(self.base_dir, 'latest.zip'))
