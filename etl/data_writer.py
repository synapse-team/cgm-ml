import os
import numpy as np
import h5py
import logging
import csv

log = logging.getLogger(__name__)


class DataWriter:
    def __init__(self, config, run_id):
        base_dir = config.get('output', 'base_dir')
        self.run_dir = os.path.join(base_dir, run_id)
        self.initialize()

    def initialize(self):
        # create directory
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

    def write(self, qrcode, x_input, y_output, file_path):
        # qr code is the name of the file
        # xinput is ndarray
        # output is the target values
        h5filename = '%s.h5' % qrcode
        with h5py.File(os.path.join(self.run_dir, h5filename), 'w') as hf:
            hf.create_dataset("init", data=x_input)

        # target filename
        targetfilename = '%s.target' % qrcode
        with open(os.path.join(self.run_dir, targetfilename), "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(y_output)
