"""
Issue-16 | Export Training Data from Firebase to Cloud Data Storage


"""
import logging
import os
import glob2
import json
from .data_reader import DataReader
import configparser


log = logging.getLogger(__name__)

# 1. get all qr codes which can be used
# 2. setup processing of individual qr code


class ETL:
    def __init__(self):
        """
        read parameters from config file for pointcloud or voxelgrid
        """
        self.config = configparser.ConfigParser()
        self.data_reader = None

    def initialize(self, config_path):
        self.config.read(config_path)
        dataset_path = self.config['DataReader']['dataset_path']
        output_targets = self.config['DataReader']['output_targets'].split(',')
        self.data_reader = DataReader(dataset_path, output_targets)

    def run(self):
        log.info("ETL: RUN")
        log.info("Create qr code dictionary")
        qrcode_dict = self.data_reader.create_qrcodes_dictionary()
        log.info("Created qr code dictionary. Number of qr codes = %d" % len(qrcode_dict))
        # push each qr code to a queue
        # process each qr code, sending the output to the writer
        # writer creates the necessary files (h5)





