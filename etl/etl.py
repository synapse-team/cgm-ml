"""
Issue-16 | Export Training Data from Firebase to Cloud Data Storage


"""
import logging
import os
import glob2
import json

log = logging.getLogger(__name__)

# 1. get all qr codes which can be used
# 2. setup processing of individual qr code


class ETL:
    def __init__(self):
        """
        read parameters from config file for pointcloud or voxelgrid
        """
        pass

    def run(self):
        # create configuration object
        # use data reader to make qrcode dictionary
        # iterate over each qr code, creating the pickle for each datapoint
        # merge all data and push to cloud bucket
        # TODO: questions
        # 1. each qrcode produces the following output: x-input, y-output and filepath
        # and they are converted to ndarray and merged together to a big pickle file
        # For ETL, we should avoid using pickle. also, loading everything in memory will not be feasible in the future
        # esp during ETL.
        # So what format of data makes sense during ETL ?
        # A text file with format: qrcode | X path | target ?
        pass