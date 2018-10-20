import logging
import os
import glob2
import json
import etl.utils as etl_utils

log = logging.getLogger(__name__)


class DataReader:
    """
    provide methods to read all data
    has understanding of paths etc
    """

    def __init__(self, dataset_path, output_targets):
        self.dataset_path = dataset_path
        self.output_targets = output_targets
        self._get_paths()
        #self.qrcodes = self.find_qrcodes()
        #self.qrcodes_dictionary = self._create_qrcodes_dictionary()

    def _get_jpg_paths(self):
        glob_search_path = os.path.join(self.dataset_path, "storage/person",
                                        "**/*.jpg")
        return glob2.glob(glob_search_path)

    def _get_pcd_path(self):
        glob_search_path = os.path.join(self.dataset_path, "storage/person",
                                        "**/*.pcd")
        return glob2.glob(glob_search_path)

    def _get_json_paths(self):
        glob_search_path = os.path.join(self.dataset_path, "db/**/*.json")
        json_paths = glob2.glob(glob_search_path)
        json_paths_personal = [
            json_path for json_path in json_paths
            if "measures" not in json_path
        ]
        json_paths_measures = [
            json_path for json_path in json_paths if "measures" in json_path
        ]
        return json_paths_personal, json_paths_measures

    def _get_paths(self):
        """
        Retrieves all the relevant paths.

        That is: Paths of JPGs, PCDs, and JSONs.
        """
        log.info("Dataset path: %s" % self.dataset_path)
        self.jpg_paths = self._get_jpg_paths()
        self.pcd_paths = self._get_pcd_path()
        self.json_paths_personal, self.json_paths_measures = self._get_json_paths(
        )

    # TODO: add error handling
    def find_qrcodes(self):
        """
        Finds all QR-codes.

        Each individual is represented via a unique QR-codes. This method extracts the set of QR-codes.
        """
        # Go through all the measures and extract their QR-codes.
        qrcodes = []
        for json_path_measure in self.json_paths_measures:
            with open(json_path_measure) as json_path_measure_file:
                json_data_measure = json.load(json_path_measure_file)
                qrcode = self._extract_qrcode(json_data_measure)
                if qrcode is None:
                    continue
                qrcodes.append(qrcode)

        # Provide a sorted set.
        qrcodes = sorted(list(set(qrcodes)))
        return qrcodes

    # TODO: add error handling
    def _extract_qrcode(self, json_data_measure):
        """
        Extracts a QR-code from a JSON.
        """
        person_id = json_data_measure["personId"]["value"]
        log.debug("Processing person id %s" % str(person_id))
        json_path_personal = [
            json_path for json_path in self.json_paths_personal
            if person_id in json_path
        ]
        log.debug("Json path personal for person %s = %s" %
                  (person_id, str(json_path_personal)))
        #assert len(json_path_personal) == 1
        if len(json_path_personal) == 0:
            log.warning("Cannot create json path personal for person %s " %
                        person_id)
            return None
        if len(json_path_personal) > 1:
            log.warning("More than 1 json path personal for person %s" %
                        person_id)

        json_path_personal = json_path_personal[0]
        try:
            json_data_personal = json.load(open(json_path_personal))
        except Exception:
            log.exception("Unable to json load file %s" % json_path_personal)
            return None

        qrcode = json_data_personal["qrcode"]["value"]

        return qrcode

    def _extract_targets(self, json_data_measure):
        """
        Extracts a list of targets from JSON.
        """

        targets = []
        for output_target in self.output_targets:
            value = json_data_measure[output_target]["value"]
            targets.append(value)
        return targets

    def create_qrcodes_dictionary(self):
        """
        Creates a QR-Code-dictionary.

        This basically sorts all PCDs and JPGs.
        With respect to the targets and the QR-Codes.
        This is used heavily during data generation.
        Takes into account timestamps in order to connect data and measures.
        """

        qrcodes_dictionary = {}
        log.info("inside create_qrcodes_dictionary ")
        log.info("Total number of json paths measures %d" %
                 len(self.json_paths_measures))

        # Go thorugh all measures.
        for json_path_measure in self.json_paths_measures:
            log.info("Processing json path measure file %s" %
                     str(json_path_measure))

            # Load the data and get type.
            json_data_measure = json.load(open(json_path_measure))
            measure_type = json_data_measure["type"]["value"]
            # Ensure manual data. If it is not a manual measurement, skip.
            if measure_type != "manual":
                log.warning("Ignoring measure file %s as type != manual" %
                            str(json_path_measure))
                continue

            # Extract the QR-code.
            qrcode = self._extract_qrcode(json_data_measure)
            if qrcode is None:
                continue
            log.info("Extracted qr code %s" % str(qrcode))

            # Create an array in the dictionary if necessary.
            # Extract the targets from the JSON-data.
            targets = self._extract_targets(json_data_measure)
            log.debug("Extracted targets from file %s %s" %
                      (json_path_measure, str(targets)))

            # Extract the timestamp from the JSON-data.
            timestamp = etl_utils.extract_timestamp_from_path(
                json_path_measure)

            # Filter paths for qrcodes and measurements.
            # Find all JPGs and PCDs for a given QR-code and make sure that the timestamps are related.
            jpg_paths = [
                jpg_path for jpg_path in self.jpg_paths
                if etl_utils.is_matching_measurement(jpg_path, qrcode,
                                                     timestamp)
            ]
            log.debug("Extracted JPG paths for file %s : %d" %
                      (str(json_path_measure), len(jpg_paths)))

            pcd_paths = [
                pcd_path for pcd_path in self.pcd_paths
                if etl_utils.is_matching_measurement(pcd_path, qrcode,
                                                     timestamp)
            ]
            log.debug("Extracted pcd_path for file %s : %d" %
                      (str(json_path_measure), len(pcd_paths)))

            if len(pcd_paths) == 0:
                log.warning("Ignoring qr code %s as pcd paths are empty" %
                            str(qrcode))
                continue

            log.info("Added qr code %s into dictionary with targets %s" %
                     (qrcode, str(targets)))
            if qrcode not in qrcodes_dictionary.keys():
                qrcodes_dictionary[qrcode] = []

            qrcodes_dictionary[qrcode].append((targets, jpg_paths, pcd_paths, timestamp))

        return qrcodes_dictionary
