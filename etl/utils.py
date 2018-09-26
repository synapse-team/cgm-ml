import os
import numpy as np
import logging

log = logging.getLogger(__name__)


def extract_timestamp_from_path(file_path):
    """
    Extracts a timestamp from a path.
    """
    try:
        log.debug("Extracting timestamp from file %s" % file_path)
        timestamp = file_path.split(os.sep)[-1].split("_")[2]
        log.debug("Extracted timestamp %s from file %s" % (timestamp, file_path))
        #assert len(timestamp) == 13, len(timestamp)
        #assert timestamp.isdigit()
        if not timestamp.isdigit():
            return None
        return timestamp
    except Exception as e:
        log.exception("Unable to extract timestamp")
        return None


def is_matching_measurement(path,
                            qrcode,
                            timestamp,
                            threshold=int(60 * 60 * 24 * 1000)):
    """
    Returns True if timetamps match.

    Given a timestamp it extracts a second one from the path.
    It then computes the difference between those two timestamps.
    If the differences are lower than the threshold it is a match.
    And the QR-code must match too.

    Args:
        path (string): Path to some file.
        qrcode (string): A QR-code that is supposed to be related to the file.
        timestamp (string): A timestamp that is supposed to be related to the file.
        threshold (int): A threshold for a match. In milliseconds. Default is one day.

    Returns:
        type: True if it is a match. False otherwise.
    """

    if qrcode not in path:
        return False

    if "measurements" not in path:
        return False

    # Extract the timestamp from the path. Compute difference. Decide.
    path_timestamp = extract_timestamp_from_path(path)
    if path_timestamp is None:
        return False

    difference = abs(int(timestamp) - int(path_timestamp))
    if difference > threshold:
        return False
    return True


def _rotate_point_cloud(point_cloud):

    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0], [-sinval, cosval, 0],
                                [0, 0, 1]])

    rotated_data = np.zeros(point_cloud.shape, dtype=np.float32)
    for k in range(point_cloud.shape[0]):

        shape_pc = point_cloud[k, ...]
        rotated_data[k, ...] = np.dot(
            shape_pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data
