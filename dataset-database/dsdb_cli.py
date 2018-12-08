import sys
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings("ignore")
import argparse
import dbconnector
import os
import glob2 as glob
import time
import datetime
from cgmcore import utils
import numpy as np
import progressbar
import cv2
import pprint

    
commands = ["init", "update", "filterpcds", "filterjpgs", "sortpcds", "sortjpgs"]
db_connector = dbconnector.JsonDbConnector()
args = None
default_etl_path = "../../data/etl/2018_10_31_14_19_42"


def main():
    parse_args()
    execute_command()
    
    
    
# Parsing command-line arguments.

def parse_args():
    # Parse arguments from command-line.
    global args
    parser = argparse.ArgumentParser(description="Interact with the dataset database.")
    parser.add_argument("command", metavar='command', type=str, help="command to perform")
    #parser.add_argument("--path", action="store_const", default="../../data/etl/2018_10_31_14_19_42", const=666)
    parser.add_argument('--path', help="The folder of alignments",
        action=FullPaths, type=is_dir, default=default_etl_path)
    args = parser.parse_args()
    print(args)
    

class FullPaths(argparse.Action):
    """Expand user- and relative-paths"""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

def is_dir(dirname):
    """Checks if a path is an actual directory"""
    if not os.path.isdir(dirname):
        msg = "{0} is not a directory".format(dirname)
        raise argparse.ArgumentTypeError(msg)
    else:
        return dirname
    
    
# Executing commands.

def execute_command():
    result = None
    if args.command not in commands:
        print("ERROR: Invalid command {}! Valid commands are {}.".format(args.command, commands))
        # TODO print list of commands
    elif args.command == "init":
        result = execute_command_init()
    elif args.command == "update":
        result = execute_command_update()
    elif args.command == "filterpcds":
        result = execute_command_filterpcds()
    elif args.command == "filterjpgs":
        result = execute_command_filterjpgs()
    elif args.command == "sortpcds":
        result = execute_command_sortpcds(sort_key="number_of_points", reverse=True)
    elif args.command == "sortjpgs":
        result = execute_command_sortjpgs()
    else:
        raise Exception("Unexpected {}.".format(args.command))
        
    if result != None:
        if type(result) == dict:
            #pp = pprint.PrettyPrinter(indent=4)
            pprint.pprint(result)
        else:
            print(result)
     
    
def execute_command_init():
    print("Initializing DB...")
    db_connector.initialize()
    print("Done.")

    
def execute_command_update():
    print("Updating DB...")
    
    # Process PCDs.
    # TODO error
    # TODO rejected_by_user
    glob_search_path = os.path.join(args.path, "**/*.pcd")
    pcd_paths = glob.glob(glob_search_path)
    print("Found {} PCDs.".format(len(pcd_paths)))
    insert_count = 0
    bar = progressbar.ProgressBar(max_value=len(pcd_paths))
    for index, path in enumerate(pcd_paths):
        bar.update(index)
        id = os.path.basename(path)
        result = db_connector.select(from_table="pcd_table", where_id=id)
        if result == None:
            values = { "id": id }
            values.update(get_default_values(path))
            values.update(get_pointcloud_values(path))
            db_connector.insert(into_table="pcd_table", id=id, values=values)
            insert_count += 1
        if index % 50 == 0:
            db_connector.synchronize()
    bar.finish()
    print("Inserted {} new entries.".format(insert_count))
    
    # Process JPGs.
    # TODO blurriness
    # TODO openpose
    # TODO ...
    glob_search_path = os.path.join(args.path, "**/*.jpg")
    jpg_paths = glob.glob(glob_search_path)
    print("Found {} JPGs.".format(len(jpg_paths)))
    insert_count = 0
    bar = progressbar.ProgressBar(max_value=len(pcd_paths))
    for index, path in enumerate(jpg_paths):
        bar.update(index)
        id = os.path.basename(path)
        result = db_connector.select(from_table="jpg_table", where_id=id)
        if result == None:
            values = { "id": id }
            values.update(get_default_values(path))
            values.update(get_image_values(path))
            
            db_connector.insert(into_table="jpg_table", id=id, values=values)
            insert_count += 1
        if index % 50 == 0:
            db_connector.synchronize()
    bar.finish()
    print("Inserted {} new entries.".format(insert_count))
       
    db_connector.synchronize()
    print("Done.")


def get_default_values(path):
    last_updated = time.time()
    last_updated_readable = datetime.datetime.fromtimestamp(last_updated).strftime('%Y-%m-%d %H:%M:%S')
    values = {}
    values["path"] = path
    values["last_updated"] = last_updated
    values["last_updated_readable"] = last_updated_readable
    values["rejected_by_expert"] = False
    return values
    
    
def get_pointcloud_values(path):
    number_of_points = 0.0
    confidence_min = 0.0
    confidence_avg = 0.0
    confidence_std = 0.0
    confidence_max = 0.0
    error = False
    error_message = ""
    
    try:
        pointcloud = utils.load_pcd_as_ndarray(path)
        number_of_points = len(pointcloud)
        confidence_min = float(np.min(pointcloud[:,3]))
        confidence_avg = float(np.mean(pointcloud[:,3]))
        confidence_std = float(np.std(pointcloud[:,3]))
        confidence_max = float(np.max(pointcloud[:,3]))
    except Exception as e:
        print("\n", path, e)
        error = True
        error_message = str(e)
    except ValueError as e:
        print("\n", path, e)
        error = True
        error_message = str(e)
    
    values = {}
    values["number_of_points"] = number_of_points
    values["confidence_min"] = confidence_min
    values["confidence_avg"] = confidence_avg
    values["confidence_std"] = confidence_std
    values["confidence_max"] = confidence_max
    values["error"] = error
    values["error_message"] = error_message
    return values


def get_image_values(path):
    width = 0.0
    height = 0.0
    blur_variance = 0.0
    error = False
    error_message = ""
    try:
        image = cv2.imread(path)
        width = image.size[0]
        height = image.size[0]
        blur_variance = get_blur_variance(image)
    except Exception as e:
        print("\n", path, e)
        error = True
        error_message = str(e)
    except ValueError as e:
        print("\n", path, e)
        error = True
        error_message = str(e)

    values = {}
    values["width"] = width
    values["height"] = height
    values["blur_variance"] = blur_variance
    values["error"] = error
    values["error_message"] = error_message
    return values
    
    
def get_blur_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()
 
    
def execute_command_filterpcds(
    number_of_points_threshold=10000, 
    confidence_avg_threshold=0.75, 
    remove_errors=True, 
    remove_rejects=True, 
    sort_key=None, 
    sort_reverse=False):
    
    print("Filtering DB...")
    
    entries = db_connector.select_all(from_table="pcd_table")
    result_entries = []
    for values in entries:
        
        # Remove everything that does not have enough points.
        if int(values["number_of_points"]) < number_of_points_threshold:
            continue
        # Remove everything that does not have a high enough average confidence.
        if float(values["confidence_avg"]) < confidence_avg_threshold:
            continue
        # Remove everything that has an error.
        if remove_errors == True and bool(values["error"]) == True:
            continue
        # Remove everything that has been rejected by an expert.
        if remove_rejects == True and bool(values["rejected_by_expert"]) == True:
            continue
        result_entries.append(values)
        
    if sort_key != None:
        print("Sorting", sort_key, sort_reverse)
        result_entries = list(sorted(result_entries, key=lambda x: float(x[sort_key]), reverse=sort_reverse))   
    
    return { "results" : result_entries }

        
def execute_command_filterjpgs():
    assert False, "Implement!"
        
        
def execute_command_sortpcds(sort_key, sort_reverse):
    print("Sorting DB...")
    
    entries = db_connector.select_all(from_table="pcd_table")
    sorted_entries = sorted(entries, key=lambda x: x[sort_key], reverse=sort_reverse)
    
    return { "results" : sorted_entries }
        
        
def execute_command_sortjpgs(sort_key, reverse):
    assert False, "Implement!"
        
        
if __name__ == "__main__":
    main()