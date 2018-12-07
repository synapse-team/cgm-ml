import argparse
import dbconnector
import os
import glob2 as glob
import time
import datetime
    
    
commands = ["init", "update"]
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
    if args.command not in commands:
        print("ERROR: Invalid command {}! Valid commands are {}.".format(args.command, commands))
        # TODO print list of commands
    elif args.command == "init":
        execute_command_init()
    elif args.command == "update":
        execute_command_update()
    
def execute_command_init():
    print("Initializing DB...")
    db_connector.initialize()
    print("Done.")

def execute_command_update():
    print("Updating DB...")
    
    glob_search_path = os.path.join(args.path, "**/*.pcd")
    pcd_paths = glob.glob(glob_search_path)
    print("Found {} PCDs.".format(len(pcd_paths)))
    for pcd_path in pcd_paths:
        result = db_connector.select(from_table="pcd_table", where_id="")
        if result == None:
            id = os.path.basename(pcd_path)
            last_updated = time.time()
            last_updated_readable = datetime.datetime.fromtimestamp(last_updated).strftime('%Y-%m-%d %H:%M:%S')
            values = {
                "last_updated": last_updated,
                "last_updated_readable": last_updated_readable
            }
            db_connector.insert(into_table="pcd_table", id=id, values=values)
    
    glob_search_path = os.path.join(args.path, "**/*.jpg")
    pcd_paths = glob.glob(glob_search_path)
    print("Found {} PCDs.".format(len(pcd_paths)))
    for pcd_path in pcd_paths:
        result = db_connector.select(from_table="jpg_table", where_id="")
        if result == None:
            id = os.path.basename(pcd_path)
            last_updated = time.time()
            last_updated_readable = datetime.datetime.fromtimestamp(last_updated).strftime('%Y-%m-%d %H:%M:%S')
            values = {
                "last_updated": last_updated,
                "last_updated_readable": last_updated_readable
            }
            db_connector.insert(into_table="jpg_table", id=id, values=values)
            
    db_connector.synchronize()
    print("Done.")
    

if __name__ == "__main__":
    main()