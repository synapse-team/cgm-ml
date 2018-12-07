import json
import os

class AbstractDbConnector(object):
    
    def initialize():
        raise Exception("Implement!")
        
    def select(self, from_table, where_id):
        raise Exception("Implement!")
        
    def synchronize():
        raise Exception("Implement!")

    def insert(self, into_table, id, values):
        raise Exception("Implement!")

        
class JsonDbConnector(AbstractDbConnector):
    
    def __init__(self):
        self.database_path = "database.json"
        if os.path.exists(self.database_path):
            infile = open(self.database_path)
            self.database = json.loads(infile.read())
    
    def initialize(self):
        self.database = {
            "tables": 
            {
                "pcd_table" : {},
                "jpg_table" : {}
            }
        }
        with open(self.database_path, "w") as outfile:
            json.dump(self.database, outfile)

    def select(self, from_table, where_id):
        return self.database["tables"][from_table].get(where_id, None)
    
    def insert(self, into_table, id, values):
        self.database["tables"][into_table][id] = values
    
    def synchronize(self):
        with open(self.database_path, "w") as outfile:
            json.dump(self.database, outfile)

