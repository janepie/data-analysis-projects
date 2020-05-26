import csv, sqlite3
import schema1
import pandas



NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"


PATH_LIST = [NODES_PATH, NODE_TAGS_PATH, WAYS_PATH, WAY_NODES_PATH, WAY_TAGS_PATH]
db = sqlite3.connect("OSM_Hamburg_Stade.db")
# c = db.cursor()

for data in PATH_LIST: 
    df = pandas.read_csv(data)
    df.to_sql(data.split('.')[0], db, if_exists='replace', index=False, schema=schema1.schema)


db.commit()
db.close()