import sys
from pymongo import MongoClient

sys.path.append("D:\\bda-test\\mongo")
# noinspection PyUnresolvedReferences
from cfg import config

class Mongo():
    def __init__(self):
        self.conf = config.MongoConfig

    def connect(self):
        self.conn = MongoClient(self.conf.host, int(self.conf.port))

    def set_db(self, dbname):
        self.db = self.conn.get_database(dbname)

    def db_select_all(self, col):
        return col.find({})

    def db_insert(self, col, data):
        return col.insert_many(data)

if __name__ == "__main__":
    print("--Run--")
    #print("sys.path" + str(sys.path))
    server = Mongo()
    server.connect()
    server.set_db("testdb")

    print("-insert-")
    i_data = [{"item_11":"value_11"},{"item_22":"value_22"}]
    results = server.db_insert(server.db.testdb, i_data)
    print(type(results))
    for r in results.inserted_ids:
        print(r)

    print("-select-")
    results = server.db_select_all(server.db.testdb)
    print(type(results))
    for r in results:
        print(r)
