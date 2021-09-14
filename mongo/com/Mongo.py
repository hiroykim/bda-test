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

    def select_all(self, col):
        return col.find({})

    def insert_many(self, col, i_data):
        return col.insert_many(i_data)

    def update(self, col, o_data, n_data):
        return col.update(o_data, {"$set": n_data})

    def delete_many(self, col, d_data):
        return col.delete_many(d_data)

if __name__ == "__main__":
    print("--Run--")
    #print("sys.path" + str(sys.path))
    server = Mongo()
    server.connect()
    server.set_db("testdb")

    print("-insert-")
    i_data = [{"item_11":"value_11"},{"item_22":"value_22"}]
    results = server.insert_many(server.db.testdb, i_data)
    print(type(results))
    for r in results.inserted_ids:
        print(r)

    print("-select-")
    results = server.select_all(server.db.testdb)
    print(type(results))
    for r in results:
        print(r)

    print("-update-")
    o_data = {"x": 1}
    n_data = {"x": 111}
    results = server.update(server.db.testdb, o_data, n_data)
    print(type(results))
    print(results)

    print("-select-")
    results = server.select_all(server.db.testdb)
    print(type(results))
    for r in results:
        print(r)

    print("-delete-")
    d_data = {"item_22": "value_22"}
    results = server.delete_many(server.db.testdb, d_data)
    print(type(results))

    print("-select-")
    results = server.select_all(server.db.testdb)
    print(type(results))
    for r in results:
        print(r)
