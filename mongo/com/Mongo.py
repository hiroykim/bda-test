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

    def getdb(self):
        return self.conn.get_database("testdb")

    def db_select_all(self, col):
        return col.find({})

    def db_insert(self, data):
        return db.testdb.insert_many(data)

if __name__ == "__main__":
    print("--Run--")
    #print("sys.path" + str(sys.path))
    server = Mongo()
    server.connect()
    db = server.getdb()


    print("-insert-")
    i_data = [{"item_1":"value_1"},{"item_2":"value_2"}]
    results = server.db_insert(i_data)
    print(type(results))
    for r in results.inserted_ids:
        print(r)

    print("-select-")
    results = server.db_select_all(db.testdb)
    print(type(results))
    for r in results:
        print(r)
