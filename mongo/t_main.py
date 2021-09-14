# -*- coding: utf-8 -*-
from pymongo import MongoClient

def mongo_con():
    host = "localhost"
    port = "27017"
    conn = MongoClient(host, int(port))
    print(conn)
    db = conn.get_database("testdb")
    print(db)
    print(db.collection_names())

    return db


def print_select(db):
    results = db.testdb.find({})
    for r in results:
        print(r)


def do_insert(db):
    results = db.testdb.insert_many([{'x': i} for i in range(11,13)])
    for result in results.inserted_ids:
        print(result)


if __name__ == "__main__":

    db = mongo_con()
    do_insert(db)
    print_select(db)

