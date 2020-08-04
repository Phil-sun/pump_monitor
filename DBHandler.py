#!/usr/bin/env python
# coding=utf-8
__author__ = "Yi Shansong"
__version__ = "v0.1.0.0, build 06/01/2020"

import sys
import sqlite3

class DBHandler(object):
    def __init__(self,db=None):
        self.dbfile = db
        self.conn = sqlite3.connect(self.dbfile)
        self.conn.text_factory = str
        self.cur = self.conn.cursor()
    def row_factory(self):
        self.conn.row_factory = sqlite3.Row
    def dropTable(self,table):
        self.cur.execute("DROP TABLE if exists '%s';" % table)
    def createTable(self,table,item):
        self.cur.execute("CREATE TABLE if not exists '%s' (%s);" % (table,item))
    def queryAll(self,table,limit=100):
        # ecut = self.cur.execute("SELECT * from '%s' order by rowid desc limit %s;" % (table,limit))
        # ecut = self.cur.execute("SELECT * from '%s';" % (table,))
        col = self.showColumns(table)
        col_str = ','.join(col)
        # {}里列名不能用单引号，用双引号或不引，外面不加括号
        query_cmd = "select {} from (select rowid,* from '%s' order by rowid desc limit '%s') order by rowid;".format(col_str)
        print(query_cmd % (table,limit))
        ecut = self.cur.execute(query_cmd % (table,limit))
        des = [c[0] for c in ecut.description]
        fetch = ecut.fetchall()
        print("fetch",fetch[:10])
        return des,fetch
    def insertMany(self,table,data):
        # self.cur.executemany("INSERT into '%s' values (%s);" % (argv[0],argv[1]))
        l = len(data[0])
        tuple_t = "(" + "?," * l
        tuple_t = tuple_t.rstrip(',')
        tuple_t += ")"
        print("tuple_t:", tuple_t)
        ins_cmd = ("INSERT INTO '{}' VALUES " + tuple_t).format(table)
        print("ins_cmd:",ins_cmd)
        self.cur.executemany(ins_cmd,data)
    def showColumns(self,table):
        ecut = self.cur.execute("PRAGMA table_info('%s')" % table)
        fetch = ecut.fetchall()
        col = [x[1] for x in fetch]
        return col
    def dbCommit(self):
        self.conn.commit()
    def dbClose(self):
        self.conn.close()