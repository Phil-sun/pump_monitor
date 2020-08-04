import sqlite3
from flask import Flask, request, render_template, jsonify, send_from_directory
import sys
from DBHandler import DBHandler
import os
import pandas as pd
import subprocess
import multiprocessing as mp
from model_dnn import crontab_run

app = Flask(__name__)

db = "Pstatus.db"
# jh = "CFD11-1-A29H"
# app_abs_path = os.path.abspath(os.path.dirname(__file__))

def query_db(jh):
    dbquery = DBHandler(db)
    dbquery.row_factory()
    col_name,rv = dbquery.queryAll(jh,100000)
    dbquery.dbCommit()
    dbquery.dbClose()
    # return (rv[0] if rv else None) if one else rv
    return col_name,rv

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

def save2excel(col_name,res,save_path):
    print("#"*100)
    print("save2excel savepath:",save_path)
    data_df = pd.DataFrame(res,columns=col_name)
    print(data_df)
    data_df.to_csv(save_path,index=False,encoding="gbk")

@app.route("/Pstatus/<path:jh>", methods=["GET"])
def Pstatus(jh):
    if request.method == "GET":
        # jh = "CFD11-1-A29H"
        col_name,res = query_db(jh)
        save_path = "download/" + jh + ".csv"
        save2excel(col_name,res,save_path)

        # res = res[-2000:]
        data_dict = {col_name[i]:[j[i] for j in res] for i in range(len(col_name))}
        data_dict["col_name"] = col_name
        print("col_name:",col_name)
        # print("data_dict:",data_dict)
        return jsonify(data_dict)
    #return jsonify(month=[x[0] for x in res],pred=[x[-1] for x in res])

@app.route('/map')
def map():
    return render_template('map.html')

@app.route("/download/<path:filename>")
def downloader(filename):
    # dirpath = os.path.join(app.root_path,'download')
    return send_from_directory(app.root_path,"Pstatus.db",as_attachment=True)

if __name__ == "__main__":
    print("data-api")
    # p = mp.Process(target=crontab_run)
    # p.start()
    print("开启网页")
    # app.run(host="0.0.0.0", port=8080)
    app.run(debug=True)
    # p.join()