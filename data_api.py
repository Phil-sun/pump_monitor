#!/usr/bin/env python
# coding:utf-8

import requests
import json
from DBHandler import DBHandler
from urllib.parse import urlencode
import datetime

item_dict = {
    "日期":"rq",
    "油压":"yy",
    "泵出口压力":"bckyl",
    "泵入口温度":"brkwd",
    "A相运行电流":"axdl",     # A相电流
    "泵入口压力":"brkyl",
    "电机温度":"djwd",
    "运行频率":"yxpl",
    "漏电电流":"lddl",
    "运行电压":"yxdy",
    "X轴震动":"djzd_x",       # 电机振动X方向
    "Y轴震动":"djzd_y",       # 电机振动Y方向
    "油温":"yw",             # mei you '油温'
    "油嘴开度":"yzzj"         # 油嘴直径
}
# CFD11-1-A29H: rq:2020-04-01 -- 2020-05-01    mei you:'泵入口温度','电机温度','油温',
#              you:'油压', '泵出口压力','A相运行电流', '泵入口压力','运行电压'
# sql_item = ['status','日期','油压', '泵出口压力', '泵入口温度', 'A相运行电流', '泵入口压力', '电机温度', '运行电压', '油温','predict']
# sql_item = ['status','日期','油压', '泵出口压力', 'A相运行电流', '泵入口压力', '运行电压','predict']
sql_item = ['日期','油压', '泵出口压力', 'A相运行电流', '泵入口压力', '运行电压','predict']
use_item = sql_item[1:-1]
# jh = "CFD11-1-A16H"
jh = "CFD11-1-A29H"

def bstring2json(bstring):
    str_content = str(bstring,'utf-8')
    # print("str_content:",str_content)
    json_content = json.loads(str_content)
    # print("json_content:",json_content)
    return json_content

def data2sql(dict_content):
    dict_data = dict_content["data"]
    print("dict_data.keys():",dict_data.keys())
    sql_data = []
    # yi chang chu li
    for k,v in dict_data.items():
        if len(v) == 0:
            print("k {} is []".format(k))
            return sql_data

    list_0 = dict_data[list(dict_data.keys())[0]]
    print("list_0:",list_0[:10])
    for i in range(len(list_0)):
        one_sql = []
        # status
        # one_sql.append(0)
        one_sql.append(list_0[i][item_dict["日期"]])
        for j in use_item:
            v = dict_data[item_dict[j]][i]["value"]
            one_sql.append(v)
        # predict
        one_sql.append(0)
        sql_data.append(one_sql)
    return sql_data

def getToken():
    token_url = "http://oo-ppd-gateway-znzc.tjdevapp.cnooc/auth/oauth/token"
    token_data = {"username":"big1","password":"Abc@123#","scope":"server","grant_type":"password"}
    token_headers = {"Authorization":"Basic c2p6bDpzanps","Content-Type":"application/x-www-form-urlencoded","isToken":"false"}
    t = requests.post(token_url,data=token_data,headers=token_headers)
    token = t.json()
    print("token json:",token)
    return token["access_token"]

def getData(begtime,endtime,token):
    # url = "http://oo-ppd-gateway-znzc.tjdevapp.cnooc/interface-realtime/realData/queryRealData/jh/CFD11-1-A16H?begTime=2020-06-09%2000%3A00%3A00&endTime=2020-06-10%2000%3A00%3A00&params=axdl%2Cbxdl"
    url = "http://oo-ppd-gateway-znzc.tjdevapp.cnooc/interface-realtime/realData/queryRealData/jh/" + jh
    #Setting Headers
    # header_dict['Content-Type'] = 'application/json'
    # token = getToken()
    header_dict = {"Authorization": "Bearer "+token}
    # Setting params
    # para_dict = {"begTime":"2020-06-09 00:00:00","endTime":"2020-06-10 00:00:00","params":"axdl,bxdl"}
    shorter_item = [item_dict[x] for x in use_item]
    para_dict = {"begTime": begtime, "endTime": endtime, "params": ",".join(shorter_item)}
    print("urlencode:",urlencode(para_dict))
    # Make the requests
    r = requests.get(url,headers=header_dict,params=para_dict)
    # Check the response
    print(r.status_code)
    # print(r.content)
    return r.content

def run(y_string,now_string):
    token = getToken()
    # now_date = datetime.datetime.now().date()
    # yesterday_date = now_date - datetime.timedelta(days=1)
    # now_string = now_date.strftime("%Y-%m-%d %H:%M:%S")
    # y_string = yesterday_date.strftime("%Y-%m-%d %H:%M:%S")
    bstring_data = getData(y_string,now_string,token)
    dict_data = bstring2json(bstring_data)
    print("#"*100)
    print("dict_data.keys():",dict_data.keys())
    col = sql_item
    sql_data = data2sql(dict_data)
    print("#"*100)
    print("col:",col)
    print("sql_data[:10]:",sql_data[:10])
    print("len(sql_data):",len(sql_data))
    return col,sql_data

if __name__ == '__main__':
    now_date = datetime.datetime.now().date()
    yesterday_date = now_date - datetime.timedelta(days=1)
    now_string = now_date.strftime("%Y-%m-%d %H:%M:%S")
    y_string = yesterday_date.strftime("%Y-%m-%d %H:%M:%S")
    run(y_string,now_string)