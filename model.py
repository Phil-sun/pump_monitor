from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,label_binarize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.metrics import f1_score as f1,accuracy_score,roc_auc_score,make_scorer
from sklearn.metrics import confusion_matrix as con
import numpy as np
import pickle
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from DBHandler import DBHandler
import data_api
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
import datetime

# 忽略一些版本不兼容等警告
warnings.filterwarnings("ignore")


jh = "CFD11-1-A29H"

def train_model():
    ''' 训练模型 '''
    '''一、读取文件'''
    # csv文件路径，放在当前py文件统一路径(存放路径自己选择)
    train_file_path = 'data/6-CFD22-1-A37H.csv'
    all_data = pd.read_csv(train_file_path)
    all_data.fillna(value=0, inplace=True)
    # item_data = all_data.drop(['Date',' Time','predict'], axis=1)
    item_data = all_data.drop(['Date', ' Time', 'predict','泵入口温度','电机温度','油温'], axis=1)
    feat_labels = item_data.columns

    # 归一化
    print("归一化")
    mm= MinMaxScaler()
    x = mm.fit_transform(item_data)
    y = all_data["predict"]

    '''二、确认预测特征变量和选择要训练的特征'''
    # 要训练的特征列表
    # predictor_cols = ['油压','泵出口压力','泵入口温度','A相运行电流','泵入口压力','电机温度','运行频率','漏电电流','运行电压','油温','油嘴开度']

    # 数据拆分
    print("数据拆分")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


    '''三、创建模型和训练'''
    # 创建随机森林模型
    my_model = RandomForestClassifier(max_depth=None, min_samples_split=2, random_state=0,max_features='auto')
    # 把要训练的数据丢进去，进行模型训练
    result = my_model.fit(x_train,y_train)
    print("模型训练")
    # print('result',result)

    #构建决策树模型
    # tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
    #
    # #构建bagging分类器
    # bag = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=1.0,
    #                         max_features=1.0,bootstrap=True, bootstrap_features=False,
    #                         n_jobs=1, random_state=1)

    # 保存模型
    # 保存Model(注:save文件夹要预先建立，否则会报错)
    print("保存Model")
    with open('model/my_model.pickle', 'wb') as f:
        pickle.dump(my_model, f)


    '''四、用测试集预测'''
    predicted_result = my_model.predict(x_test)
    # predict_proba返回的是一个 n 行 k 列的数组，
    # 第 i 行 第 j 列上的数值是模型预测 第 i 个预测样本为某个标签的概率，并且每一行的概率和为1。
    # predictions = my_model.predict_proba(x_test)
    forest_score=my_model.score(x_test,y_test)

    ####打印混淆矩阵
    print('\n'+'混淆矩阵：')
    print(con(y_test,predicted_result))

    ###打印F1得分
    print('\n'+'F1 Score：')
    print(f1(y_test,predicted_result))

    ##打印测试精确率
    print('\n'+'Accuracy:')
    # print(metrics.score(y_test,predicted_result))
    print(metrics.accuracy_score(y_test,predicted_result))

    # 模型准确率
    # print('\n'+'Precision:')
    # print(metrics.precision_score(y_test,predicted_result))

    # #模型召回率
    print('\n'+'Recall:')
    print(metrics.recall_score(y_test,predicted_result))


    print('总的测试数据量：' + str(len(predicted_result)))
    # print('预测错误的数量：' + str(f))

    '''五、计算AUC的值'''
    y_test_hot = label_binarize(y_test,classes =(0, 1)) # 将测试集标签数据用二值化编码的方式转换为矩阵
    forest_y_score=my_model.predict_proba(x_test)
    forest_fpr,forest_tpr,forest_threasholds=metrics.roc_curve(y_test_hot.ravel(),forest_y_score[:,1].ravel())    # 计算ROC的值,forest_threasholds为阈值

    forest_auc=metrics.auc(forest_fpr,forest_tpr) #_auc值
    print('forest_auc',forest_auc)

    '''六、使用(RMSE)均方对数误差是做评价指标'''
    print('RMSE:',metrics.mean_squared_log_error(predicted_result, y_test))

    '''七、把预测的值按照格式保存为csv文件'''
    # my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_result})
    # you could use any filename. We choose submission here
    # my_submission.to_csv('modle/submission.csv', index=False)

    '''八、下面对训练好的随机森林，完成重要性评估'''

    # feature_importances_  可以调取关于特征重要程度
    importances = my_model.feature_importances_
    print('特征重要性：',importances)

    # x_columns = all_data.columns[2:-1]
    x_columns = feat_labels
    print('x_columns',x_columns)
    indices = np.argsort(importances)[::-1]

    for f in range(x_train.shape[1]):
    # # 对于最后需要逆序排序，我认为是做了类似决策树回溯的取值，从叶子收敛
    # # 到根，根部重要程度高于叶子。
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))



def predict_data(col, raw_data):
    # col, raw_data = data_api.run(y_string,now_string)
    # if len(raw_data) == 0:
    #     return False
    data_df = pd.DataFrame(raw_data, columns=col)
    data_df.fillna(value=0, inplace=True)
    # item_data = data_df.drop(['status', '日期', 'predict'], axis=1)
    item_data = data_df.drop(['日期', 'predict'], axis=1)
    # 归一化
    print("归一化")
    mm = MinMaxScaler()
    x = mm.fit_transform(item_data)

    # 读取Model
    with open('model/my_model.pickle', 'rb') as f:
        my_model = pickle.load(f)
    predicted_result = my_model.predict(x)

    predicted_data = data_df.drop('predict',axis=1)
    predicted_data['predict'] = predicted_result
    return predicted_data

'''保存结果到数据库'''
def saveDB(col,data,db="Pstatus.db",table=jh):
    Pdb = DBHandler(db)
    # 创建表
    # table_cmd =
    # status
    # table_cmd = col[0] + "         INT,"
    # 日期
    table_cmd = col[0] + "         text,"
    for i in range(1,len(col)):
        table_cmd += col[i] + "    REAL DEFAULT 0,"
    table_cmd = table_cmd.rstrip(',')
    print("table_cmd:",table_cmd)
    Pdb.createTable(table,table_cmd)

    # 插入数据
    Pdb.insertMany(table,data)

    # 提交
    Pdb.dbCommit()

    # 查询
    # des,res = Pdb.queryAll("Pstatus",1000)
    # print(res)

    # 关闭
    Pdb.dbClose()

def run_today():
    now_date = datetime.datetime.now().date()
    yesterday_date = now_date - datetime.timedelta(days=1)
    now_string = now_date.strftime("%Y-%m-%d %H:%M:%S")
    y_string = yesterday_date.strftime("%Y-%m-%d %H:%M:%S")

    col, raw_data = data_api.run(y_string, now_string)
    if len(raw_data) == 0:
        return
    data_df = predict_data(col, raw_data)
    # col = data_df.columns.tolist()
    # print(data_df)
    # data_df[col[0]] = data_df[col[0]]+" "+data_df[col[1]]
    # data_df.drop(columns=col[1],inplace=True)
    # print(data_df)
    col = data_df.columns.tolist()
    print(col)
    data = data_df.values.tolist()
    print("data[:10]:",data[:10])
    # jh = "CFD11-1-A16H"
    saveDB(col,data,"Pstatus.db",jh)

def run_allday():
    start_date = datetime.datetime.strptime("2020-04-28","%Y-%m-%d").date()
    end_date = datetime.datetime.strptime("2020-05-01","%Y-%m-%d").date()
    while start_date < end_date:
        next_date = start_date + datetime.timedelta(days=1)
        start_string = start_date.strftime("%Y-%m-%d %H:%M:%S")
        next_string = next_date.strftime("%Y-%m-%d %H:%M:%S")
        print("#"*100)
        print("start_string:",start_string)
        print("next_string:",next_string)
        col, raw_data = data_api.run(start_string,next_string)
        if len(raw_data) == 0:
            start_date = start_date + datetime.timedelta(days=1)
            continue
        data_df = predict_data(col, raw_data)
        col = data_df.columns.tolist()
        print(col)
        data = data_df.values.tolist()
        # jh = "CFD11-1-A16H"
        saveDB(col, data, "Pstatus.db", jh)

        start_date  = start_date + datetime.timedelta(days=1)

def job_test():
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def crontab_run():
    print("run_allday")
    run_allday()
    print("scheduler start")
    scheduler = BlockingScheduler()
    # scheduler.add_job(job_test,'cron',hour=0,minute=0,second=0)
    scheduler.add_job(run_today, 'cron', hour=6, minute=0, second=0)
    scheduler.start()

if __name__ == '__main__':
    # run_today()
    # run_allday()
    # train_model()
    crontab_run()
