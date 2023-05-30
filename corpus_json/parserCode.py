

import pandas as pd
#多进程
from multiprocessing import Pool as ThreadPool

import sys
sys.path.append("../code_parser/")

#python解析
from python_structured import python_code_parse
from python_structured import python_query_parse


#java解析
from javang_structured import javang_code_parse
from javang_structured import javang_query_parse


##########################################python处理####################
def multi_query_propython(data_list):
    result=[' '.join(python_query_parse(line)) for line in data_list]
    return result

def multi_code_propython(data_list):
    result=[' '.join(python_code_parse(line))  for line in data_list]
    return result


def multi_query_projavang(data_list):
    result=[' '.join(javang_query_parse(line)) for line in data_list]
    return result

def multi_code_projavang(data_list):
    result=[' '.join(javang_code_parse(line))  for line in data_list]
    return result

def parse_for_python(split_num):
    # 加载单选数据
    csndf_data = pd.read_csv('csn_annotation_pred.csv', index_col=0, sep='|', encoding='utf-8')

    # 选取python数据
    csndf_data = csndf_data[csndf_data['Language'] == 'Python']

    q2id_dict = {}

    for i,j in zip(set(csndf_data['Query'].tolist()),range(len(set(csndf_data['Query'].tolist())))):
        q2id_dict[i] =j

    qid_data = [q2id_dict[i] for i in csndf_data['Query'].tolist()]

    col_name = csndf_data.columns.tolist()
    # 构建集
    csndf_data.insert(col_name.index('Language') + 1, 'Id', qid_data)

    csndf_data = csndf_data.sort_values(by=['Query','Relevance'],ascending=False)

    query_data=csndf_data['Query'].tolist()
    query_split_list=[query_data[i:i+split_num] for i in range(0,len(query_data), split_num)]
    pool= ThreadPool(10)
    query_list=pool.map(multi_query_propython, query_split_list)
    pool.close()
    pool.join()
    query=[]
    for p in query_list:
        query+=p
    print('query解析处理条数：%d'%len(query))

    code_data=csndf_data['Code'].tolist()
    code_split_list=[code_data[i:i+split_num] for i in range(0,len(code_data), split_num)]
    pool= ThreadPool(10)
    code_list=pool.map(multi_code_propython, code_split_list)
    pool.close()
    pool.join()
    code=[]
    for p in code_list:
        code+=p
    print('code解析处理条数：%d'%len(code))

    col_name = csndf_data.columns.tolist()

    # 构建集
    csndf_data.insert(col_name.index('Query') + 1, 'Parse_Query', query)
    csndf_data.insert(col_name.index('Code') + 2, 'Parse_Code', code)

    # 去除解析无效
    csndf_data = csndf_data[csndf_data['Parse_Code'] != '-1000']
    # 重新0-索引
    csndf_data = csndf_data.reset_index(drop=True)

    csndf_data.to_csv(pred_path + 'human_python_pred.csv', sep='|', encoding='utf-8')

    ######################################保存TXT数据#############################################
    # 提取query
    query_pred = csndf_data['Parse_Query'].tolist()
    # 提取code
    code_pred = csndf_data['Parse_Code'].tolist()

    # 预计长度
    pred_length = len(csndf_data)

    qid_data = csndf_data['Id'].tolist()

    # 候选ID的长度
    pred_idx = list({}.fromkeys(qid_data).keys())
    # 统计频次
    pred_list = [len([i for i in qid_data if i==j]) for j in pred_idx]

    # 查询id
    qid_pred = [['Q_%d' % i] * j for i, j in zip(range(pred_length), pred_list)]
    qid_pred = [i for j in qid_pred for i in j]
    # 代码id
    cid_pred = [['C_%d_%d' % (i, k) for k in range(j)] for i, j in zip(range(pred_length),pred_list)]
    cid_pred = [i for j in cid_pred for i in j]
    # 不相关标签0，相关标签是1
    relevance = csndf_data['Relevance'].tolist()
    # 代码标签
    label_pred = [1 if i!=0 else 0 for i in relevance]

    # 构建预测集
    pred_data = pd.DataFrame({'q_id': qid_pred, 'query': query_pred, 'c_id': cid_pred, 'code': code_pred, 'label': label_pred})
    pred_data = pred_data.reindex(columns=['q_id', 'query', 'c_id', 'code', 'label'])
    print('cosncc类型的python预测集的长度:\t%d' % len(pred_data))

    pred_data.to_csv(pred_path + 'human_python_pred.txt', index=False, header=False, sep='\t', encoding='utf-8')
    ###################################处理预测集###################################################
    print('csncc类型的python的pred数据处理完毕！')
    

def parse_for_javang(split_num):
    # 加载单选数据
    csndf_data = pd.read_csv('csn_annotation_pred.csv', index_col=0, sep='|', encoding='utf-8')

    # 选取python数据
    csndf_data = csndf_data[csndf_data['Language'] == 'Java']

    q2id_dict = {}

    for i,j in zip(set(csndf_data['Query'].tolist()),range(len(set(csndf_data['Query'].tolist())))):
        q2id_dict[i] =j

    qid_data = [q2id_dict[i] for i in csndf_data['Query'].tolist()]

    col_name = csndf_data.columns.tolist()
    # 构建集
    csndf_data.insert(col_name.index('Language') + 1, 'Id', qid_data)

    csndf_data = csndf_data.sort_values(by=['Query','Relevance'],ascending=False)

    query_data=csndf_data['Query'].tolist()
    query_split_list=[query_data[i:i+split_num] for i in range(0,len(query_data), split_num)]
    pool= ThreadPool(10)
    query_list=pool.map(multi_query_projavang, query_split_list)
    pool.close()
    pool.join()
    query=[]
    for p in query_list:
        query+=p
    print('query解析处理条数：%d'%len(query))

    code_data=csndf_data['Code'].tolist()
    code_split_list=[code_data[i:i+split_num] for i in range(0,len(code_data), split_num)]
    pool= ThreadPool(10)
    code_list=pool.map(multi_code_projavang, code_split_list)
    pool.close()
    pool.join()
    code=[]
    for p in code_list:
        code+=p
    print('code解析处理条数：%d'%len(code))

    col_name = csndf_data.columns.tolist()

    # 构建集
    csndf_data.insert(col_name.index('Query') + 1, 'Parse_Query', query)
    csndf_data.insert(col_name.index('Code') + 2, 'Parse_Code', code)

    # 去除解析无效
    csndf_data = csndf_data[csndf_data['Parse_Code'] != '-1000']
    # 重新0-索引
    csndf_data = csndf_data.reset_index(drop=True)

    csndf_data.to_csv(pred_path + 'human_java_pred.csv', sep='|', encoding='utf-8')

    ######################################保存TXT数据#############################################
    # 提取query
    query_pred = csndf_data['Parse_Query'].tolist()
    # 提取code
    code_pred = csndf_data['Parse_Code'].tolist()

    # 预计长度
    pred_length = len(csndf_data)

    qid_data = csndf_data['Id'].tolist()

    # 候选ID的长度
    pred_idx = list({}.fromkeys(qid_data).keys())
    # 统计频次
    pred_list = [len([i for i in qid_data if i==j]) for j in pred_idx]

    # 查询id
    qid_pred = [['Q_%d' % i] * j for i, j in zip(range(pred_length), pred_list)]
    qid_pred = [i for j in qid_pred for i in j]
    # 代码id
    cid_pred = [['C_%d_%d' % (i, k) for k in range(j)] for i, j in zip(range(pred_length),pred_list)]
    cid_pred = [i for j in cid_pred for i in j]
    # 不相关标签0，相关标签是1
    relevance = csndf_data['Relevance'].tolist()
    # 代码标签
    label_pred = [1 if i!=0 else 0 for i in relevance]

    # 构建预测集
    pred_data = pd.DataFrame({'q_id': qid_pred, 'query': query_pred, 'c_id': cid_pred, 'code': code_pred, 'label': label_pred})
    pred_data = pred_data.reindex(columns=['q_id', 'query', 'c_id', 'code', 'label'])
    print('cosncc类型的java预测集的长度:\t%d' % len(pred_data))

    pred_data.to_csv(pred_path + 'human_java_pred.txt', index=False, header=False, sep='\t', encoding='utf-8')
    ###################################处理预测集###################################################
    print('csncc类型的java的pred数据处理完毕！')


python_type = 'python'
java_type= 'java'

split_num=100

pred_path ='../code_search/pred_data/'

if __name__ == '__main__':
    parse_for_python(split_num)
    parse_for_javang(split_num)


