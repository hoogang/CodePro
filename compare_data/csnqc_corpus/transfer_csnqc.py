
import os
import re
import json


import numpy as np
import pandas as pd


# 保证以CSV保存字符串
def filter_inva_char(line):
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    # 去除\r
    line = line.replace('\r', ' ')
    # 切换换行
    line = line.replace('\n', '\\n')
    return line

def remove_code_comment(line):
    # 去除代码注释
    # ’单引号替换双引号“
    line = line.replace('\'', '\"')
    # 移除''' '''多行注释
    line = re.sub(r'\t*"\"\"[\s\S]*?\"\"\"\n+', '', line)
    # 移除#或##或##等单行注释
    line = re.sub(r'\t*#+.*?\n+', '', line)
    # 两个间隔换行符变1个
    line= re.sub('\n \n', '\n', line)
    return line

def get_first_sentence(docstr):
    docstr = re.split(r'[.\n\r]',docstr.strip('\n'))[0]
    return docstr

def join_doc_cuttokens(tokens):
    docstr= ' '.join(tokens).replace('\n','')
    return docstr


def contain_non_English(line):
    # 包含非英语
    return bool(re.search(r'[^\x00-\xff]',line))

def remove_unprint_chars(line):
    line= line.replace('\n','\\n')
    """移除所有不可见字符  保留\n"""
    line = ''.join(x for x in line if x.isprintable())
    line = line.replace('\\n', '\n')
    return line


def jsonl_tocsv(data_folder,lang_type,save_path):

    zip_file = data_folder + 'codesearchnet_%s.zip' % lang_type
    # 解压文件夹
    os.popen('unzip -d %s %s' % (data_folder, zip_file)).read()

    #  验证训练ID
    numids_list = []
    #  查询字符
    querys_list = []
    #  代码字符
    codes_list = []
    #  识别标签
    labels_list = []


    for name in ['train','valid','test']:
        difer_path = data_folder+'codesearchnet_%s/%s/'%(lang_type,name)
        # gz 文件
        jsgz_files = os.listdir(difer_path)
        sorted_jsgz = sorted(jsgz_files, key=lambda i:int(i.split('.')[0].split('_')[-1]))
        for j in sorted_jsgz:
            os.popen('gzip -d %s' % (difer_path+j)).read()
        # json文件
        json_files = os.listdir(difer_path)
        sorted_json = sorted(json_files, key=lambda i: int(i.split('.')[0].split('_')[-1]))

        # 遍历路径
        for j in sorted_json:
            with open(difer_path+j,'r',encoding='utf-8') as f:
                # 通过json转变u字符
                line_dicts  =  [json.loads(i) for i in f.readlines()]
                filter_dicts = [d for d in line_dicts if not contain_non_English(str(d))]
                for line in filter_dicts:

                    # 训练-验证-测试ID
                    numids_list.append(line['url'])

                    # 存储查询 # ’单引号替换双引号“  相对于docstring取一行（'\n')
                    query = filter_inva_char(get_first_sentence(line['docstring']))
                    querys_list.append(query)

                    # 存储代码 # ’单引号替换双引号“
                    code = filter_inva_char(remove_code_comment(line['code']))
                    codes_list.append(code)

                    labels_list.append(0)

    # 删除解压文件夹
    os.popen('rm -rf %s' % data_folder+'codesearchnet_%s'% lang_type).read()

    data_dict = {'ID': numids_list, 'orgin_query': querys_list, 'orgin_code': codes_list, 'code_rank': labels_list}
    # 数据的列表
    column_list = ['ID', 'orgin_query', 'orgin_code', 'code_rank']
    single_data = pd.DataFrame(data_dict, columns=column_list)
    print('原始所有数据的大小', single_data.shape)      # python:451636  #java:480729

    # 空字符串即""(一个或多个空格)
    # 替换成np.nan
    single_data.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)
    single_data= single_data.dropna(axis=0)
    # 重新0-索引
    single_data = single_data.reset_index(drop=True)
    print('丢弃无效数据的大小', single_data.shape)     # python:450612  #java:480706

    # 单引号置换双引号 便于解析 (主要是代码）
    for name in ['orgin_query', 'orgin_code']:
        single_data[name] = single_data[name].str.replace('\'', '\"')

    # 保存csv
    single_data.to_csv(save_path + 'origin_csnqc_%s.csv'%lang_type, encoding='utf-8', sep='|')
    print('\n%s数据类型转换完毕！！'%lang_type)


data_folder = '../../corpus_json/'
save_path = '../../code_search/rank_data/'

python_type = 'python'
java_type = 'java'

if __name__ == '__main__':
    jsonl_tocsv(data_folder,python_type,save_path)
    jsonl_tocsv(data_folder,java_type,save_path)

