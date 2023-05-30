
import os
import re
import json
import torch

import numpy as np
import pandas as pd

import sys
sys.path.append("../../nlqf_tool/")

import ruler
import modeler

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

def remove_unprint_chars(line):
    line= line.replace('\n','\\n')
    """移除所有不可见字符  保留\n"""
    line = ''.join(x for x in line if x.isprintable())
    line = line.replace('\\n', '\n')
    return line


def load_vocab_model(lang):
    with open('../../nlqf_tool/vae/data/%s/word_vocab.json'%lang,'r') as f:
        word_vocab = json.load(f)
    model_file = torch.load('../../nlqf_tool/vae/run/%s/model/model.ckpt'%lang)
    return word_vocab,model_file

def jsonl_tocsv(data_folder,lang_type,save_path):

    word_vocab, model_file = load_vocab_model(lang_type)

    zip_file = data_folder + 'codesearchnet_%s.zip' % lang_type
    # 解压文件夹
    os.popen('unzip -d %s %s' % (data_folder, zip_file)).read()

    #  验证训练ID
    numids_lis = []
    #  查询字符
    raw_docs  = []
    #  代码字符
    raw_codes = []
    #  数据标签
    labels_lis = []

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
                for row in f.readlines():
                    line = json.loads(row)

                    # 训练-验证-测试ID
                    numids_lis.append(line['url'])

                    # 存储查询  相对于docstring取一行
                    docstr = remove_unprint_chars(line['docstring'])
                    docs  =  get_first_sentence(docstr)
                    raw_docs.append(docs)   #python 457461  #java

                    # 存储代码 # ’单引号替换双引号“
                    code = remove_unprint_chars(line['code'])
                    code = remove_code_comment(code)
                    raw_codes.append(code)

                    labels_lis.append(0)

    # 删除解压文件夹
    os.popen('rm -rf %s' % data_folder+'codesearchnet_%s'% lang_type).read()

    print('filter before:',len(raw_docs))

    filter_docs, idx = ruler.filter(raw_docs)
    print('rule filter:', len(filter_docs))
    # idx是保留有用注释的索引
    filter_codes  = [raw_codes[i] for i in idx]
    filter_numids = [numids_lis[i] for i in idx]

    finall_docs, idx = modeler.filter(filter_docs, word_vocab, model_file)
    print('model filter:', len(finall_docs))

    # idx是保留有用注释的索引
    finall_codes  = [filter_codes[i] for i in idx]
    finall_numids = [filter_numids[i] for i in idx]
    finall_labels = labels_lis[:len(finall_numids)]

    # 保存到csv
    finall_docs  = [filter_inva_char(i) for i in finall_docs]
    finall_codes = [filter_inva_char(i) for i in finall_codes]

    print('filter after:',len(finall_docs))

    data_dict = {'ID': finall_numids, 'orgin_query': finall_docs, 'orgin_code': finall_codes, 'code_rank': finall_labels}
    # 数据的列表
    column_list = ['ID', 'orgin_query', 'orgin_code', 'code_rank']
    single_data = pd.DataFrame(data_dict, columns=column_list)
    print('原始所有数据的大小', single_data.shape)

    # 空字符串即""(一个或多个空格)
    # 替换成np.nan
    single_data.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)
    single_data = single_data.dropna(axis=0)
    # 重新0-索引
    single_data = single_data.reset_index(drop=True)
    print('丢弃无效数据的大小', single_data.shape)

    # 单引号置换双引号 便于解析 (主要是代码）
    for name in ['orgin_query', 'orgin_code']:
        single_data[name] = single_data[name].str.replace('\'', '\"')

    # 保存csv
    single_data.to_csv(save_path + 'filtered_csnqc_%s.csv'%lang_type, encoding='utf-8', sep='|')
    print('\n%s数据类型转换完毕！！'%lang_type)


data_folder = '../../corpus_json/'
save_path = '../../code_search/rank_data/'

python_type = 'python'
java_type ='java'

if __name__ == '__main__':
    jsonl_tocsv(data_folder,python_type,save_path)
    jsonl_tocsv(data_folder,java_type,save_path)





