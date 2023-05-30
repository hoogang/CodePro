import os
import re
import json
import pickle

def remove_code_comment(line):
    # 去除代码注释
    # ’单引号替换双引号“
    line = line.replace('\'', '\"')
    # 移除''' '''多行注释
    line = re.sub(r'\t*"\"\"[\s\S]*?\"\"\"\n+', '', line)
    # 移除#或##或##等单行注释
    line = re.sub(r'\t*#+.*?\n+', '', line)
    # 两个间隔换行符变1个
    line = re.sub('\n \n', '\n', line)
    return line

def get_first_sentence(docstr):
    docstr = re.split(r'[.\n\r]',docstr.strip('\n'))[0]
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

def label_code_comment(line):
    # ’单引号替换双引号“
    line = line.replace('\'', '\"')
    # 找出''' '''多行注释
    pat1 = re.compile(r'\t*\"\"\"([\s\S]*?)\"\"\"\n+')
    com1s = ['[q]' + i + '[/q]' for i in pat1.findall(line)] if pat1.findall(line) != [] else '-100'
    # 找出#或##或##等单行注释
    pat2 = re.compile(r'\t*#+(.*?)\n+')
    com2s = ['[q]' + i + '[/q]' for i in pat2.findall(line)] if pat2.findall(line) != [] else '-100'

    if pat1.findall(line)!=[]:
        n = len(pat1.findall(line))
        for i in range(n):
            line = re.sub(r'\t*\"\"\"[\s\S]*?\"\"\"\n+', '[com1_%d]'%i, line,count=1)
    if pat2.findall(line)!=[]:
        m = len(pat2.findall(line))
        for j in range(m):
            # count=1,每次匹配一次
            line = re.sub(r'\t*#+.*?\n+', '[com2_%d]'%j, line, count=1)

    # 文本重组
    if com1s != '-100' and com2s != '-100':
        for (i,j) in zip(com1s,range(len(com1s))):
            line = line.replace('[com1_%d]'%j, i)
        for (i,j) in zip(com2s,range(len(com2s))):
            line = line.replace('[com2_%d]'%j, i)
        return line
    if com1s != '-100' and com2s == '-100':
        for (i,j) in zip(com1s,range(len(com1s))):
            line = line.replace('[com1_%d]'%j, i)
        return line
    if com1s == '-100' and com2s != '-100':
        for (i,j) in zip(com2s,range(len(com2s))):
            line = line.replace('[com2_%d]'%j, i)
        return line
    if com1s == '-100' and com2s == '-100':
        return line

# 存储query的数据
def process_jsdata(data_folder,lang_type,save_folder):

    zip_file = data_folder + 'codesearchnet_%s.zip' % lang_type
    # 解压文件夹
    os.popen('unzip -d %s %s' % (data_folder, zip_file)).read()

    corpus_lis = []

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

        count=0

        # 遍历路径
        for j in sorted_json:
            with open(difer_path+j,'r',encoding='utf-8') as f:
                # 通过json转变u字符
                line_dicts  =  [json.loads(i) for i in f.readlines()]
                filter_dicts = [d for d in line_dicts if not contain_non_English(str(d))]
                for row in filter_dicts:

                    # 创建块组合
                    block_addlis = []

                    # 训练-验证-测试ID
                    qid = '-'.join([name,str(count)])
                    block_addlis.append((qid, 0))

                    # 存储代码 ’单引号替换双引号“
                    code = remove_unprint_chars(row['code'])
                    code = remove_code_comment(code)
                    block_addlis.append([[code]])

                    # 存储查询 # ’单引号替换双引号“  相对于docstring取一行（'\n')
                    docstr = remove_unprint_chars(row['docstring'])
                    docstr = get_first_sentence(docstr)
                    block_addlis.append([docstr])

                    # 存储代码上下文
                    code =  remove_unprint_chars(row['code'])
                    ccont = label_code_comment(code)
                    block_addlis.append([ccont])

                    block_addlis.append(1)
                    # 加入数据
                    corpus_lis.append(block_addlis)

                    count+=1

    # 删除解压文件夹
    os.popen('rm -rf %s' % data_folder+'codesearchnet_%s'% lang_type).read()


    with open(save_folder + '%s_same_codesn_corpus_qid2index_blocks_labeled.pickle' % lang_type, "wb") as f:
        pickle.dump(corpus_lis, f)

    # python：451636, java: 480729
    print('same_codesn_corpus最终能对应qid索引长度-%d' % len(corpus_lis))


# 参数配置设置
data_folder = '../corpus_json/'
save_folder = './corpus/'


python_type= 'python'
java_type= 'java'



if __name__ == '__main__':
    # python
    process_jsdata(data_folder, python_type, save_folder)
    # java
    process_jsdata(data_folder, java_type, save_folder)


