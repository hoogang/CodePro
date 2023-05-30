# coding=utf-8
import os
import re

import pickle

#网页转义
import html
#解析数据
from bs4 import BeautifulSoup


#分类标签
from svm import Classfier


import sys
sys.path.append("..")

# 解析结构
from data_processing.parsers import python_structured
from data_processing.parsers import sqlang_structured

#--------------------------linux代码，配置可执行-----------------------
def extract_differ_xml(data_folder,posts_xml,python_xml,sqlang_xml,answer_xml):

    # grep 正则表达式  #提取xml中python标签
    cmd_python = 'egrep "python" %s > %s' % (data_folder+posts_xml, data_folder + python_xml)
    print(cmd_python)
    os.popen(cmd_python).read()

    # grep  正则表达式  #提取xml中sql标签
    cmd_sqlang = 'egrep "sql" %s > %s' % (data_folder + posts_xml, data_folder + sqlang_xml)
    print(cmd_sqlang)
    os.popen(cmd_sqlang).read()

    # grep 正则表达式  #所有回答xml数据
    cmd_answer = 'egrep \'PostTypeId="2"\' %s > %s' % (data_folder + posts_xml, data_folder + answer_xml)
    print(cmd_answer)
    os.popen(cmd_answer).read()

    print('XML数据抽取完毕！！')


# 解析属性标签
def get_tag_param(rowTag, tag):
    try:
        output = rowTag[tag]
        return output
    except:
        return '-10000'


# 替换标签
def filter_other_tag(htmlSoup):
    nhtmlStr = str(htmlSoup)
    precodeTags = htmlSoup.find_all('pre')
    precodeTags = [str(i) for i in precodeTags]
    nprecodeTags = []
    for code in precodeTags:
        for j in ['<h2>', '<h3>', '<p>', '<ul>', '<li>']:
            if j in code:
                code = code.replace(j, '').replace(j.replace('<', '</'), '')
        nprecodeTags.append(code)
    for i, j in zip(precodeTags, nprecodeTags):
        nhtmlStr = nhtmlStr.replace(i, j)
    return nhtmlStr


# 匹配块
def match_str_block(p, block):
    matchstr = re.finditer(p, block)
    bocklist = [i.group() for i in matchstr]
    return bocklist


# 匹配中间块
def match_str_ptag(p_nl, ptagStr):
    ptaglist = match_str_block(p_nl, ptagStr)
    ptagtext = parse_tag_text(ptaglist[-1]) if len(ptaglist) > 1 else parse_tag_text(ptaglist[0])
    return ptagtext


# 匹配最后块
def match_str_btag(p_nl, ptagStr):
    ptaglist = match_str_block(p_nl, ptagStr)
    ptagtext = parse_tag_text(ptaglist[0])
    return ptagtext


# 取p标签文本
def parse_tag_text(bodyTag):
    bodysoup = BeautifulSoup(bodyTag, 'lxml')
    bodytext = bodysoup.findAll(text=True)
    bodystr = ' '.join(bodytext)
    return bodystr


def match_tag_code(p_hcode, p_pcode, p_pcode1, p_pcode2, ctagStr):
    # h标签内容
    hcodestr = p_hcode.findall(ctagStr)[0] if p_hcode.findall(ctagStr) != [] else ''
    # 找到pre/pre class
    pcodetag = match_str_block(p_pcode, ctagStr)[0]
    # pre标签内容 或 pre class内容
    pcodestr = p_pcode1.findall(pcodetag)[0] if p_pcode1.findall(pcodetag) != [] else p_pcode2.findall(pcodetag)[0]
    # 两种标签合并
    codestr = ' '.join([hcodestr, pcodestr])
    return codestr


def parse_tag_block(htmlSoup):
    nhtmlStr = filter_other_tag(htmlSoup)

    #  识别块标签
    p_block = re.compile(r'(<p>((?!<pre>|<pre class=[\s\S]*>|<h[2|3]>)[\s\S])*</p>)?\n*'
                         r'(<h[2|3]>((?!<p>|<pre>|<pre class=[\s\S]*>|<h[2|3]>|<ul>|<li>)[\s\S])*</h[2|3]>)?\n*'
                         r'<pre((?!<p>|<pre>|<pre class=[\s\S]*>|<h[2|3]>)[\s\S])*><code>((?!<p>|<pre>|<pre class=[\s\S]*>|<h[2|3]>)[\s\S])*</code></pre>\n*'
                         r'((?!<p>|<pre>|<pre class=[\s\S]*>|<h[2|3]>)[\s\S])*\n*'
                         r'(<p>((?!<pre>|<pre class=[\s\S]*>|<h[2|3]>)[\s\S])*</p>)?')

    all_blocklis = match_str_block(p_block, nhtmlStr)

    # 上文模块
    p_forward = re.compile(r'(<p>[\s\S]*</p>)?\n*[\s\S]*(<pre>|<pre class=[\s\S]*>)<code>')
    # 中间内容
    p_context = re.compile(r'<p>((?!<p>)[\s\S])*</p>')
    # 下文模块
    p_backward = re.compile(r'</code></pre>\n*[\s\S]*(<p>[\s\S]*</p>)?')

    # 代码模块
    p_code = re.compile(r'(<h[2|3]>[\s\S]*</h[2|3]>)?\n*(<pre>|<pre class=[\s\S]*>)<code>[\s\S]*</code></pre>')
    # h2/3标签
    p_hcode = re.compile(r'h[2|3]>([\s\S]*)</h[2|3]>')
    # pre两种
    p_pcode = re.compile(r'(<pre>|<pre class=[\s\S]*>)<code>[\s\S]*</code></pre>')

    p_pcode1 = re.compile(r'<pre><code>([\s\S]*)</code></pre>')
    p_pcode2 = re.compile(r'<pre class=[\s\S]*><code>([\s\S]*)</code></pre>')

    block_list = []

    # 遍历每个block
    for i in range(0, len(all_blocklis)):

        # print('########################Block %s#############################\n' % str(i), all_blocklis[i])

        if i == 0:
            n_f = match_str_ptag(p_context, match_str_block(p_forward, all_blocklis[i])[0]) if p_context.findall(match_str_block(p_forward, all_blocklis[i])[0]) != [] else '-10000'
            # print('--------forward nl description-------\n', n_f)
        else:
            n_f = match_str_ptag(p_context, match_str_block(p_backward, all_blocklis[i - 1])[0]) if p_context.findall(match_str_block(p_backward, all_blocklis[i - 1])[0]) != [] else '-10000'
            # print('--------forward nl description-------\n', n_f)

        code = match_tag_code(p_hcode, p_pcode, p_pcode1, p_pcode2, match_str_block(p_code, all_blocklis[i])[0])
        # 对网页标签转义
        c = html.unescape(code)
        # print('-------------context code----------\n', c)

        if i == len(all_blocklis) - 1:
            n_b = match_str_btag(p_context, match_str_block(p_backward, all_blocklis[i])[0]) if p_context.findall(match_str_block(p_backward, all_blocklis[i])[0]) != [] else '-10000'
            # print('-----------backward nl description--------\n', n_b)
        else:
            n_b = match_str_ptag(p_context, match_str_block(p_backward, all_blocklis[i])[0]) if p_context.findall(match_str_block(p_backward, all_blocklis[i])[0]) != [] else '-10000'
            # print('-----------backward nl description--------\n', n_b)

        # print('########################Block %s#############################\n' % str(i))

        # 模块的分组
        block_tuple = (n_f, c, n_b)

        block_list.append(block_tuple)

    return block_list



#分类器
s = Classfier()
s.train("./balanced/pos_train.txt", "./balanced/neg_train.txt")
s.test("./balanced/pos_test.txt", "./balanced/neg_test.txt")


# [(id, index),[[si，si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度]

# 存储query的数据
def process_xmldata(data_folder,lang_type,split_num,save_folder):

    staqc_iids = pickle.load(open('../label_tool/data/code_solution_labeled_data/filter/%s_staqc_iids.pickle' % lang_type, 'rb'))
    # 取帖子的qid列表
    staqc_qids = set([i[0] for i in staqc_iids])

    hnn_iids_labels = pickle.load(open('../data_hnn/%s_hnn_iids_labeles.pickle' % lang_type, 'rb'))
    # 取帖子的qid列表
    hnn_qids = set([i[0] for i in hnn_iids_labels])

    # 经判定staqc待测语料是包含hnn评测语料的
    judge_cover = [hnn_qids.issubset(staqc_qids)]
    # 打印True 覆盖
    print(judge_cover)

    # 再次判定staqc和hnn中语料来自相同xml
    staqc_qidsfind_inxml = pickle.load(open('../label_tool/data/code_solution_labeled_data/filter/%s_staqc_qidsfind_inxml.pickle'%lang_type,'rb'))

    hnn_qidsfind_inxml = pickle.load(open('../data_hnn/%s_hnn_qidsfind_inxml.pickle' % lang_type, 'rb'))

    # 不重复一个集合是否包含另一个集合
    judge_cover = [set(hnn_qidsfind_inxml[i]).issubset(set(staqc_qidsfind_inxml[i])) for i in [2017,2018]]
    # 打印True 覆盖
    print(judge_cover)

    # 提取XML 2017,2018,2022 训练词向量

    qidsfind_2017xml= staqc_qidsfind_inxml[2017]

    # 来对索引index分割,正则匹配命令太长，必须切割！
    split_index = [qidsfind_2017xml[i:i + split_num] for i in range(0, len(qidsfind_2017xml), split_num)]

    # 执行每个部分index
    parts_file = []

    for i in range(len(split_index)):
        # linux执行命令
        eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
        # 每份xml命名
        epart_xml = '%s_codedb_map_2017.xml'%lang_type
        epart_xml = epart_xml.replace('map_2017', 'map_2017Part%d' % (i + 1))
        # 添加到列表
        parts_file.append(data_folder + epart_xml)
        # 命令字符串
        eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str,data_folder+'posts_2017.xml', data_folder+epart_xml)
        # 执行命令符
        os.popen(eping_cmd).read()  # 得到part_xml文件
        print('索引的第%d部分检索完毕！' % (i+1))

    ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_codedb_map_2017.xml'%lang_type)
    # 执行命令
    os.popen(ping_cmd).read()
    print('XML整体文件合并完毕!')

    remove_cmd = 'rm %s' % data_folder + '%s_codedb_map_2017Part*.xml'% lang_type
    os.popen(remove_cmd).read()
    # print('XML部分文件删除完毕!')


    print('#######################开始执行词向量语料%s语言的Block块解析############################' % lang_type)

    # 帖子id
    qidnum_lis = []

    # 接受ID
    accept_lis = []

    # 查询
    query_dict = {}

    # 计数
    count1 = 0
    count2 = 0

    with open(data_folder + '%s_codedb_map_2017.xml'%lang_type,'r') as f:
        for line in f:
            count1 += 1
            rowTag = BeautifulSoup(line, 'lxml').row  # row标签
            # 查询
            query = get_tag_param(rowTag, 'title')
            # 判断查询是否满足要求
            if query != '-10000':  # 存在
                # 可接受
                accept = get_tag_param(rowTag, 'acceptedanswerid')
                if accept != '-10000':  # 存在
                    count2 += 1
                    # 接受id
                    accept_lis.append(int(accept))
                    # 帖子id
                    qidnum = get_tag_param(rowTag, 'id')
                    qidnum_lis.append(int(qidnum))
                    # 字典数据 （更新/添加）
                    query_dict[(int(qidnum), int(accept))] = query

    print('2017XML查询阶段完全遍历处理-%d条数据' % count1)
    print('2017XML查询阶段有效总计处理-%d条数据' % count2)

    print('###################################2017XML查询阶段执行完毕！###################################')

    #  判断长度是否相等
    assert len(accept_lis) == len(query_dict.keys())

    print ('2017XML的文件可解析回复qid长度-%s'%len(qidnum_lis))

    ###############################################找寻相关的XML文件#########################################

    # accept从小到大的序号排序
    sort_accept_lis = sorted(accept_lis)

    # 索引的长度 #
    accept_num = len(sort_accept_lis)  #python：68480,sql：36683
    print('2017XML回复答案aid的索引长度-%d' % accept_num)

    # # 来对索引index分割,正则匹配命令太长，必须切割！
    split_index = [sort_accept_lis[i:i + split_num] for i in range(0, len(sort_accept_lis ), split_num)]
    # 执行每个部分index
    parts_file = []

    for i in range(len(split_index)):
        # linux执行命令
        eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
        # 每份xml命名
        epart_xml = '%s_codedb_ans_2017.xml' % lang_type
        epart_xml = epart_xml.replace('ans_2017', 'ans_2017Part%d' % (i + 1))
        # 添加到列表
        parts_file.append(data_folder+epart_xml)
        # 命令字符串
        eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str, data_folder + 'answer_2017.xml', data_folder + epart_xml)
        # 执行命令符
        os.popen(eping_cmd).read()  # 得到part_xml文件
        print('索引的第%d部分检索完毕！' % (i + 1))

    ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_codedb_ans_2017.xml'%lang_type)

    # 执行命令
    os.popen(ping_cmd).read()  # 得到ans_xml文件
    print('XML整体文件合并完毕!')

    remove_cmd =  'rm %s' % data_folder + '%s_codedb_ans_2017Part*.xml'% lang_type
    os.popen(remove_cmd).read()
    # print('XML部分文件删除完毕!')

    ###################################解析具有全部accept_lis索引的ans_xml文件#######################################
    # 序号id对应accept的id  qid：accept_id (aid)
    q2aids_dict = {}
    for i in query_dict.keys():
        q2aids_dict[i[0]] = i[1]

    a2qids_dict = {}
    for i in query_dict.keys():
        a2qids_dict[i[1]] = i[0]


    with open(data_folder + '%s_codedb_ans_2017.xml' % lang_type, 'r') as f:
        # 构建词典 aid:line
        line_dict= {}
        for line in f:
            sta = line.find("\"")
            end = line.find("\"", sta + 1)
            # 提取line对应的accept id
            aid = int(line[(sta + 1):end])
            line_dict[aid]=line
        # 对查询的键值和accept键值进行合并
        same_aids= (set(accept_lis) & set(line_dict.keys()))
        # python:68480  sql:36683
        print('2017XML回复答案id合并的有效长度-%d' % len(same_aids))
        # 对合并后的qid进行从小到大排序
        sort_same_qids = sorted([a2qids_dict[aid] for aid in same_aids])

        print('###################################2017XML回复阶段执行完毕！###################################')

        # 不包含社交属性语料
        codedb_lis = []

        for qid in sort_same_qids:
            # 获取查询
            query = query_dict[(qid, q2aids_dict[qid])]
            # 每行数据
            line = line_dict[q2aids_dict[qid]]
            # row标签
            rowTag = BeautifulSoup(line, 'lxml').row
            # body属性 一定存在
            body = get_tag_param(rowTag, 'body')
            htmlSoup = BeautifulSoup(body, 'lxml')
            # 抽取代码和上下文 直接用soup（不能转了str)
            block_list = parse_tag_block(htmlSoup)

            if block_list != []:
                # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度]
                # 候选的遍历
                for index in range(0, len(block_list)):
                    # 创建块组合
                    block_addlis = []
                    # 添加索引
                    block_addlis.append((qid, index))
                    # 添加上下文
                    block_tuple = block_list[index]
                    block_addlis.append([[block_tuple[0]], [block_tuple[-1]]])
                    # 添加代码
                    block_addlis.append([[block_tuple[1]]])
                    block_addlis.append([query])
                    block_addlis.append(len(block_list))
                    # 加入数据
                    codedb_lis.append(block_addlis)

        print('###################################2017XML回复阶段解析完毕！###################################')

        filter_2017qids= set([i[0][0] for i in codedb_lis])

        assert sorted(list(filter_2017qids)) == qidsfind_2017xml

        # 再到2018xml继续找
        qidsfind_2018xml = staqc_qidsfind_inxml[2018]

        # 来对索引index分割,正则匹配命令太长，必须切割！
        split_index = [qidsfind_2018xml[i:i + split_num] for i in range(0, len(qidsfind_2018xml), split_num)]

        # 执行每个部分index
        parts_file = []

        for i in range(len(split_index)):
            # linux执行命令
            eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
            # 每份xml命名
            epart_xml = '%s_codedb_map_2018.xml' % lang_type
            epart_xml = epart_xml.replace('map_2018', 'map_2018Part%d' % (i + 1))
            # 添加到列表
            parts_file.append(data_folder + epart_xml)
            # 命令字符串
            eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str, data_folder + 'posts_2018.xml', data_folder + epart_xml)
            # 执行命令符
            os.popen(eping_cmd).read()  # 得到part_xml文件
            print('索引的第%d部分检索完毕！' % (i + 1))

        ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_codedb_map_2018.xml' % lang_type)
        # 执行命令
        os.popen(ping_cmd).read()
        print('XML整体文件合并完毕!')

        remove_cmd = 'rm %s' % data_folder + '%s_codedb_map_2018Part*.xml' % lang_type
        os.popen(remove_cmd).read()
        # print('XML部分文件删除完毕!')

        # 帖子id
        new_qidnum_lis = []

        # 接受id
        new_accept_lis = []

        # 计数
        count1 = 0
        count2 = 0

        with open(data_folder + '%s_codedb_map_2018.xml' % lang_type, 'r', encoding='utf-8') as f:
            for line in f:
                count1 += 1
                rowTag = BeautifulSoup(line, 'lxml').row  # row标签
                # 查询
                query = get_tag_param(rowTag, 'title')
                # 判断查询是否满足要求
                if query != '-10000':  # 存在
                    # 可接受
                    accept = get_tag_param(rowTag, 'acceptedanswerid')
                    if accept != '-10000':  # 存在
                        count2 += 1
                        # 接受id
                        new_accept_lis.append(int(accept))
                        # 帖子id
                        qidnum = get_tag_param(rowTag, 'id')
                        new_qidnum_lis.append(int(qidnum))
                        # 字典数据 （更新/添加）
                        query_dict[(int(qidnum), int(accept))] = query


        print('2018XML查询阶段完全遍历处理-%d条数据' % count1)
        print('2018XML查询阶段有效总计处理-%d条数据' % count2)

        print('###################################2018XML查询阶段执行完毕！###################################')

        # accept从小到大的序号排序
        new_sort_accept_lis = sorted(new_accept_lis)

        print('2018XML的文件可解析回复qid长度-%s' % len(new_qidnum_lis))

        ###############################################找寻相关的XML文件#########################################

        # 索引的长度
        new_accept_num = len(new_sort_accept_lis)
        print('2018XML回复答案aid需要的解析长度-%d' % new_accept_num)

        # 来对索引index分割,正则匹配命令太长，必须切割！
        split_index = [new_sort_accept_lis[i:i + split_num] for i in range(0, len(new_sort_accept_lis), split_num)]
        # 执行每个部分index
        parts_file = []

        for i in range(len(split_index)):
            # linux执行命令
            eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
            # 每份xml命名
            epart_xml = '%s_codedb_ans_2018.xml' % lang_type
            epart_xml = epart_xml.replace('ans_2018', 'ans_2018Part%d' % (i + 1))
            # 添加到列表
            parts_file.append(data_folder + epart_xml)
            # 命令字符串
            eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str, data_folder + 'answer_2018.xml', data_folder + epart_xml)
            # 执行命令符
            os.popen(eping_cmd).read()  # 得到part_xml文件
            print('索引的第%d部分检索完毕！' % (i + 1))

        ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_codedb_ans_2018.xml' % lang_type)

        # 执行命令
        os.popen(ping_cmd).read()  # 得到ans_xml文件
        print('XML整体文件合并完毕!')

        remove_cmd = 'rm %s' % data_folder + '%s_codedb_ans_2018Part*.xml' % lang_type
        os.popen(remove_cmd).read()
        # print('XML部分文件删除完毕!')

        ###################################解析具有全部accept_lis索引的ans_xml文件#######################################
        # 序号id对应accept的id  qid：accept_id (aid)
        for i in query_dict.keys():
            q2aids_dict[i[0]] = i[1]

        for i in query_dict.keys():
            a2qids_dict[i[1]] = i[0]

        # 计数
        count1=0

        with open(data_folder + '%s_codedb_ans_2018.xml' % lang_type, 'r', encoding='utf-8') as f:
            canfind_aids = []
            for line in f:
                count1 += 1
                if count1 % 1000000 == 0:  # 总计
                    print('2018XML回复阶段已经处理--%d条数据' % count1)
                sta = line.find("\"")
                end = line.find("\"", sta + 1)
                # 提取line对应的accept id
                aid = int(line[(sta + 1):end])
                canfind_aids.append(aid)
                # 更新网页
                line_dict[aid] = line
            # 对查询的键值和accept键值进行合并
            new_same_aids = (set(new_accept_lis) & set(canfind_aids))
            print('2018XML回复答案aid合并的有效长度-%d' % len(new_same_aids))
            # 对合并后的qid进行排序
            new_sort_same_qids= sorted([a2qids_dict[aid] for aid in new_same_aids])

            print('###################################2018XML回复阶段执行完毕！###################################')

            for qid in new_sort_same_qids:
                # 获取查询
                query = query_dict[(qid, q2aids_dict[qid])]
                # 每行数据
                line = line_dict[q2aids_dict[qid]]
                # row标签
                rowTag = BeautifulSoup(line, 'lxml').row
                # body属性 一定存在
                body = get_tag_param(rowTag, 'body')
                htmlSoup = BeautifulSoup(body, 'lxml')
                # 抽取代码和上下文 直接用soup（不能转了str)
                block_list = parse_tag_block(htmlSoup)

                if block_list != []:
                    # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询 , 块长度]
                    # 候选的遍历
                    for index in range(0, len(block_list)):
                        # 创建块组合
                        block_addlis = []
                        # 添加索引
                        block_addlis.append((qid, index))
                        # 添加上下文
                        block_tuple = block_list[index]
                        block_addlis.append([[block_tuple[0]], [block_tuple[-1]]])
                        # 添加代码
                        block_addlis.append([[block_tuple[1]]])
                        block_addlis.append([query])
                        block_addlis.append(len(block_list))
                        # 加入数据
                        codedb_lis.append(block_addlis)

        find_htmls = {
            (33137311, 33487726): ["How to model several tournaments / brackets types into a SQL database?","<html><body><p>You could create tables to hold tournament types, league types, playoff types, and have a schedule table, showing an even name along with its tournament type, and then use that relationship to retrieve information about that tournament. Note, this is not MySQL, this is more generic SQL language:</p>\n\n<pre><code>CREATE TABLE tournTypes (\nID int autoincrement primary key,\nleagueId int constraint foreign key references leagueTypes.ID,\nplayoffId int constraint foreign key references playoffTypes.ID\n--...other attributes would necessitate more tables\n)\n\nCREATE TABLE leagueTypes(\nID int autoincrement primary key,\nnoOfTeams int,\nnoOfDivisions int,\ninterDivPlay bit -- e.g. a flag indicating if teams in different divisions would play\n)\n\nCREATE TABLE playoffTypes(\nID int autoincrement primary key,\nnoOfTeams int,\nisDoubleElim bit -- e.g. flag if it is double elimination\n)\n\nCREATE TABLE Schedule(\nID int autoincrement primary key,\nName text,\nstartDate datetime,\nendDate datetime,\ntournId int constraint foreign key references tournTypes.ID\n)\n</code></pre>\n\n<p>Populating the tables...</p>\n\n<pre><code>INSERT INTO tournTypes VALUES\n(1,2),\n(1,3),\n(2,3),\n(3,1)\n\nINSERT INTO leagueTypes VALUES\n(16,2,0), -- 16 teams, 2 divisions, teams only play within own division\n(8,1,0),\n(28,4,1)\n\nINSERT INTO playoffTypes VALUES\n(8,0), -- 8 teams, single elimination\n(4,0),\n(8,1)\n\nINSERT INTO Schedule VALUES\n('Champions league','2015-12-10','2016-02-10',1),\n('Rec league','2015-11-30','2016-03-04-,2)\n</code></pre>\n\n<p>Getting info on a tournament...</p>\n\n<pre><code>SELECT Name\n,startDate\n,endDate\n,l.noOfTeams as LeagueSize\n,p.noOfTeams as PlayoffTeams\n,case p.doubleElim when 0 then 'Single' when 1 then 'Double' end as Elimination\nFROM Schedule s\nINNER JOIN tournTypes t\nON s.tournId = t.ID\nINNER JOIN leagueTypes l\nON t.leagueId = l.ID\nINNER JOIN playoffTypes p\nON t.playoffId = p.ID\n</code></pre></body></html>"]}

        new_html_qids = [i[0] for i in find_htmls.keys()]

        if lang_type=='sql':
            for i in find_htmls.keys():
                q2aids_dict[i[0]] = i[1]
            # 排除这些网页
            codedb_lis = [i for i in codedb_lis if i[0][0] not in new_html_qids]
            # 先判断不为空
            for qid in new_html_qids:
                # 获取查询
                query = find_htmls[(qid, q2aids_dict[qid])][0]
                # 每行数据
                line = find_htmls[(qid, q2aids_dict[qid])][1]
                htmlSoup = BeautifulSoup(line, 'lxml')

                # 抽取代码和上下文 直接用soup（不能转了str)
                block_list = parse_tag_block(htmlSoup)

                if block_list != []:
                    # [(id, index),[[si，si+1]] 文本块，[[c]] 代码，[q] 查询,  块长度]
                    # 候选的遍历
                    for index in range(0, len(block_list)):
                        # 创建块组合
                        block_addlis = []
                        # 添加索引
                        block_addlis.append((qid, index))
                        # 添加上下文
                        block_tuple = block_list[index]
                        block_addlis.append([[block_tuple[0]], [block_tuple[-1]]])
                        # 添加代码
                        block_addlis.append([[block_tuple[1]]])
                        block_addlis.append([query])
                        block_addlis.append(len(block_list))
                        # 加入数据
                        codedb_lis.append(block_addlis)

        print('###################################2018XML回复阶段解析完毕！###################################')

        final_qids = set([i[0][0] for i in codedb_lis])

        filter_2018qids= final_qids.difference(filter_2017qids)

        assert sorted(list(filter_2018qids)) == qidsfind_2018xml

        # 再到2022xml继续找
        filter_qids =  sorted(qidsfind_2017xml+qidsfind_2018xml)

        # python：68495  sql:
        print('需要排除2017XML和2018XML滤除的qid共计-%d'%len(filter_qids))


        # 帖子id
        new0_new_qidnum_lis = []

        # 接受id
        new0_new_accept_lis = []

        # 计数
        count1 = 0
        count2 = 0

        with open(data_folder + '%s_tag_2022.xml'%lang_type, 'r', encoding='utf-8') as f:
            for line in f:
                count1 += 1
                if count1 % 100000 == 0:  # 总计
                    print('2022XML查询阶段已经处理--%d条数据' % count1)
                rowTag = BeautifulSoup(line, 'lxml').row  # row标签
                # 查询
                query = get_tag_param(rowTag, 'title')
                # 判断查询是否满足要求
                if query != '-10000':  # 存在
                    # 过滤查询query
                    queryFilter = s.filter(query)
                    if queryFilter == 0:
                        # 可接受
                        accept = get_tag_param(rowTag, 'acceptedanswerid')
                        if accept != '-10000'and lang_type in get_tag_param(rowTag, "tags") :  # 存在
                            count2 += 1
                            if count2 % 100000 == 0:  # 总计
                                print('2022XML查询阶段有效处理--%d条数据' % count2)
                            # 接受id
                            new0_new_accept_lis.append(int(accept))
                            # 帖子id
                            qidnum = get_tag_param(rowTag, 'id')
                            new0_new_qidnum_lis.append(int(qidnum))
                            # 字典数据 （更新/添加）
                            query_dict[(int(qidnum), int(accept))] = query

        print('2022XML查询阶段完全遍历处理-%d条数据' % count1)
        print('2022XML查询阶段有效总计处理-%d条数据' % count2)

        print('###################################2022XML查询阶段执行完毕！###################################')

        # accept从小到大的序号排序
        new0_new_sort_accept_lis = sorted(new0_new_accept_lis)

        print('2022XML的文件可解析回复qid长度-%s' % len(new0_new_qidnum_lis))

        ###############################################找寻相关的XML文件#########################################

        # 索引的长度
        new0_new_accept_num = len(new0_new_sort_accept_lis)
        print('2022XML回复答案aid需要的解析长度-%d' % new0_new_accept_num)

        # 来对索引index分割,正则匹配命令太长，必须切割！
        split_index = [new0_new_sort_accept_lis[i:i + split_num] for i in range(0, len(new0_new_sort_accept_lis), split_num)]
        # 执行每个部分index
        parts_file = []

        for i in range(len(split_index)):
            # linux执行命令
            eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
            # 每份xml命名
            epart_xml = '%s_codedb_oth_2022.xml' % lang_type
            epart_xml = epart_xml.replace('oth_2022', 'oth_2022Part%d' % (i + 1))
            # 添加到列表
            parts_file.append(data_folder + epart_xml)
            # 命令字符串
            eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str, data_folder + 'answer_2022.xml', data_folder + epart_xml)
            # 执行命令符
            os.popen(eping_cmd).read()  # 得到part_xml文件
            print('索引的第%d部分检索完毕！' % (i + 1))

        ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_codedb_oth_2022.xml' % lang_type)

        # 执行命令
        os.popen(ping_cmd).read()  # 得到ans_xml文件
        print('XML整体文件合并完毕!')

        remove_cmd = 'rm %s' % data_folder + '%s_codedb_oth_2022Part*.xml' % lang_type
        os.popen(remove_cmd).read()
        # print('XML部分文件删除完毕!')

        ###################################解析具有全部accept_lis索引的ans_xml文件#######################################
        # 序号id对应accept的id  qid：accept_id (aid)
        for i in query_dict.keys():
            q2aids_dict[i[0]] = i[1]

        for i in query_dict.keys():
            a2qids_dict[i[1]] = i[0]

        # 计数
        count1 = 0

        with open(data_folder + '%s_codedb_oth_2022.xml' % lang_type, 'r', encoding='utf-8') as f:
            canfind_aids = []
            for line in f:
                count1 += 1
                if count1 % 100000 == 0:  # 总计
                    print('2022XML回复阶段已经处理--%d条数据' % count1)
                sta = line.find("\"")
                end = line.find("\"", sta + 1)
                # 提取line对应的accept id
                aid = int(line[(sta + 1):end])
                canfind_aids.append(aid)
                # 更新网页
                line_dict[aid] = line

            # 对查询的键值和accept键值进行合并
            new0_new_same_aids = (set(new0_new_accept_lis) & set(canfind_aids))
            print('2022XML回复答案aid合并的有效长度-%d' % len(new0_new_same_aids))
            # 对合并后的qid进行排序
            new0_new_sort_same_qids = sorted([a2qids_dict[aid] for aid in new0_new_same_aids])
            # 进行qid过滤
            new0_filter_sort_qids = sorted([i for i in new0_new_sort_same_qids if i not in filter_qids])

            print('###################################2022XML回复阶段执行完毕！###################################')

            for qid in new0_filter_sort_qids:
                # 获取查询
                query = query_dict[(qid, q2aids_dict[qid])]
                # 每行数据
                line = line_dict[q2aids_dict[qid]]
                # row标签
                rowTag = BeautifulSoup(line, 'lxml').row
                # body属性 一定存在
                body = get_tag_param(rowTag, 'body')
                htmlSoup = BeautifulSoup(body, 'lxml')
                # 抽取代码和上下文 直接用soup（不能转了str)
                block_list = parse_tag_block(htmlSoup)

                if block_list != []:
                    # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度]
                    # 候选的遍历
                    for index in range(0, len(block_list)):
                        # 创建块组合
                        block_addlis = []
                        # 添加索引
                        block_addlis.append((qid, index))
                        # 添加上下文
                        block_tuple = block_list[index]
                        block_addlis.append([[block_tuple[0]], [block_tuple[-1]]])
                        # 添加代码
                        block_addlis.append([[block_tuple[1]]])
                        block_addlis.append([query])
                        block_addlis.append(len(block_list))
                        # 加入数据
                        codedb_lis.append(block_addlis)

        print('###################################2022XML回复阶段解析完毕！###################################')

        new_filter_qids = set([i[0][0] for i in codedb_lis])

        # 帖子id
        new1_new_qidnum_lis = []

        # 接受id
        new1_new_accept_lis = []

        # 计数
        count1 = 0
        count2 = 0

        with open(data_folder + '%s_tag_2021.xml'%lang_type, 'r', encoding='utf-8') as f:
            for line in f:
                count1 += 1
                if count1 % 100000 == 0:  # 总计
                    print('2021XML查询阶段已经处理--%d条数据' % count1)
                rowTag = BeautifulSoup(line, 'lxml').row  # row标签
                # 查询
                query = get_tag_param(rowTag, 'title')
                # 判断查询是否满足要求
                if query != '-10000':  # 存在
                    # 过滤查询query
                    queryFilter = s.filter(query)
                    if queryFilter == 0:
                        # 可接受
                        accept = get_tag_param(rowTag, 'acceptedanswerid')
                        if accept != '-10000' and lang_type in get_tag_param(rowTag, "tags"):  # 存在
                            count2 += 1
                            if count2 % 100000 == 0:  # 总计
                                print('2021XML查询阶段有效处理--%d条数据' % count2)
                            # 接受id
                            new1_new_accept_lis.append(int(accept))
                            # 帖子id
                            qidnum = get_tag_param(rowTag, 'id')
                            new1_new_qidnum_lis.append(int(qidnum))
                            # 字典数据 （更新/添加）
                            query_dict[(int(qidnum), int(accept))] = query

        print('2021XML查询阶段完全遍历处理-%d条数据' % count1)
        print('2021XML查询阶段有效总计处理-%d条数据' % count2)

        print('###################################2021XML查询阶段执行完毕！###################################')

        # accept从小到大的序号排序
        new1_new_sort_accept_lis = sorted(new1_new_accept_lis)

        print('2021XML的文件可解析回复qid长度-%s' % len(new1_new_qidnum_lis))

        ###############################################找寻相关的XML文件#########################################

        # 索引的长度
        new1_new_accept_num = len(new1_new_sort_accept_lis)
        print('2021XML回复答案aid需要的解析长度-%d' % new1_new_accept_num)

        # 来对索引index分割,正则匹配命令太长，必须切割！
        split_index = [new1_new_sort_accept_lis[i:i + split_num] for i in range(0, len(new1_new_sort_accept_lis), split_num)]
        # 执行每个部分index
        parts_file = []

        for i in range(len(split_index)):
            # linux执行命令
            eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
            # 每份xml命名
            epart_xml = '%s_codedb_oth_2021.xml' % lang_type
            epart_xml = epart_xml.replace('oth_2021', 'oth_2021Part%d' % (i + 1))
            # 添加到列表
            parts_file.append(data_folder + epart_xml)
            # 命令字符串
            eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str, data_folder + 'answer_2021.xml', data_folder + epart_xml)
            # 执行命令符
            os.popen(eping_cmd).read()  # 得到part_xml文件
            print('索引的第%d部分检索完毕！' % (i + 1))

        ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_codedb_oth_2021.xml' % lang_type)

        # 执行命令
        os.popen(ping_cmd).read()  # 得到ans_xml文件
        print('XML整体文件合并完毕!')

        remove_cmd = 'rm %s' % data_folder + '%s_codedb_oth_2021Part*.xml' % lang_type
        os.popen(remove_cmd).read()
        # print('XML部分文件删除完毕!')

        ###################################解析具有全部accept_lis索引的ans_xml文件#######################################
        # 序号id对应accept的id  qid：accept_id (aid)
        for i in query_dict.keys():
            q2aids_dict[i[0]] = i[1]

        for i in query_dict.keys():
            a2qids_dict[i[1]] = i[0]

        # 计数
        count1 = 0

        with open(data_folder + '%s_codedb_oth_2021.xml' % lang_type, 'r', encoding='utf-8') as f:
            canfind_aids = []
            for line in f:
                count1 += 1
                if count1 % 100000 == 0:  # 总计
                    print('2021XML回复阶段已经处理--%d条数据' % count1)
                sta = line.find("\"")
                end = line.find("\"", sta + 1)
                # 提取line对应的accept id
                aid = int(line[(sta + 1):end])
                canfind_aids.append(aid)
                # 更新网页
                line_dict[aid] = line

            # 对查询的键值和accept键值进行合并
            new1_new_same_aids = (set(new1_new_accept_lis) & set(canfind_aids))
            print('2021XML回复答案aid合并的有效长度-%d' % len(new1_new_same_aids))
            # 对合并后的qid进行排序
            new1_new_sort_same_qids = sorted([a2qids_dict[aid] for aid in new1_new_same_aids])
            # 进行qid过滤
            new1_filter_sort_qids = sorted([i for i in new1_new_sort_same_qids if i not in new_filter_qids])

            print('###################################2021XML回复阶段执行完毕！###################################')

            for qid in new1_filter_sort_qids:
                # 获取查询
                query = query_dict[(qid, q2aids_dict[qid])]
                # 每行数据
                line = line_dict[q2aids_dict[qid]]
                # row标签
                rowTag = BeautifulSoup(line, 'lxml').row
                # body属性 一定存在
                body = get_tag_param(rowTag, 'body')
                htmlSoup = BeautifulSoup(body, 'lxml')
                # 抽取代码和上下文 直接用soup（不能转了str)
                block_list = parse_tag_block(htmlSoup)

                if block_list != []:
                    # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度，标签]
                    # 候选的遍历
                    for index in range(0, len(block_list)):
                        # 创建块组合
                        block_addlis = []
                        # 添加索引
                        block_addlis.append((qid, index))
                        # 添加上下文
                        block_tuple = block_list[index]
                        block_addlis.append([[block_tuple[0]], [block_tuple[-1]]])
                        # 添加代码
                        block_addlis.append([[block_tuple[1]]])
                        block_addlis.append([query])
                        block_addlis.append(len(block_list))
                        # 加入数据
                        codedb_lis.append(block_addlis)

        print('###################################2021XML回复阶段解析完毕！###################################')

        new1_filter_qids = set([i[0][0] for i in codedb_lis])

        # 帖子id
        new2_new_qidnum_lis = []

        # 接受id
        new2_new_accept_lis = []

        # 计数
        count1 = 0
        count2 = 0

        with open(data_folder + '%s_tag_2020.xml'%lang_type, 'r', encoding='utf-8') as f:
            for line in f:
                count1 += 1
                if count1 % 100000 == 0:  # 总计
                    print('2020XML查询阶段已经处理--%d条数据' % count1)
                rowTag = BeautifulSoup(line, 'lxml').row  # row标签
                # 查询
                query = get_tag_param(rowTag, 'title')
                # 判断查询是否满足要求
                if query != '-10000':  # 存在
                    # 过滤查询query
                    queryFilter = s.filter(query)
                    if queryFilter == 0:
                        # 可接受
                        accept = get_tag_param(rowTag, 'acceptedanswerid')
                        if accept != '-10000' and lang_type in get_tag_param(rowTag, "tags"):  # 存在
                            count2 += 1
                            if count2 % 100000 == 0:  # 总计
                                print('2020XML查询阶段有效处理--%d条数据' % count2)
                            # 接受id
                            new2_new_accept_lis.append(int(accept))
                            # 帖子id
                            qidnum = get_tag_param(rowTag, 'id')
                            new2_new_qidnum_lis.append(int(qidnum))
                            # 字典数据 （更新/添加）
                            query_dict[(int(qidnum), int(accept))] = query

        print('2020XML查询阶段完全遍历处理-%d条数据' % count1)
        print('2020XML查询阶段有效总计处理-%d条数据' % count2)

        print('###################################2020XML查询阶段执行完毕！###################################')

        # accept从小到大的序号排序
        new2_new_sort_accept_lis = sorted(new2_new_accept_lis)

        print('2020XML的文件可解析回复qid长度-%s' % len(new2_new_qidnum_lis))

        ###############################################找寻相关的XML文件#########################################

        # 索引的长度
        new2_new_accept_num = len(new2_new_sort_accept_lis)
        print('2020XML回复答案aid需要的解析长度-%d' % new2_new_accept_num)

        # 来对索引index分割,正则匹配命令太长，必须切割！
        split_index = [new2_new_sort_accept_lis[i:i + split_num] for i in range(0, len(new2_new_sort_accept_lis), split_num)]
        # 执行每个部分index
        parts_file = []

        for i in range(len(split_index)):
            # linux执行命令
            eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
            # 每份xml命名
            epart_xml = '%s_codedb_oth_2020.xml' % lang_type
            epart_xml = epart_xml.replace('oth_2020', 'oth_2020Part%d' % (i + 1))
            # 添加到列表
            parts_file.append(data_folder + epart_xml)
            # 命令字符串
            eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str, data_folder + 'answer_2020.xml', data_folder + epart_xml)
            # 执行命令符
            os.popen(eping_cmd).read()  # 得到part_xml文件
            print('索引的第%d部分检索完毕！' % (i + 1))

        ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_codedb_oth_2020.xml' % lang_type)

        # 执行命令
        os.popen(ping_cmd).read()  # 得到ans_xml文件
        print('XML整体文件合并完毕!')

        remove_cmd = 'rm %s' % data_folder + '%s_codedb_oth_2020Part*.xml' % lang_type
        os.popen(remove_cmd).read()
        # print('XML部分文件删除完毕!')

        ###################################解析具有全部accept_lis索引的ans_xml文件#######################################
        # 序号id对应accept的id  qid：accept_id (aid)
        for i in query_dict.keys():
            q2aids_dict[i[0]] = i[1]

        for i in query_dict.keys():
            a2qids_dict[i[1]] = i[0]

        # 计数
        count1 = 0

        with open(data_folder + '%s_codedb_oth_2020.xml' % lang_type, 'r', encoding='utf-8') as f:
            canfind_aids = []
            for line in f:
                count1 += 1
                if count1 % 100000 == 0:  # 总计
                    print('2017XML回复阶段已经处理--%d条数据' % count1)
                sta = line.find("\"")
                end = line.find("\"", sta + 1)
                # 提取line对应的accept id
                aid = int(line[(sta + 1):end])
                canfind_aids.append(aid)
                # 更新网页
                line_dict[aid] = line

            # 对查询的键值和accept键值进行合并
            new2_new_same_aids = (set(new2_new_accept_lis) & set(canfind_aids))
            print('2020XML回复答案aid合并的有效长度-%d' % len(new2_new_same_aids))
            # 对合并后的qid进行排序
            new2_new_sort_same_qids = sorted([a2qids_dict[aid] for aid in new2_new_same_aids])
            # 进行qid过滤
            new2_filter_sort_qids = sorted([i for i in new2_new_sort_same_qids if i not in new1_filter_qids])

            print('###################################2020XML回复阶段执行完毕！###################################')

            for qid in new2_filter_sort_qids:
                # 获取查询
                query = query_dict[(qid, q2aids_dict[qid])]
                # 每行数据
                line = line_dict[q2aids_dict[qid]]
                # row标签
                rowTag = BeautifulSoup(line, 'lxml').row
                # body属性 一定存在
                body = get_tag_param(rowTag, 'body')
                htmlSoup = BeautifulSoup(body, 'lxml')
                # 抽取代码和上下文 直接用soup（不能转了str)
                block_list = parse_tag_block(htmlSoup)

                if block_list != []:
                    # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度，标签]
                    # 候选的遍历
                    for index in range(0, len(block_list)):
                        # 创建块组合
                        block_addlis = []
                        # 添加索引
                        block_addlis.append((qid, index))
                        # 添加上下文
                        block_tuple = block_list[index]
                        block_addlis.append([[block_tuple[0]], [block_tuple[-1]]])
                        # 添加代码
                        block_addlis.append([[block_tuple[1]]])
                        block_addlis.append([query])
                        block_addlis.append(len(block_list))
                        # 加入数据
                        codedb_lis.append(block_addlis)

        print('###################################2020XML回复阶段解析完毕！###################################')


        new2_filter_qids = set([i[0][0] for i in codedb_lis])

        # 帖子id
        new3_new_qidnum_lis = []

        # 接受id
        new3_new_accept_lis = []

        # 计数
        count1 = 0
        count2 = 0

        with open(data_folder + '%s_tag_2019.xml'%lang_type, 'r', encoding='utf-8') as f:
            for line in f:
                count1 += 1
                if count1 % 100000 == 0:  # 总计
                    print('2017XML查询阶段已经处理--%d条数据' % count1)
                rowTag = BeautifulSoup(line, 'lxml').row  # row标签
                # 查询
                query = get_tag_param(rowTag, 'title')
                # 判断查询是否满足要求
                if query != '-10000':  # 存在
                    # 过滤查询query
                    queryFilter = s.filter(query)
                    if queryFilter == 0:
                        # 可接受
                        accept = get_tag_param(rowTag, 'acceptedanswerid')
                        if accept != '-10000' and lang_type in get_tag_param(rowTag, "tags"):  # 存在
                            count2 += 1
                            if count2 % 100000 == 0:  # 总计
                                print('2017XML查询阶段有效处理--%d条数据' % count2)
                            # 接受id
                            new3_new_accept_lis.append(int(accept))
                            # 帖子id
                            qidnum = get_tag_param(rowTag, 'id')
                            new3_new_qidnum_lis.append(int(qidnum))
                            # 字典数据 （更新/添加）
                            query_dict[(int(qidnum), int(accept))] = query

        print('2019XML查询阶段完全遍历处理-%d条数据' % count1)
        print('2019XML查询阶段有效总计处理-%d条数据' % count2)

        print('###################################2019XML查询阶段执行完毕！###################################')

        # accept从小到大的序号排序
        new3_new_sort_accept_lis = sorted(new3_new_accept_lis)

        print('2019XML的文件可解析回复qid长度-%s' % len(new3_new_qidnum_lis))

        ###############################################找寻相关的XML文件#########################################

        # 索引的长度
        new3_new_accept_num = len(new3_new_sort_accept_lis)
        print('2019XML回复答案aid需要的解析长度-%d' % new3_new_accept_num)

        # 来对索引index分割,正则匹配命令太长，必须切割！
        split_index = [new3_new_sort_accept_lis[i:i + split_num] for i in range(0, len(new3_new_sort_accept_lis), split_num)]
        # 执行每个部分index
        parts_file = []

        for i in range(len(split_index)):
            # linux执行命令
            eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
            # 每份xml命名
            epart_xml = '%s_codedb_oth_2019.xml' % lang_type
            epart_xml = epart_xml.replace('oth_2019', 'oth_2019Part%d' % (i + 1))
            # 添加到列表
            parts_file.append(data_folder + epart_xml)
            # 命令字符串
            eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str, data_folder + 'answer_2019.xml', data_folder + epart_xml)
            # 执行命令符
            os.popen(eping_cmd).read()  # 得到part_xml文件
            print('索引的第%d部分检索完毕！' % (i + 1))

        ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_codedb_oth_2019.xml' % lang_type)

        # 执行命令
        os.popen(ping_cmd).read()  # 得到ans_xml文件
        print('XML整体文件合并完毕!')

        remove_cmd = 'rm %s' % data_folder + '%s_codedb_oth_2019Part*.xml' % lang_type
        os.popen(remove_cmd).read()
        # print('XML部分文件删除完毕!')

        ###################################解析具有全部accept_lis索引的ans_xml文件#######################################
        # 序号id对应accept的id  qid：accept_id (aid)
        for i in query_dict.keys():
            q2aids_dict[i[0]] = i[1]

        for i in query_dict.keys():
            a2qids_dict[i[1]] = i[0]

        # 计数
        count1 = 0

        with open(data_folder + '%s_codedb_oth_2019.xml' % lang_type, 'r', encoding='utf-8') as f:
            canfind_aids = []
            for line in f:
                count1 += 1
                if count1 % 100000 == 0:  # 总计
                    print('2019XML回复阶段已经处理--%d条数据' % count1)
                sta = line.find("\"")
                end = line.find("\"", sta + 1)
                # 提取line对应的accept id
                aid = int(line[(sta + 1):end])
                canfind_aids.append(aid)
                # 更新网页
                line_dict[aid] = line

            # 对查询的键值和accept键值进行合并
            new3_new_same_aids = (set(new3_new_accept_lis) & set(canfind_aids))
            print('2019XML回复答案aid合并的有效长度-%d' % len(new3_new_same_aids))
            # 对合并后的qid进行排序
            new3_new_sort_same_qids = sorted([a2qids_dict[aid] for aid in new3_new_same_aids])
            # 进行qid过滤
            new3_filter_sort_qids = sorted([i for i in new3_new_sort_same_qids if i not in new2_filter_qids])

            print('###################################2019XML回复阶段执行完毕！###################################')

            for qid in new3_filter_sort_qids:
                # 获取查询
                query = query_dict[(qid, q2aids_dict[qid])]
                # 每行数据
                line = line_dict[q2aids_dict[qid]]
                # row标签
                rowTag = BeautifulSoup(line, 'lxml').row
                # body属性 一定存在
                body = get_tag_param(rowTag, 'body')
                htmlSoup = BeautifulSoup(body, 'lxml')
                # 抽取代码和上下文 直接用soup（不能转了str)
                block_list = parse_tag_block(htmlSoup)

                if block_list != []:
                    # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度]
                    # 候选的遍历
                    for index in range(0, len(block_list)):
                        # 创建块组合
                        block_addlis = []
                        # 添加索引
                        block_addlis.append((qid, index))
                        # 添加上下文
                        block_tuple = block_list[index]
                        block_addlis.append([[block_tuple[0]], [block_tuple[-1]]])
                        # 添加代码
                        block_addlis.append([[block_tuple[1]]])
                        block_addlis.append([query])
                        block_addlis.append(len(block_list))
                        # 加入数据
                        codedb_lis.append(block_addlis)

        print('###################################2019XML回复阶段解析完毕！###################################')

        new3_filter_qids = set([i[0][0] for i in codedb_lis])

        # 帖子id
        new4_new_qidnum_lis = []

        # 接受id
        new4_new_accept_lis = []

        # 计数
        count1 = 0
        count2 = 0

        with open(data_folder + '%s_tag_2018.xml'%lang_type, 'r', encoding='utf-8') as f:
            for line in f:
                count1 += 1
                if count1 % 100000 == 0:  # 总计
                    print('2018XML查询阶段已经处理--%d条数据' % count1)
                rowTag = BeautifulSoup(line, 'lxml').row  # row标签
                # 查询
                query = get_tag_param(rowTag, 'title')
                # 判断查询是否满足要求
                if query != '-10000':  # 存在
                    # 过滤查询query
                    queryFilter = s.filter(query)
                    if queryFilter == 0:
                        # 可接受
                        accept = get_tag_param(rowTag, 'acceptedanswerid')
                        if accept != '-10000' and lang_type in get_tag_param(rowTag, "tags"):  # 存在
                            count2 += 1
                            if count2 % 100000 == 0:  # 总计
                                print('2018XML查询阶段有效处理--%d条数据' % count2)
                            # 接受id
                            new4_new_accept_lis.append(int(accept))
                            # 帖子id
                            qidnum = get_tag_param(rowTag, 'id')
                            new4_new_qidnum_lis.append(int(qidnum))
                            # 字典数据 （更新/添加）
                            query_dict[(int(qidnum), int(accept))] = query

        print('2018XML查询阶段完全遍历处理-%d条数据' % count1)
        print('2018XML查询阶段有效总计处理-%d条数据' % count2)

        print('###################################2018XML查询阶段执行完毕！###################################')

        # accept从小到大的序号排序
        new4_new_sort_accept_lis = sorted(new4_new_accept_lis)

        print('2018XML的文件可解析回复qid长度-%s' % len(new4_new_qidnum_lis))

        ###############################################找寻相关的XML文件#########################################

        # 索引的长度
        new4_new_accept_num = len(new4_new_sort_accept_lis)
        print('2018XML回复答案aid需要的解析长度-%d' % new4_new_accept_num)

        # 来对索引index分割,正则匹配命令太长，必须切割！
        split_index = [new4_new_sort_accept_lis[i:i + split_num] for i in range(0, len(new4_new_sort_accept_lis), split_num)]
        # 执行每个部分index
        parts_file = []

        for i in range(len(split_index)):
            # linux执行命令
            eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
            # 每份xml命名
            epart_xml = '%s_codedb_oth_2018.xml' % lang_type
            epart_xml = epart_xml.replace('oth_2018', 'oth_2018Part%d' % (i + 1))
            # 添加到列表
            parts_file.append(data_folder + epart_xml)
            # 命令字符串
            eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str, data_folder + 'answer_2018.xml', data_folder + epart_xml)
            # 执行命令符
            os.popen(eping_cmd).read()  # 得到part_xml文件
            print('索引的第%d部分检索完毕！' % (i + 1))

        ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_codedb_oth_2018.xml' % lang_type)

        # 执行命令
        os.popen(ping_cmd).read()  # 得到ans_xml文件
        print('XML整体文件合并完毕!')

        remove_cmd = 'rm %s' % data_folder + '%s_codedb_oth_2018Part*.xml' % lang_type
        os.popen(remove_cmd).read()
        # print('XML部分文件删除完毕!')

        ###################################解析具有全部accept_lis索引的ans_xml文件#######################################
        # 序号id对应accept的id  qid：accept_id (aid)
        for i in query_dict.keys():
            q2aids_dict[i[0]] = i[1]

        for i in query_dict.keys():
            a2qids_dict[i[1]] = i[0]

        # 计数
        count1 = 0

        with open(data_folder + '%s_codedb_oth_2018.xml' % lang_type, 'r', encoding='utf-8') as f:
            canfind_aids = []
            for line in f:
                count1 += 1
                if count1 % 100000 == 0:  # 总计
                    print('2018XML回复阶段已经处理--%d条数据' % count1)
                sta = line.find("\"")
                end = line.find("\"", sta + 1)
                # 提取line对应的accept id
                aid = int(line[(sta + 1):end])
                canfind_aids.append(aid)
                # 更新网页
                line_dict[aid] = line

            # 对查询的键值和accept键值进行合并
            new4_new_same_aids = (set(new4_new_accept_lis) & set(canfind_aids))
            print('2018XML回复答案aid合并的有效长度-%d' % len(new4_new_same_aids))
            # 对合并后的qid进行排序
            new4_new_sort_same_qids = sorted([a2qids_dict[aid] for aid in new4_new_same_aids])
            # 进行qid过滤
            new4_filter_sort_qids = sorted([i for i in new4_new_sort_same_qids if i not in new3_filter_qids])

            print('###################################2018XML回复阶段执行完毕！###################################')

            for qid in new4_filter_sort_qids:
                # 获取查询
                query = query_dict[(qid, q2aids_dict[qid])]
                # 每行数据
                line = line_dict[q2aids_dict[qid]]
                # row标签
                rowTag = BeautifulSoup(line, 'lxml').row
                # body属性 一定存在
                body = get_tag_param(rowTag, 'body')
                htmlSoup = BeautifulSoup(body, 'lxml')
                # 抽取代码和上下文 直接用soup（不能转了str)
                block_list = parse_tag_block(htmlSoup)

                if block_list != []:
                    # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度]
                    # 候选的遍历
                    for index in range(0, len(block_list)):
                        # 创建块组合
                        block_addlis = []
                        # 添加索引
                        block_addlis.append((qid, index))
                        # 添加上下文
                        block_tuple = block_list[index]
                        block_addlis.append([[block_tuple[0]], [block_tuple[-1]]])
                        # 添加代码
                        block_addlis.append([[block_tuple[1]]])
                        block_addlis.append([query])
                        block_addlis.append(len(block_list))
                        # 加入数据
                        codedb_lis.append(block_addlis)

        print('###################################2018XML回复阶段解析完毕！###################################')


        new4_filter_qids = set([i[0][0] for i in codedb_lis])

        # 帖子id
        new5_new_qidnum_lis = []

        # 接受id
        new5_new_accept_lis = []

        # 计数
        count1 = 0
        count2 = 0

        with open(data_folder + '%s_tag_2017.xml'%lang_type, 'r', encoding='utf-8') as f:
            for line in f:
                count1 += 1
                if count1 % 100000 == 0:  # 总计
                    print('2017XML查询阶段已经处理--%d条数据' % count1)
                rowTag = BeautifulSoup(line, 'lxml').row  # row标签
                # 查询
                query = get_tag_param(rowTag, 'title')
                # 判断查询是否满足要求
                if query != '-10000':  # 存在
                    # 过滤查询query
                    queryFilter = s.filter(query)
                    if queryFilter == 0:
                        # 可接受
                        accept = get_tag_param(rowTag, 'acceptedanswerid')
                        if accept != '-10000' and lang_type in get_tag_param(rowTag, "tags"):  # 存在
                            count2 += 1
                            if count2 % 100000 == 0:  # 总计
                                print('2017XML查询阶段有效处理--%d条数据' % count2)
                            # 接受id
                            new5_new_accept_lis.append(int(accept))
                            # 帖子id
                            qidnum = get_tag_param(rowTag, 'id')
                            new5_new_qidnum_lis.append(int(qidnum))
                            # 字典数据 （更新/添加）
                            query_dict[(int(qidnum), int(accept))] = query

        print('2017XML查询阶段完全遍历处理-%d条数据' % count1)
        print('2017XML查询阶段有效总计处理-%d条数据' % count2)

        print('###################################2017XML查询阶段执行完毕！###################################')

        # accept从小到大的序号排序
        new5_new_sort_accept_lis = sorted(new2_new_accept_lis)

        print('2017XML的文件可解析回复qid长度-%s' % len(new5_new_qidnum_lis))

        ###############################################找寻相关的XML文件#########################################

        # 索引的长度
        new5_new_accept_num = len(new5_new_sort_accept_lis)
        print('2017XML回复答案aid需要的解析长度-%d' % new5_new_accept_num)

        # 来对索引index分割,正则匹配命令太长，必须切割！
        split_index = [new5_new_sort_accept_lis[i:i + split_num] for i in range(0, len(new5_new_sort_accept_lis), split_num)]
        # 执行每个部分index
        parts_file = []

        for i in range(len(split_index)):
            # linux执行命令
            eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
            # 每份xml命名
            epart_xml = '%s_codedb_oth_2017.xml' % lang_type
            epart_xml = epart_xml.replace('oth_2017', 'oth_2017Part%d' % (i + 1))
            # 添加到列表
            parts_file.append(data_folder + epart_xml)
            # 命令字符串
            eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str, data_folder + 'answer_2017.xml', data_folder + epart_xml)
            # 执行命令符
            os.popen(eping_cmd).read()  # 得到part_xml文件
            print('索引的第%d部分检索完毕！' % (i + 1))

        ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_codedb_oth_2017.xml' % lang_type)

        # 执行命令
        os.popen(ping_cmd).read()  # 得到ans_xml文件
        print('XML整体文件合并完毕!')

        remove_cmd = 'rm %s' % data_folder + '%s_codedb_oth_2017Part*.xml' % lang_type
        os.popen(remove_cmd).read()
        # print('XML部分文件删除完毕!')

        ###################################解析具有全部accept_lis索引的ans_xml文件#######################################
        # 序号id对应accept的id  qid：accept_id (aid)
        for i in query_dict.keys():
            q2aids_dict[i[0]] = i[1]

        for i in query_dict.keys():
            a2qids_dict[i[1]] = i[0]

        # 计数
        count1 = 0

        with open(data_folder + '%s_codedb_oth_2017.xml' % lang_type, 'r', encoding='utf-8') as f:
            canfind_aids = []
            for line in f:
                count1 += 1
                if count1 % 100000 == 0:  # 总计
                    print('2017XML回复阶段已经处理--%d条数据' % count1)
                sta = line.find("\"")
                end = line.find("\"", sta + 1)
                # 提取line对应的accept id
                aid = int(line[(sta + 1):end])
                canfind_aids.append(aid)
                # 更新网页
                line_dict[aid] = line

            # 对查询的键值和accept键值进行合并
            new5_new_same_aids = (set(new5_new_accept_lis) & set(canfind_aids))
            print('2017XML回复答案aid合并的有效长度-%d' % len(new5_new_same_aids))
            # 对合并后的qid进行排序
            new5_new_sort_same_qids = sorted([a2qids_dict[aid] for aid in new5_new_same_aids])
            # 进行qid过滤
            new5_filter_sort_qids = sorted([i for i in new5_new_sort_same_qids if i not in new4_filter_qids])

            print('###################################2017XML回复阶段执行完毕！###################################')

            for qid in new5_filter_sort_qids:
                # 获取查询
                query = query_dict[(qid, q2aids_dict[qid])]
                # 每行数据
                line = line_dict[q2aids_dict[qid]]
                # row标签
                rowTag = BeautifulSoup(line, 'lxml').row
                # body属性 一定存在
                body = get_tag_param(rowTag, 'body')
                htmlSoup = BeautifulSoup(body, 'lxml')
                # 抽取代码和上下文 直接用soup（不能转了str)
                block_list = parse_tag_block(htmlSoup)

                if block_list != []:
                    # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度]
                    # 候选的遍历
                    for index in range(0, len(block_list)):
                        # 创建块组合
                        block_addlis = []
                        # 添加索引
                        block_addlis.append((qid, index))
                        # 添加上下文
                        block_tuple = block_list[index]
                        block_addlis.append([[block_tuple[0]], [block_tuple[-1]]])
                        # 添加代码
                        block_addlis.append([[block_tuple[1]]])
                        block_addlis.append([query])
                        block_addlis.append(len(block_list))
                        # 加入数据
                        codedb_lis.append(block_addlis)

        print('###################################2017XML回复阶段解析完毕！###################################')

        # 判断查询为空
        noquery_qids = set([i[0][0] for i in codedb_lis if  python_structured.python_query_parse(i[3][0]) == []]) if lang_type == 'python' else \
            set([i[0][0] for i in codedb_lis if sqlang_structured.sqlang_query_parse(i[3][0]) == []])

        print('codedb最终查询解析为空的qid索引的长度-%d\n'%len(noquery_qids), noquery_qids)

        # 过滤掉不满足要求
        codedb_lis = [i for i in codedb_lis if i[0][0] not in noquery_qids]

        # 判断代码结构
        noparse_qids = set([i[0][0] for i in codedb_lis if python_structured.python_code_parse(i[2][0][0]) == '-1000']) if lang_type == 'python' else \
                           set([i[0][0] for i in codedb_lis if sqlang_structured.sqlang_code_parse(i[2][0][0]) == '-1000'])

        print('codedb最终代码无法解析的qid索引的长度-%d\n'% len(noparse_qids), noparse_qids)

        # 过滤掉不满足要求
        codedb_lis = [i for i in codedb_lis if i[0][0] not in noparse_qids]
        codedb_qids = set([i[0][0] for i in codedb_lis])
        print(len(codedb_qids))

        new_final_qids = set([i[0][0] for i in codedb_lis])

        # 从小到大排序
        sort_codedb_lis = sorted(codedb_lis, key=lambda x: (x[0][0], x[0][1]))

        with open(save_folder+'%s_codedb_qid2index_blocks_unlabeled.pickle'%lang_type, "wb") as f:
            pickle.dump(sort_codedb_lis, f)

        print('codedb最终能对应qid索引长度-%d'%len(new_final_qids))



# 参数配置设置
data_folder = '../corpus_xml/'
save_folder ='./corpus/'

split_num = 5000

sql_type ='sql'
python_type= 'python'


if __name__ == '__main__':
    # 抽取xml
    #extract_differ_xml(data_folder, 'posts_2017.xml','python_tag_2017.xml','sql_tag_2017.xml','answer_2017.xml')
    #extract_differ_xml(data_folder, 'posts_2018.xml','python_tag_2018.xml','sql_tag_2018.xml','answer_2018.xml')
    #extract_differ_xml(data_folder, 'posts_2019.xml','python_tag_2019.xml','sql_tag_2019.xml','answer_2019.xml')
    #extract_differ_xml(data_folder, 'posts_2020.xml','python_tag_2020.xml','sql_tag_2020.xml','answer_2020.xml')
    #extract_differ_xml(data_folder, 'posts_2021.xml','python_tag_2021.xml','sql_tag_2021.xml','answer_2021.xml')
    #extract_differ_xml(data_folder, 'posts_2022.xml','python_tag_2022.xml','sql_tag_2022.xml','answer_2022.xml')
    # 处理python
    process_xmldata(data_folder, python_type,split_num,save_folder)
    # 处理sql
    process_xmldata(data_folder, sql_type,split_num,save_folder)









