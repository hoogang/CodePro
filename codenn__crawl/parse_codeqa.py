#!/usr/bin/env python3
import os
import re
import pickle

#网页转义
import html
#解析数据
from bs4 import BeautifulSoup

import sys
sys.path.append("..")

# 解析结构
from crawl_process.parsers import python_structured
from crawl_process.parsers import sqlang_structured


#--------------------------linux代码，配置可执行-----------------------
def extract_xml(data_folder,posts_xml,python_xml,sqlang_xml,answer_xml):

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

#解析属性标签
def get_tag_param(rowTag,tag):
    try:
        output = rowTag[tag]
        return output
    except KeyError:
        return '-10000'

#解析code中数据
def parse_all_mucodes(codeText):
    #  候选代码
    mucodes = []
    #  代码一个或多个标签
    for oneTag in codeText:
        code = oneTag.get_text().strip()
        #  代码片段10-1000字符,过滤字符长度
        if (len(code) >= 10 and len(code) <= 1000):
            #  过滤代码的关键片段
            if code[0] == '<' or code[0] == '=' or code[0] == '@' or code[0] == '$' or \
                    code[0:7].lower() == 'select' or code[0:7].lower() == 'update' or code[0:6].lower() == 'alter' or \
                    code[0:2].lower() == 'c:' or code[0:4].lower() == 'http' or code[0:4].lower() == 'hkey' or \
                    re.match(r'^[a-zA-Z0-9_]*$', code) is not None:
                #  加入代码
                mucodes.append('-1000')
            else:
                #  满足要求
                # 对网页标签转义
                code = html.unescape(code)
                mucodes.append(code)
        else:
            #  代码长度不满足要求
            mucodes.append('-1000')

    return mucodes


def parse_tag_code(htmlSoup):
    # 代码code
    codeText = htmlSoup.find_all('code')
    # 至少有一个或多个标签
    mucodes = parse_all_mucodes(codeText)

    code_list=[]

    # 判断候选个数，最佳代码存在
    if len(mucodes) > 1 and mucodes[0] != '-1000':
        # 去除非条件候选,更新候选集
        mucodes = [code for code in mucodes if code != '-1000']
        # 至少第二个候选满足条件，其他候选不作参考
        if len(mucodes) > 1:
            for code in  mucodes:
                code_list.append(code)

    return code_list

#存储query的数据
def process_xmldata(data_folder,lang_type,split_num):


    # 帖子id
    qidnum_lis = []
    # 接受id
    accept_lis = []
    # 查询
    query_dict = {}

    # 计数
    count1=0
    count2=0

    with open(data_folder+'%s_staqc_map_2017.xml'%lang_type,'r',encoding='utf-8') as f:
        for line in f:
            count1+=1
            if count1 % 10000 == 0:  # 总计
                print('2017XML查询阶段已经处理--%d条数据' % count1)
            rowTag=BeautifulSoup(line,'lxml').row #row标签
            # 查询
            query=get_tag_param(rowTag,'title')
            # 判断查询是否满足要求
            if query != '-10000':  # 存在
                accept=get_tag_param(rowTag,'acceptedanswerid')
                if accept!='-10000'and lang_type in get_tag_param(rowTag, "tags") :  # 存在:
                    count2+=1
                    if count2 % 10000 == 0:  # 总计
                        print('2017XML查询阶段有效处理--%d条数据' % count2)
                    # 接受id
                    accept_lis.append(int(accept))
                    # 帖子id
                    qidnum = get_tag_param(rowTag, 'id')
                    qidnum_lis.append(int(qidnum))
                    # 字典数据 （更新/添加）
                    query_dict[(int(qidnum), int(accept))] = query

    print('2017XML查询阶段完全遍历处理-%d条数据'%count1)
    print('2017XML查询阶段有效总计处理-%d条数据'%count2)


    print('###################################2017XML查询阶段执行完毕！###################################')

    #  判断长度是否相等
    assert len(accept_lis) == len(query_dict.keys())

    print('2017XML的文件可解析回复qid长度-%s' % len(qidnum_lis))

    # ###############################################找寻相关的XML文件#########################################

    ###################################解析具有全部accept_lis索引的ans_xml文件#######################################
    # 序号id对应accept的id  qid：accept_id (aid)
    q2aids_dict = {}
    for i in query_dict.keys():
        q2aids_dict[i[0]] = i[1]

    a2qids_dict = {}
    for i in query_dict.keys():
        a2qids_dict[i[1]] = i[0]

    with open(data_folder + '%s_staqc_ans_2017.xml' % lang_type, 'r') as f:
        # 构建词典 aid:line
        line_dict = {}
        for line in f:
            sta = line.find("\"")
            end = line.find("\"", sta + 1)
            # 提取line对应的accept id
            aid = int(line[(sta + 1):end])
            line_dict[aid] = line
        # 对查询的键值和accept键值进行合并
        same_aids = (set(accept_lis) & set(line_dict.keys()))
        print('2017XML回复答案id合并的有效长度-%d' % len(same_aids))
        # 对合并后的qid进行从小到大排序
        sort_same_qids = sorted([a2qids_dict[aid] for aid in same_aids])

        print('###################################2017XML回复阶段执行完毕！###################################')

        # 不包含社交属性语料
        codeqa_lis = []

        for qid in sort_same_qids:
            # 获取查询
            query = query_dict[(qid, q2aids_dict[qid])]

            # 每行数据
            line = line_dict[q2aids_dict[qid]]
            # row标签
            rowTag = BeautifulSoup(line, 'lxml').row
            # body属性 一定存在
            body = get_tag_param(rowTag, 'body')
            # <html><body>
            htmlSoup = BeautifulSoup(body, 'lxml')

            # 抽取代码和上下文 直接用soup（不能转了str)
            code_list = parse_tag_code(htmlSoup)

            if code_list != []:

                # [(id, index), [[c]] 代码，[q] 查询,  块长度]
                # 候选的遍历
                for index in range(0, len(code_list)):
                    # 创建块组合
                    block_addlis = []
                    # 添加索引
                    block_addlis.append((qid, index))
                    # 添加代码
                    code = code_list[index]
                    block_addlis.append([[code]])
                    block_addlis.append([query])
                    block_addlis.append(len(code_list))
                    # 加入数据
                    codeqa_lis.append(block_addlis)

    print('###################################2018XML回复阶段解析完毕！###################################')

    # 判断查询为空
    noquery_qids = set([i[0][0] for i in codeqa_lis if  python_structured.python_query_parse(i[2][0]) == []]) if lang_type == 'python' else \
        set([i[0][0] for i in codeqa_lis if sqlang_structured.sqlang_query_parse(i[2][0]) == []])

    print('codeqa开始查询分词为空qid索引的长度-%d\n' % len(noquery_qids), noquery_qids)

    # 过滤掉不满足要求
    codeqa_lis = [i for i in codeqa_lis if i[0][0] not in noquery_qids]

    # 判断代码结构
    noparse_qids = set([i[0][0] for i in codeqa_lis if  python_structured.python_code_parse(i[1][0][0]) == '-1000']) if lang_type == 'python' \
        else set([i[0][0] for i in codeqa_lis if sqlang_structured.sqlang_code_parse(i[1][0][0]) == '-1000'])

    print('codeqa开始代码无法解析qid索引的长度-%d\n' % len(noparse_qids), noparse_qids)

    # 过滤掉不满足要求
    codeqa_lis = [i for i in codeqa_lis if i[0][0] not in noparse_qids]

    # 帖子id
    new_qidnum_lis = []

    # 接受id
    new_accept_lis = []

    # 计数
    count1 = 0
    count2 = 0

    with open(data_folder + '%s_staqc_map_2018.xml' % lang_type, 'r', encoding='utf-8') as f:
        for line in f:
            count1 += 1
            if count1 % 10000 == 0:  # 总计
                print('2018XML查询阶段已经处理--%d条数据' % count1)
            rowTag = BeautifulSoup(line, 'lxml').row  # row标签
            # 查询
            query = get_tag_param(rowTag, 'title')
            # 判断查询是否满足要求
            if query != '-10000':  # 存在
                accept = get_tag_param(rowTag, 'acceptedanswerid')
                if accept != '-10000' and lang_type in get_tag_param(rowTag, "tags"):  # 存在
                    count2 += 1
                    if count2 % 10000 == 0:  # 总计
                        print('2018XML查询阶段有效处理--%d条数据' % count2)
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
    print('2018XML回复答案aid需要的补充长度-%d' % new_accept_num)

    ##################################解析具有全部accept_lis索引的ans_xml文件#######################################
    # 序号id对应accept的id  qid：accept_id (aid)
    for i in query_dict.keys():
        q2aids_dict[i[0]] = i[1]

    for i in query_dict.keys():
        a2qids_dict[i[1]] = i[0]

    # 计数
    count1 = 0

    with open(data_folder + '%s_staqc_ans_2018.xml' % lang_type, 'r', encoding='utf-8') as f:
        canfind_aids = []
        for line in f:
            count1 += 1
            if count1 % 10000 == 0:  # 总计
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
        new_sort_same_qids = sorted([a2qids_dict[aid] for aid in new_same_aids])

        print('###################################2018XML回复阶段执行完毕！###################################')

        for qid in  new_sort_same_qids:
            # 获取查询
            query = query_dict[(qid, q2aids_dict[qid])]
            # 每行数据
            line = line_dict[q2aids_dict[qid]]
            # row标签
            rowTag = BeautifulSoup(line, 'lxml').row
            # body属性 一定存在
            body = get_tag_param(rowTag, 'body')
            # <html><body>
            htmlSoup = BeautifulSoup(body, 'lxml')

            # 抽取代码和上下文 直接用soup（不能转了str)
            code_list = parse_tag_code(htmlSoup)

            if code_list != []:

                # [(id, index), [[c]] 代码，[q] 查询,  块长度]
                # 候选的遍历
                for index in range(0, len(code_list)):
                    # 创建块组合
                    block_addlis = []
                    # 添加索引
                    block_addlis.append((qid, index))
                    # 添加代码
                    code = code_list[index]
                    block_addlis.append([[code]])
                    block_addlis.append([query])
                    block_addlis.append(len(code_list))
                    # 加入数据
                    codeqa_lis.append(block_addlis)

    print('###################################2018XML回复阶段解析完毕！###################################')

    find_htmls = { (33137311, 33487726): ["How to model several tournaments / brackets types into a SQL database?", "<html><body><p>You could create tables to hold tournament types, league types, playoff types, and have a schedule table, showing an even name along with its tournament type, and then use that relationship to retrieve information about that tournament. Note, this is not MySQL, this is more generic SQL language:</p>\n\n<pre><code>CREATE TABLE tournTypes (\nID int autoincrement primary key,\nleagueId int constraint foreign key references leagueTypes.ID,\nplayoffId int constraint foreign key references playoffTypes.ID\n--...other attributes would necessitate more tables\n)\n\nCREATE TABLE leagueTypes(\nID int autoincrement primary key,\nnoOfTeams int,\nnoOfDivisions int,\ninterDivPlay bit -- e.g. a flag indicating if teams in different divisions would play\n)\n\nCREATE TABLE playoffTypes(\nID int autoincrement primary key,\nnoOfTeams int,\nisDoubleElim bit -- e.g. flag if it is double elimination\n)\n\nCREATE TABLE Schedule(\nID int autoincrement primary key,\nName text,\nstartDate datetime,\nendDate datetime,\ntournId int constraint foreign key references tournTypes.ID\n)\n</code></pre>\n\n<p>Populating the tables...</p>\n\n<pre><code>INSERT INTO tournTypes VALUES\n(1,2),\n(1,3),\n(2,3),\n(3,1)\n\nINSERT INTO leagueTypes VALUES\n(16,2,0), -- 16 teams, 2 divisions, teams only play within own division\n(8,1,0),\n(28,4,1)\n\nINSERT INTO playoffTypes VALUES\n(8,0), -- 8 teams, single elimination\n(4,0),\n(8,1)\n\nINSERT INTO Schedule VALUES\n('Champions league','2015-12-10','2016-02-10',1),\n('Rec league','2015-11-30','2016-03-04-,2)\n</code></pre>\n\n<p>Getting info on a tournament...</p>\n\n<pre><code>SELECT Name\n,startDate\n,endDate\n,l.noOfTeams as LeagueSize\n,p.noOfTeams as PlayoffTeams\n,case p.doubleElim when 0 then 'Single' when 1 then 'Double' end as Elimination\nFROM Schedule s\nINNER JOIN tournTypes t\nON s.tournId = t.ID\nINNER JOIN leagueTypes l\nON t.leagueId = l.ID\nINNER JOIN playoffTypes p\nON t.playoffId = p.ID\n</code></pre></body></html>"]}

    new_html_qids = [i[0] for i in find_htmls.keys()]

    if lang_type == 'sql':
        for i in find_htmls.keys():
            q2aids_dict[i[0]] = i[1]
        # 排除这些网页
        codeqa_lis = [i for i in codeqa_lis if i[0][0] not in new_html_qids]
        # 先判断不为空
        for qid in new_html_qids:
            # 获取查询
            query = find_htmls[(qid, q2aids_dict[qid])][0]
            # 每行数据
            line = find_htmls[(qid, q2aids_dict[qid])][1]
            htmlSoup = BeautifulSoup(line, 'lxml')

            # 抽取代码和上下文 直接用soup（不能转了str)
            code_list = parse_tag_code(htmlSoup)

            if code_list != []:

                # [(id, index), [[c]] 代码，[q] 查询,  块长度]
                # 候选的遍历
                for index in range(0, len(code_list)):
                    # 创建块组合
                    block_addlis = []
                    # 添加索引
                    block_addlis.append((qid, index))
                    # 添加代码
                    code = code_list[index]
                    block_addlis.append([[code]])
                    block_addlis.append([query])
                    block_addlis.append(len(code_list))
                    # 加入数据
                    codeqa_lis.append(block_addlis)

    # 判断查询为空
    new_noquery_qids = set([i[0][0] for i in codeqa_lis if python_structured.python_query_parse(i[2][0]) == []]) if lang_type == 'python' else \
        set([i[0][0] for i in codeqa_lis if sqlang_structured.sqlang_query_parse(i[2][0]) == []])

    print('codeqa存在查询分词为空qid索引的长度-%d\n' % len(new_noquery_qids), new_noquery_qids)

    # 过滤掉不满足要求
    codeqa_lis = [i for i in codeqa_lis if i[0][0] not in new_noquery_qids]

    # 判断代码结构
    new_noparse_qids = set([i[0][0] for i in codeqa_lis if python_structured.python_code_parse(i[1][0][0]) == '-1000']) if lang_type == 'python' else \
        set([i[0][0] for i in codeqa_lis if sqlang_structured.sqlang_code_parse(i[1][0][0]) == '-1000'])

    print('codeqa存在代码无法解析qid索引的长度-%d\n' % len(new_noparse_qids), new_noparse_qids)

    # 过滤掉不满足要求
    codeqa_lis = [i for i in codeqa_lis if i[0][0] not in new_noparse_qids]


    with open('../final_collection/%s_staqc&samqc_common_qids.txt'% lang_type,"r") as f:
        # staqc和samqc交集
        common_qids=[eval(i) for i in f.read().splitlines()]
        # 为了参照多候选必须和staqc-samqc一致

    # 过滤掉不满足要求多候选
    codeqa_lis = [i for i in codeqa_lis if i[0][0] in common_qids]

    codeqa_qids = set([i[0][0] for i in codeqa_lis])

    print('codeqa开始保留的mutile-qid长度-%d\n' %len(codeqa_qids))     #sql: 26468   python：42565

    print('codeqa开始保留的mutile-qid-index长度-%d\n'%len(codeqa_lis)) #sql: 77263   python:141023

    # 补齐单候选类型
    single_qids = pickle.load(open('../label_tool/data/code_solution_labeled_data/filter/%s_staqc_single_qids.pickle' % lang_type, "rb"))

    # 对单候选进行排序
    single_qids = sorted(single_qids)

    # 来对索引index分割,正则匹配命令太长，必须切割！
    split_index = [single_qids[i:i + split_num] for i in range(0, len(single_qids), split_num)]

    # 执行每个部分index
    parts_file = []

    for i in range(len(split_index)):
        # linux执行命令
        eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
        # 每份xml命名
        epart_xml = '%s_codeqa_map_2017.xml' % lang_type
        epart_xml = epart_xml.replace('map_2017', 'map_2017Part%d' % (i + 1))
        # 添加到列表
        parts_file.append(data_folder + epart_xml)
        # 命令字符串
        eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (
        eping_str, data_folder + 'posts_2017.xml', data_folder + epart_xml)
        # 执行命令符
        os.popen(eping_cmd).read()  # 得到part_xml文件
        print('索引的第%d部分检索完毕！' % (i + 1))

    ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_codeqa_map_2017.xml' % lang_type)
    # 执行命令
    os.popen(ping_cmd).read()
    print('XML整体文件合并完毕!')

    remove_cmd = 'rm %s' % data_folder + '%s_codeqa_map_2017Part*.xml' % lang_type
    os.popen(remove_cmd).read()
    print('XML部分文件删除完毕!')

    # 帖子id
    new0_qidnum_lis = []

    # 接受id
    new0_accept_lis = []

    # 计数
    count1 = 0
    count2 = 0

    with open(data_folder+'%s_codeqa_map_2017.xml'%lang_type,'r') as f:
        for line in f:
            count1 += 1
            if count1 % 10000 == 0:  # 总计
                print('2017XML查询阶段已经处理--%d条数据' % count1)
            rowTag = BeautifulSoup(line, 'lxml').row  # row标签
            # 查询
            query = get_tag_param(rowTag, 'title')
            # 判断查询是否满足要求
            if query != '-10000':  # 存在
                # 可接受
                accept = get_tag_param(rowTag, 'acceptedanswerid')
                if accept != '-10000':  # 存在
                    count2 += 1
                    if count2 % 10000 == 0:  # 总计
                        print('2017XML查询阶段有效处理--%d条数据' % count2)
                    # 接受id
                    new0_accept_lis.append(int(accept))
                    # 帖子id
                    qidnum = get_tag_param(rowTag, 'id')
                    new0_qidnum_lis.append(int(qidnum))
                    # 字典数据
                    query_dict[(int(qidnum), int(accept))] = query

    print('2017XML查询阶段完全遍历处理-%d条数据' % count1)
    print('2017XML查询阶段有效总计处理-%d条数据' % count2)

    print('###################################2017XML查询阶段执行完毕！###################################')

    # accept从小到大的序号排序
    new0_sort_accept_lis = sorted(new0_accept_lis)

    print('2017XML的codeqa中原始的qid长度-%s' % len(single_qids))
    print('2017XML的文件可解析回复qid长度-%s' % len(new0_qidnum_lis))

    noblock_qids = set(single_qids).difference(set(new0_qidnum_lis))

    # python: ,sql:
    print('2017XML的解析回复不在的qid长度-%s\n' % len(noblock_qids), noblock_qids)

    ###############################################找寻相关的XML文件#########################################
    # 索引的长度
    new0_accept_num = len(new0_sort_accept_lis)
    print('2017XML回复答案aid需要的补充长度-%d' % new0_accept_num)

    # 来对索引index分割,正则匹配命令太长，必须切割！
    split_index = [new0_sort_accept_lis[i:i + split_num] for i in range(0, len(new0_sort_accept_lis), split_num)]

    # 执行每个部分index
    parts_file = []

    for i in range(len(split_index)):
        # linux执行命令
        eping_str = '|'.join(['(%s)' % str(s) for s in split_index[i]])
        # 每份xml命名
        epart_xml = '%s_codeqa_ans_2017.xml' % lang_type
        epart_xml = epart_xml.replace('ans_2017', 'ans_2017Part%d' % (i + 1))
        # 添加到列表
        parts_file.append(data_folder+epart_xml)
        # 命令字符串
        eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str, data_folder + 'answer_2017.xml', data_folder + epart_xml)
        # 执行命令符
        os.popen(eping_cmd).read()  # 得到part_xml文件
        print('索引的第%d部分检索完毕！' % (i + 1))

    ping_cmd = 'cat %s > %s' % (' '.join(parts_file), data_folder + '%s_codeqa_ans_2017.xml'%lang_type)

    # 执行命令
    os.popen(ping_cmd).read()  # 得到ans_xml文件
    print('XML整体文件合并完毕!')

    remove_cmd =  'rm %s' % data_folder + '%s_codeqa_ans_2017Part*.xml'% lang_type
    os.popen(remove_cmd).read()
    # print('XML部分文件删除完毕!')

    ##################################解析具有全部accept_lis索引的ans_xml文件#######################################
    # 序号id对应accept的id  qid：accept_id (aid)
    for i in query_dict.keys():
        q2aids_dict[i[0]] = i[1]

    for i in query_dict.keys():
        a2qids_dict[i[1]] = i[0]

    # 计数
    count1 = 0

    with open(data_folder + '%s_codeqa_ans_2017.xml' % lang_type, 'r', encoding='utf-8') as f:
        canfind_aids = []
        for line in f:
            count1 += 1
            if count1 % 10000 == 0:  # 总计
                print('2017XML回复阶段已经处理--%d条数据' % count1)
            sta = line.find("\"")
            end = line.find("\"", sta + 1)
            # 提取line对应的accept id
            aid = int(line[(sta + 1):end])
            canfind_aids.append(aid)
            # 更新网页
            line_dict[aid] = line

        # 对查询的键值和accept键值进行合并
        new0_same_aids = (set(new0_sort_accept_lis) & set(canfind_aids))
        print('2017XML回复答案aid合并的有效长度-%d' % len(new0_same_aids))
        # 对合并后的qid进行排序
        new0_sort_same_qids = sorted([a2qids_dict[aid] for aid in new0_same_aids])

        print('###################################2017XML回复阶段执行完毕！###################################')

        for qid in  new0_sort_same_qids:
            # 获取查询
            query = query_dict[(qid, q2aids_dict[qid])]
            # 每行数据
            line = line_dict[q2aids_dict[qid]]
            # row标签
            rowTag = BeautifulSoup(line, 'lxml').row
            # body属性 一定存在
            body = get_tag_param(rowTag, 'body')
            # <html><body>
            htmlSoup = BeautifulSoup(body, 'lxml')

            # 抽取代码和上下文 直接用soup（不能转了str)
            code_list = parse_tag_code(htmlSoup)

            if code_list != []:

                # [(id, index), [[c]] 代码，[q] 查询,  块长度]
                # 候选的遍历
                for index in range(0, len(code_list)):
                    # 创建块组合
                    block_addlis = []
                    # 添加索引
                    block_addlis.append((qid, index))
                    # 添加代码
                    code = code_list[index]
                    block_addlis.append([[code]])
                    block_addlis.append([query])
                    block_addlis.append(len(code_list))
                    # 加入数据
                    codeqa_lis.append(block_addlis)

    # 判断查询为空
    new0_noquery_qids = set([i[0][0] for i in codeqa_lis if python_structured.python_query_parse(i[2][0]) == []]) if lang_type == 'python' else \
        set([i[0][0] for i in codeqa_lis if sqlang_structured.sqlang_query_parse(i[2][0]) == []])

    print('codeqa存在查询分词为空qid索引的长度-%d\n' % len(new_noquery_qids), new0_noquery_qids)

    # 过滤掉不满足要求
    codeqa_lis = [i for i in codeqa_lis if i[0][0] not in new0_noquery_qids]

    # 判断代码结构
    new0_noparse_qids = set([i[0][0] for i in codeqa_lis if python_structured.python_code_parse(i[1][0][0]) == '-1000']) if lang_type == 'python' else \
        set([i[0][0] for i in codeqa_lis if sqlang_structured.sqlang_code_parse(i[1][0][0]) == '-1000'])

    print('codeqa存在代码无法解析qid索引的长度-%d\n' % len(new0_noparse_qids), new0_noparse_qids)

    # 过滤掉不满足要求
    codeqa_lis = [i for i in codeqa_lis if i[0][0] not in new0_noparse_qids]

    # 再到2018xml继续找单候选
    # 从小到大的序号排序
    sort_noblock_qids = sorted(noblock_qids)

    # linux执行命令
    eping_str = '|'.join(['(%s)' % str(s) for s in sort_noblock_qids])
    # 每份xml命名
    epart_xml = '%s_codeqa_map_2018.xml' % lang_type
    # 命令字符串 （不能先限定语言标签，直接在posts.xml中提取）
    eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str, data_folder + 'posts_2018.xml', data_folder + epart_xml)
    # 执行命令符
    os.popen(eping_cmd).read()

    # 帖子id
    new1_qidnum_lis = []

    # 接受id
    new1_accept_lis = []

    # 计数
    count1 = 0
    count2 = 0

    with open(data_folder+'%s_codeqa_map_2018.xml'%lang_type,'r') as f:
        for line in f:
            count1 += 1
            if count1 % 10000 == 0:  # 总计
                print('2018XML查询阶段已经处理--%d条数据' % count1)
            rowTag = BeautifulSoup(line, 'lxml').row  # row标签
            # 查询
            query = get_tag_param(rowTag, 'title')
            # 判断查询是否满足要求
            if query != '-10000':  # 存在
                # 可接受
                accept = get_tag_param(rowTag, 'acceptedanswerid')
                if accept != '-10000':  # 存在
                    count2 += 1
                    if count2 % 10000 == 0:  # 总计
                        print('2018XML查询阶段有效处理--%d条数据' % count2)
                    # 接受id
                    new1_accept_lis.append(int(accept))
                    # 帖子id
                    qidnum = get_tag_param(rowTag, 'id')
                    new1_qidnum_lis.append(int(qidnum))
                    # 字典数据
                    query_dict[(int(qidnum), int(accept))] = query

    print('2018XML查询阶段完全遍历处理-%d条数据' % count1)
    print('2018XML查询阶段有效总计处理-%d条数据' % count2)

    print('###################################2017XML查询阶段执行完毕！###################################')

    # accept从小到大的序号排序
    new1_sort_accept_lis = sorted(new1_accept_lis)

    print('2018XML的codeqa中原始的qid长度-%s' % len(noblock_qids))
    print('2018XML的文件可解析回复qid长度-%s' % len(new1_qidnum_lis))

    new_nofind_qids = set(noblock_qids).difference(set(new1_qidnum_lis))

    # python: ,sql:
    print('2018XML的解析回复不在的qid长度-%s\n' % len(new_nofind_qids), new_nofind_qids)

    ###############################################找寻相关的XML文件#########################################
    # 索引的长度
    new1_accept_num = len(new1_sort_accept_lis)
    print('2018XML回复答案aid需要的补充长度-%d' % new1_accept_num)

    # linux执行命令
    eping_str = '|'.join(['(%s)' % str(s) for s in new1_sort_accept_lis])
    # 每份xml命名
    epart_xml = '%s_codeqa_ans_2018.xml' % lang_type
    # 命令字符串
    eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str, data_folder + 'answer_2018.xml', data_folder + epart_xml)
    # 执行命令符
    os.popen(eping_cmd).read()

    ##################################解析具有全部accept_lis索引的ans_xml文件#######################################
    # 序号id对应accept的id  qid：accept_id (aid)
    for i in query_dict.keys():
        q2aids_dict[i[0]] = i[1]

    for i in query_dict.keys():
        a2qids_dict[i[1]] = i[0]

    # 计数
    count1 = 0

    with open(data_folder + '%s_codeqa_ans_2018.xml' % lang_type, 'r', encoding='utf-8') as f:
        canfind_aids = []
        for line in f:
            count1 += 1
            if count1 % 10000 == 0:  # 总计
                print('2018XML回复阶段已经处理--%d条数据' % count1)
            sta = line.find("\"")
            end = line.find("\"", sta + 1)
            # 提取line对应的accept id
            aid = int(line[(sta + 1):end])
            canfind_aids.append(aid)
            # 更新网页
            line_dict[aid] = line

        # 对查询的键值和accept键值进行合并
        new1_same_aids = (set(new1_sort_accept_lis) & set(canfind_aids))
        print('2018XML回复答案aid合并的有效长度-%d' % len(new1_same_aids))
        # 对合并后的qid进行排序
        new1_sort_same_qids = sorted([a2qids_dict[aid] for aid in new1_same_aids])

        print('###################################2017XML回复阶段执行完毕！###################################')

        for qid in  new1_sort_same_qids:
            # 获取查询
            query = query_dict[(qid, q2aids_dict[qid])]
            # 每行数据
            line = line_dict[q2aids_dict[qid]]
            # row标签
            rowTag = BeautifulSoup(line, 'lxml').row
            # body属性 一定存在
            body = get_tag_param(rowTag, 'body')
            # <html><body>
            htmlSoup = BeautifulSoup(body, 'lxml')

            # 抽取代码和上下文 直接用soup（不能转了str)
            code_list = parse_tag_code(htmlSoup)

            if code_list != []:

                # [(id, index), [[c]] 代码，[q] 查询,  块长度]
                # 候选的遍历
                for index in range(0, len(code_list)):
                    # 创建块组合
                    block_addlis = []
                    # 添加索引
                    block_addlis.append((qid, index))
                    # 添加代码
                    code = code_list[index]
                    block_addlis.append([[code]])
                    block_addlis.append([query])
                    block_addlis.append(len(code_list))
                    # 加入数据
                    codeqa_lis.append(block_addlis)

    # 判断查询为空
    new1_noquery_qids = set([i[0][0] for i in codeqa_lis if python_structured.python_query_parse(i[2][0]) == []]) if lang_type == 'python' else \
        set([i[0][0] for i in codeqa_lis if sqlang_structured.sqlang_query_parse(i[2][0]) == []])

    print('codeqa最终查询分词为空qid索引的长度-%d\n' % len(new1_noquery_qids), new1_noquery_qids)

    # 过滤掉不满足要求
    codeqa_lis = [i for i in codeqa_lis if i[0][0] not in new1_noquery_qids]

    # 判断代码结构
    new1_noparse_qids = set([i[0][0] for i in codeqa_lis if python_structured.python_code_parse(i[1][0][0]) == '-1000']) if lang_type == 'python' else \
        set([i[0][0] for i in codeqa_lis if sqlang_structured.sqlang_code_parse(i[1][0][0]) == '-1000'])

    print('codeqa最终代码无法解析qid索引的长度-%d\n' % len(new1_noparse_qids), new1_noparse_qids)

    # 过滤掉不满足要求
    codeqa_lis = [i for i in codeqa_lis if i[0][0] not in new1_noparse_qids]

    codeqa_qids = set([i[0][0] for i in codeqa_lis])

    print('codeqa中最终的mutiple-qid索引的长度-%d' % len(codeqa_qids))

    print('codeqa中最终对应的coprpus的mutiple-qid-index长度-%d' % len(codeqa_lis))

    # 从小到大排序
    sort_codeqa_lis = sorted(codeqa_lis, key=lambda x: (x[0][0], x[0][1]))

    with open('./corpus/%s_codeqa_qid2index_blocks_labeled.pickle' % lang_type, "wb") as f:
        pickle.dump(sort_codeqa_lis, f)





#参数配置设置
data_folder='../corpus_xml/'

split_num = 5000


sql_type='sql'
python_type='python'


#python
'''

'''

#sql
'''

'''

if __name__ == '__main__':
    # 处理sql
    #process_xmldata(data_folder,sql_type,split_num)
    # 处理python
    process_xmldata(data_folder,python_type,split_num)









    









