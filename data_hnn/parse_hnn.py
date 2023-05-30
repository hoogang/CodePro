#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import html
import pickle


# 解析数据
from bs4 import BeautifulSoup

# --------------------------linux代码，配置可执行-----------------------
def extract_differ_xml(data_folder, posts_xml, answer_xml):
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


# 匹配最后一个块
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
            n_f = match_str_ptag(p_context, match_str_block(p_forward, all_blocklis[i])[0]) if p_context.findall(
                match_str_block(p_forward, all_blocklis[i])[0]) != [] else '-10000'
            # print('--------forward nl description-------\n', n_f)
        else:
            n_f = match_str_ptag(p_context, match_str_block(p_backward, all_blocklis[i - 1])[0]) if p_context.findall(
                match_str_block(p_backward, all_blocklis[i - 1])[0]) != [] else '-10000'
            # print('--------forward nl description-------\n', n_f)

        code = match_tag_code(p_hcode, p_pcode, p_pcode1, p_pcode2, match_str_block(p_code, all_blocklis[i])[0])
        # 对网页标签转义
        c = html.unescape(code)
        # print('-------------context code----------\n', c)

        if i == len(all_blocklis) - 1:
            n_b = match_str_btag(p_context, match_str_block(p_backward, all_blocklis[i])[0]) if p_context.findall(
                match_str_block(p_backward, all_blocklis[i])[0]) != [] else '-10000'
            # print('-----------backward nl description--------\n', n_b)
        else:
            n_b = match_str_ptag(p_context, match_str_block(p_backward, all_blocklis[i])[0]) if p_context.findall(
                match_str_block(p_backward, all_blocklis[i])[0]) != [] else '-10000'
            # print('-----------backward nl description--------\n', n_b)

        # print('########################Block %s#############################\n' % str(i))

        # 模块的分组  (si,code, si+1)
        block_tuple = (n_f, c, n_b)

        block_list.append(block_tuple)

    return block_list


# [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询,   块长度]

def process_xmldata(data_folder, lang_type):
    print('#######################开始执行HNN模型语料%s语言的Block块解析############################' % lang_type)

    hnn_iids_labels = pickle.load(open('./%s_hnn_iids_labeles.pickle' % lang_type, 'rb'))

    # 取（qid,index) ：标签
    # python:4884  sql:3637
    hnn_iids2labels_dict = {i[:2]: i[-1] for i in hnn_iids_labels}

    hnn_iids_lis = sorted(hnn_iids2labels_dict,key=lambda x: (x[0],x[1]))

    # 取帖子的qid列表
    hnn_qids = set([i[0] for i in hnn_iids_labels])
    # python：2052  sql:1587
    print('hnn中查询id索引的长度-%d' % len(hnn_qids))

    # 先到2017xml继续找
    # 从小到大的序号排序
    sort_qids = sorted(hnn_qids)

    # linux执行命令
    eping_str = '|'.join(['(%s)' % str(s) for s in sort_qids])
    # 每份xml命名
    epart_xml = '%s_hnn_map_2017.xml'%lang_type
    # 命令字符串 （不能先限定语言标签，直接在posts.xml中提取）
    eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str,data_folder+'posts_2017.xml', data_folder+epart_xml)
    # 执行命令符
    os.popen(eping_cmd).read()

    # 帖子id
    qidnum_lis = []

    # 接受id
    accept_lis = []

    # 查询
    query_dict = {}

    # 计数
    count1 = 0
    count2 = 0

    with open(data_folder + '%s_hnn_map_2017.xml' % lang_type, 'r', encoding='utf-8') as f:
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

    # accept从小到大的序号排序
    sort_accept_lis = sorted(accept_lis)

    print('2017XML的hnn中原始的qid长度-%s' % len(hnn_qids))
    print('2017XML的文件可解析回复qid长度-%s' % len(qidnum_lis))

    nofind_qids = set(hnn_qids).difference(set(qidnum_lis))

    # python:0, sql:1
    print('2017XML的解析回复不在的qid长度-%s\n' % len(nofind_qids), nofind_qids)

    ###############################################找寻相关的XML文件#########################################

    # 索引的长度 #
    accept_num = len(sort_accept_lis)  # python：2052,sql：1586
    print('2017XML回复答案aid的原始索引长度-%d' % accept_num)

    # linux执行命令
    eping_str = '|'.join(['(%s)' % str(s) for s in sort_accept_lis])
    # 每份xml命名
    epart_xml = '%s_hnn_ans_2017.xml' %lang_type
    # 命令字符串
    eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str, data_folder + 'answer_2017.xml', data_folder + epart_xml)
    # 执行命令符
    os.popen(eping_cmd).read()

    ###################################解析具有全部accept_lis索引的ans_xml文件#######################################
    # 序号id对应accept的id  qid：accept_id (aid)
    q2aids_dict = {}
    for i in query_dict.keys():
        q2aids_dict[i[0]] = i[1]

    a2qids_dict = {}
    for i in query_dict.keys():
        a2qids_dict[i[1]] = i[0]

    with open(data_folder + '%s_hnn_ans_2017.xml' % lang_type, 'r', encoding='utf-8') as f:
        line_dict = {}
        for line in f:
            sta = line.find("\"")
            end = line.find("\"", sta + 1)
            # 提取line对应的accept id
            aid = int(line[(sta + 1):end])
            line_dict[aid] = line
        # 对查询的键值和accept键值进行合并
        same_aids = (set(accept_lis) & set(line_dict.keys()))
        print('2017XML回复答案aid合并的有效长度-%d' % len(same_aids))
        # 对合并后的qid进行排序
        filter_sort_qids = sorted([a2qids_dict[aid] for aid in same_aids])

        print('###################################2017XML回复阶段执行完毕！###################################')

        # 不包含社交属性语料
        corpus_lis = []

        # python:     sql:
        for qid in filter_sort_qids:
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
                # [(id, index),[[si，si+1]] 文本块，[[c]] 代码，[q] 查询,  块长度， 标签]
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
                    corpus_lis.append(block_addlis)

        corpus_lis = [i for i in corpus_lis if i[0] in hnn_iids_lis]

        filter_iids = [i[0] for i in corpus_lis]

        nofind_iids = set(hnn_iids2labels_dict.keys()).difference(set(filter_iids))

        print('html解析不到的qid-index长度%s\n' % len(nofind_iids), nofind_iids)  # python:0, sql:1

        noblock_qids = set([i[0] for i in nofind_iids])
        print('hnn中解析不对qid索引的长度-%d' % len(noblock_qids),noblock_qids)

        # 只要一个qid找不到就需要删除此帖子
        corpus_lis = [i for i in corpus_lis if i[0][0] not in noblock_qids]

        # 存储qid在2017XML/2018XML中
        qidsfind_inxml = {}

        qids_in2017xml= set([i[0][0] for i in corpus_lis])

        qidsfind_inxml[2017] = sorted(qids_in2017xml)

        # 再到2018xml继续找
        # 从小到大的序号排序
        sort_noblock_qids = sorted(noblock_qids)

        # linux执行命令
        eping_str = '|'.join(['(%s)' % str(s) for s in sort_noblock_qids])
        # 每份xml命名
        epart_xml = '%s_hnn_map_2018.xml' % lang_type
        # 命令字符串 （不能先限定语言标签，直接在posts.xml中提取）
        eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str, data_folder + 'posts_2018.xml', data_folder + epart_xml)
        # 执行命令符
        os.popen(eping_cmd).read()

        # 帖子id
        new_qidnum_lis = []

        # 接受id
        new_accept_lis = []

        # 计数
        count1 = 0
        count2 = 0

        with open(data_folder + '%s_hnn_map_2018.xml' % lang_type, 'r', encoding='utf-8') as f:
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
                        query_dict[(int(qidnum), int(accept))] =query

        print('2018XML查询阶段完全遍历处理-%d条数据' % count1)
        print('2018XML查询阶段有效总计处理-%d条数据' % count2)

        print('###################################2018XML查询阶段执行完毕！###################################')

        # accept从小到大的序号排序
        new_sort_accept_lis = sorted(new_accept_lis)

        print('2018XML的hnn中需要补充qid长度-%s' % len(noblock_qids))
        print('2018XML的文件可解析回复qid长度-%s' % len(new_qidnum_lis))

        new_nofind_qids = set(noblock_qids).difference(set(new_qidnum_lis))

        print('2018XML的解析回复不在的qid长度-%s\n' % len(new_nofind_qids), new_nofind_qids)

        ###############################################找寻相关的XML文件#########################################
        # 索引的长度
        new_accept_num = len(new_sort_accept_lis)
        print('2018XML回复答案aid需要的补充长度-%d' % new_accept_num)

        # linux执行命令
        eping_str = '|'.join(['(%s)' % str(s) for s in new_sort_accept_lis])
        # 每份xml命名
        epart_xml = '%s_hnn_ans_2018.xml' % lang_type
        # 命令字符串
        eping_cmd = 'egrep \'<row Id="(%s)"\' %s > %s' % (eping_str, data_folder + 'answer_2018.xml', data_folder + epart_xml)
        # 执行命令符
        os.popen(eping_cmd).read()

        ###################################解析具有全部accept_lis索引的ans_xml文件#######################################
        # 序号id对应accept的id  qid：accept_id (aid)
        for i in query_dict.keys():
            q2aids_dict[i[0]] = i[1]

        for i in query_dict.keys():
            a2qids_dict[i[1]] = i[0]

        with open(data_folder + '%s_hnn_ans_2018.xml' % lang_type, 'r', encoding='utf-8') as f:
            canfind_aids = []
            for line in f:
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
            new_filter_sort_qids = sorted([a2qids_dict[aid] for aid in new_same_aids])

            print('###################################2018XML回复阶段执行完毕！###################################')

            # python:     sql:
            for qid in new_filter_sort_qids:
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
                    # [(id, index),[[si，si+1]] 文本块，[[c]] 代码，[q] 查询,  块长度， 标签]
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
                        corpus_lis.append(block_addlis)

        corpus_lis = [i for i in corpus_lis if i[0] in hnn_iids_lis]

        new_filter_iids = [i[0] for i in corpus_lis]

        new_nofind_iids = set(hnn_iids_lis).difference(set(new_filter_iids))

        print('html再次解析不到的qid-index长度-%s\n' % len(new_nofind_iids), new_nofind_iids)

        new_noblock_qids = set([i[0] for i in new_nofind_iids])
        print('hnn中再次解析不对qid索引的长度-%d' % len(new_noblock_qids),new_noblock_qids)

        # 只要一个qid找不到就需要删除此帖子
        corpus_lis = [i for i in corpus_lis if i[0][0] not in new_noblock_qids]

        find_htmls = {
         (33137311, 33487726): ["How to model several tournaments / brackets types into a SQL database?",
        "<html><body><p>You could create tables to hold tournament types, league types, playoff types, and have a schedule table, showing an even name along with its tournament type, and then use that relationship to retrieve information about that tournament. Note, this is not MySQL, this is more generic SQL language:</p>\n\n<pre><code>CREATE TABLE tournTypes (\nID int autoincrement primary key,\nleagueId int constraint foreign key references leagueTypes.ID,\nplayoffId int constraint foreign key references playoffTypes.ID\n--...other attributes would necessitate more tables\n)\n\nCREATE TABLE leagueTypes(\nID int autoincrement primary key,\nnoOfTeams int,\nnoOfDivisions int,\ninterDivPlay bit -- e.g. a flag indicating if teams in different divisions would play\n)\n\nCREATE TABLE playoffTypes(\nID int autoincrement primary key,\nnoOfTeams int,\nisDoubleElim bit -- e.g. flag if it is double elimination\n)\n\nCREATE TABLE Schedule(\nID int autoincrement primary key,\nName text,\nstartDate datetime,\nendDate datetime,\ntournId int constraint foreign key references tournTypes.ID\n)\n</code></pre>\n\n<p>Populating the tables...</p>\n\n<pre><code>INSERT INTO tournTypes VALUES\n(1,2),\n(1,3),\n(2,3),\n(3,1)\n\nINSERT INTO leagueTypes VALUES\n(16,2,0), -- 16 teams, 2 divisions, teams only play within own division\n(8,1,0),\n(28,4,1)\n\nINSERT INTO playoffTypes VALUES\n(8,0), -- 8 teams, single elimination\n(4,0),\n(8,1)\n\nINSERT INTO Schedule VALUES\n('Champions league','2015-12-10','2016-02-10',1),\n('Rec league','2015-11-30','2016-03-04-,2)\n</code></pre>\n\n<p>Getting info on a tournament...</p>\n\n<pre><code>SELECT Name\n,startDate\n,endDate\n,l.noOfTeams as LeagueSize\n,p.noOfTeams as PlayoffTeams\n,case p.doubleElim when 0 then 'Single' when 1 then 'Double' end as Elimination\nFROM Schedule s\nINNER JOIN tournTypes t\nON s.tournId = t.ID\nINNER JOIN leagueTypes l\nON t.leagueId = l.ID\nINNER JOIN playoffTypes p\nON t.playoffId = p.ID\n</code></pre></body></html>"]}

        new_html_qids = [i[0] for i in find_htmls.keys()]

        if lang_type=='sql':
            for i in find_htmls.keys():
                q2aids_dict[i[0]] = i[1]
            # 排除这些网页
            corpus_lis = [i for i in corpus_lis if i[0][0] not in new_html_qids]
            # 先判断不为空
            for qid in new_html_qids:
                # 获取查询
                query = find_htmls[(qid, q2aids_dict[qid])][0]
                # 每行数据
                line =  find_htmls[(qid, q2aids_dict[qid])][1]
                htmlSoup = BeautifulSoup(line, 'lxml')

                # 抽取代码和上下文 直接用soup（不能转了str)
                block_list = parse_tag_block(htmlSoup)

                if block_list != []:
                    # [(id, index),[[si，si+1]] 文本块，[[c]] 代码，[q] 查询,  块长度， 标签]
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
                        corpus_lis.append(block_addlis)

        corpus_lis = [i for i in corpus_lis if i[0] in hnn_iids_lis]

        final_filter_iids = [i[0] for i in corpus_lis]

        final_nofind_iids = set(hnn_iids_lis).difference(set(final_filter_iids))

        print('html最终解析不到的qid-index长度-%s\n' % len(final_nofind_iids), final_nofind_iids)

        final_noblock_qids = set([i[0] for i in final_nofind_iids])
        print('hnn中解析不对qid索引的长度-%d' % len(final_noblock_qids),final_noblock_qids)

        # 只要一个qid找不到就需要删除此帖子
        corpus_lis = [i for i in corpus_lis if i[0][0] not in final_noblock_qids]

        final_filter_qids= set([i[0][0] for i in corpus_lis ])

        qids_in2018xml = set(final_filter_qids).difference(set(qids_in2017xml))

        qidsfind_inxml[2018] = sorted(qids_in2018xml)

        pickle.dump(qidsfind_inxml,open('./%s_hnn_qidsfind_inxml.pickle' % lang_type, 'wb'))

        # [(id, index),[[si，si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度, 标签]

        # 增加标签
        [i.append(hnn_iids2labels_dict[i[0]]) for i in corpus_lis]
        print('hnn需要匹配qid-index原始长度-%s' % len(hnn_iids2labels_dict.keys()))
        print('html能标注qid-index正确长度-%s' % len(corpus_lis))

        # 从小到大排序
        sort_corpus_lis = sorted(corpus_lis, key=lambda x: (x[0][0], x[0][1]))

        with open('./%s_hnn_qid2index_blocks_labeled.txt' % lang_type, "w") as f:
            #  存储为字典 有序
            f.write(str(sort_corpus_lis))
            f.close()






# 参数配置设置
data_folder = '../corpus_xml/'

sqlang_type = 'sql'
python_type = 'python'


if __name__ == '__main__':
    # 抽取xml
    #extract_differ_xml(data_folder, 'posts_2017.xml','answer_2017.xml')
    #extract_differ_xml(data_folder, 'posts_2018.xml','answer_2018.xml')

    # 处理python
    process_xmldata(data_folder, python_type)
    # 处理sql
    process_xmldata(data_folder, sqlang_type)
