# -*- coding: utf-8 -*-

import pickle
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


def pickle_tocsv(lang_type,save_path):

    mutiple_query = pickle.load(open('%s_multiple_qid_to_title.pickle'%lang_type, 'rb'))
    mutiple_code  = pickle.load(open('%s_multiple_iid_to_code.pickle'%lang_type, 'rb'))

    s0=list(mutiple_query.keys())[:2]
    s1=list(mutiple_query.items())[:2]
    s2=list(mutiple_code.items())[:2]

    print('%s多候选的qid-index长度为%s'%(lang_type,len(mutiple_code)))
    ##############################################打印测试######################################################
    #[31588354, 12976131]#
    print(s0,'\n')
    #[(31588354, u'Identical messages committed during a network partition'),  (12976131, u'Dynamic use of T-SQL schemas')]#
    print(s1,'\n')
    #[((26412603, 0), u"select  year, college,  case\n  when tuition < 99999.99 then 'Other'\n   else to_char(tuition)\n    end as tuition, count(*) as tuition_count\nfrom enrolment\ngroup by year, college, case\n    when tuition < 99999.99 then 'Other'\n  else to_char(tuition)\n  end\norder by year, college, case\n  when tuition < 99999.99 then 'Other'\n    else to_char(tuition)\n  end\n"),  ((6609887, 1), u'SELECT CASE WHEN y_type IN (10,20) THEN y ELSE NULL END As y -- replace y_type with column to check\n,.... -- other column list\nFROM yourtable\n')]#
    print(s2,'\n')
    ##############################################打印测试######################################################

    mutiple_qid_list  = []
    mutiple_query_list = []
    mutiple_code_list  = []
    mutiple_cid_list = []

    for id in list(mutiple_query.keys()):
    # [((31588354, 1), u"// distributed leases/leader election\nwhile (true) {\n  locked_until = read(key)\n  if (locked_until < now()) {\n    if (compare_and_set(key, locked_until, now() + TEN_MINUTES)) {\n      // I'm now the leader for ~10 minutes.\n      return;\n    }\n  }\n  sleep(TEN_MINUTES)\n}\n"),    ((31588354, 0), u'// generating a unique id\nwhile (true) {\n  unique_id = read(key)\n  if (compare_and_set(key, unique_id, unique_id + 1)) {\n    return unique_id\n  }\n}\n')]#
        cor = [i for i in list(mutiple_code.items()) if i[0][0]==id]
        #[((30146796, 0), 'album_id artist_id artist_order\n'), ((30146796, 1), 'album_id  artist_id artist_order\n  312   87  1\n  312   33   2\n  312   50 3\n')]
        cor = sorted(cor,key=lambda x:x[0][1])
        i =0
        while i < len(cor):
            mutiple_query_list.append(filter_inva_char(mutiple_query[id]))
            mutiple_qid_list.append(id)
            i+=1
        for term in cor:
            #((22850277, 1), 'select\n shift, sum(current_workers)/count(distinct week_number) avg_workers\nfrom\n work_sheet t\ngroup by\n  shift\n')#
            mutiple_cid_list.append(term[0][1])
            code = filter_inva_char(term[1])
            mutiple_code_list.append(code)

    print('多候选代码转换完毕！')

    #  查询索引
    qid_list  = mutiple_qid_list
    #  代码索引
    cid_list  = mutiple_cid_list
    #  查询字符
    query_list= mutiple_query_list
    #  代码字符
    code_list = mutiple_code_list

    data_dict = {'ID':qid_list,'code_rank':cid_list,'orgin_query':query_list,'orgin_code':code_list}

    mutiple_data = pd.DataFrame(data_dict,columns=['ID','orgin_query','orgin_code','code_rank'])

    # 单引号置换双引号 便于解析 (主要是代码）
    for name in ['orgin_query', 'orgin_code']:
        mutiple_data[name] = mutiple_data[name].str.replace('\'', '\"')

    mutiple_data.to_csv(save_path+'mutiple_alldb_%s.csv'%lang_type,encoding='utf-8',sep='|')

    print('\n%s数据类型转换完毕！！'%lang_type)



sql_type = 'sql'
python_type = 'python'


save_path = '../../code_search/use_data/'


if __name__ == '__main__':
    pickle_tocsv(sql_type,save_path)
    pickle_tocsv(python_type,save_path)







