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

    single_query= pickle.load(open('%s_single_qid_to_title.pickle'%lang_type, 'rb'))
    single_code = pickle.load(open('%s_single_qid_to_code.pickle'%lang_type, 'rb'))

    s3=list(single_query.keys())[:2]
    s4=list(single_query.items())[:2]
    s5=list(single_code.items())[:2]

    print('%s单候选的qid长度为%s' % (lang_type, len(single_code)))
    ##############################################打印测试######################################################
    #[16384000, 26258091]
    print(s3, '\n')
    #[(16384000, u'select name, COUNT(*) from two different tables as a query result'),  (20097707, u'SQL Variable date minus a year')]
    print(s4,'\n')
    #[(16384000, u'SELECT  cus.cusName,\n  COUNT(*) AS ItemBought\nFROM  Customer AS cus\n  INNER JOIN Orders AS ord\n  ON cus.CusID = ord.cusID\nGROUP   BY cus.cusName\n'), (20097707,  u'@DateSelected datetime=null,\n@year int=null,\n@week int=null,\n@DayOfWeek int=null,\n@DateSelectedLastYear datetime=null\n--@Day int=null\nas\nbegin\n    --if date parms are null get current week and year\n if (@DateSelected is null)\n  begin\n  select @Year=year_number,@Week=week_number,@DayOfWeek=day_of_week from  infrastructure..calendar\n where calendar_date=DATEADD(dd, 0, DATEDIFF(dd, 0, GETDATE()-1))\n  end\n  else\n  begin\n  select @Year=year_number,@Week=week_number,@DayOfWeek=day_of_week from  infrastructure..calendar\n  where calendar_date=@DateSelected\n end\n begin\n  Select\n  @DateSelectedLastYear = DATEADD(YEAR,-1,@DateSelected)\n   end\n')]#
    print(s5,'\n')
    ##############################################打印测试######################################################

    single_qid_list = []
    single_query_list = []
    single_code_list  = []
    single_cid_list = []
    for id in list(single_query.keys()):
        #  只有一个候选候选代码
        cor = [i for i in list(single_code.items()) if i[0]==id]
        single_query_list.append(filter_inva_char(single_query[id]))
        single_qid_list.append(id)
        single_cid_list.append(0)
        code0 = filter_inva_char(cor[0][1])
        single_code_list.append(code0)

    print('单候选代码转换完毕！')

    #  查询索引
    qid_list  = single_qid_list
    #  代码索引
    cid_list  = single_cid_list
    #  查询字符
    query_list= single_query_list
    #  代码字符
    code_list = single_code_list

    data_dict = {'ID':qid_list,'code_rank':cid_list,'orgin_query':query_list,'orgin_code':code_list}

    mutiple_data = pd.DataFrame(data_dict,columns=['ID','orgin_query','orgin_code','code_rank'])

    # 单引号置换双引号 便于解析 (主要是代码）
    for name in ['orgin_query', 'orgin_code']:
        mutiple_data[name] = mutiple_data[name].str.replace('\'', '\"')

    mutiple_data.to_csv(save_path+'single_sicqc_%s.csv'%lang_type,encoding='utf-8',sep='|')

    print('\n%s数据类型转换完毕！！'%lang_type)




sql_type = 'sql'
python_type = 'python'


save_path = '../../code_search/rank_data/'


if __name__ == '__main__':
    pickle_tocsv(sql_type,save_path)
    pickle_tocsv(python_type,save_path)







