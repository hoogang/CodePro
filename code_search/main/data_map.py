
import os
import re
import pandas as pd


def process_logdata(log_dir):

    path_file=os.listdir(log_dir)

    for txt_name in path_file:

        searchResults= re.search(r'model-(.*)-(.*)-(.*)-(.*)-2020.*.txt',txt_name)

        print(searchResults)
        # 候选类型
        candi_type = searchResults.group(1)
        print(candi_type)
        # 数据类型
        data_type =  searchResults.group(2)
        print(data_type)
        # 语言类型
        lang_type =  searchResults.group(3)
        print(lang_type)
        # id索引
        model_id = int(searchResults.group(4))
        print(model_id)

        epoch = []

        valid_loss = []
        valid_recall1 = []

        valid_mrr = []

        test_loss = []
        test_recall1 = []

        test_mrr = []

        # 打开txt文件
        with open (log_dir+txt_name,'r',encoding='utf-8') as fin:

            while True:
                line=fin.readline()
                # 如果不存在跳出循环
                if not line:
                    break
                #  抽取epoch和valid Loss的值
                Valid = re.search(r'.*epoch:(.*) -- valid Loss:(.*)',line)

                #  抽取指标的值
                Valid_Eval = re.search(r'.*valid recall@1:(.*) -- valid mrr:(.*)',line)

                #  抽取epoch和test Loss的值
                Test  = re.search(r'.*epoch:(.*) -- test Loss:(.*)', line)
                # 抽取指标的值
                Test_Eval = re.search(r'.*test recall@1:(.*) -- test mrr:(.*)', line)

                # 存入list
                if Valid:
                    epoch.append(int(Valid.group(1)))
                    valid_loss.append(float(Valid.group(2)))

                # 存入list
                if Valid_Eval:
                    valid_recall1.append(float(Valid_Eval.group(1)))
                    valid_mrr.append(float(Valid_Eval.group(2)))

                # 存入list
                if Test:
                    test_loss.append(float(Test.group(2)))

                # 存入list
                if Test_Eval:
                    test_recall1.append(float(Test_Eval.group(1)))
                    test_mrr.append(float(Test_Eval.group(2)))


        data_dict = {'epoch': epoch, 'valid_Loss': valid_loss, 'test_Loss':test_loss, 'valid_recall@1':valid_recall1,
                     'valid_mrr': valid_mrr ,'test_recall@1':test_recall1, 'test_mrr':test_mrr}
        # 索引列表
        column_list = ['epoch','valid_Loss','test_Loss','valid_recall@1','valid_mrr','test_recall@1','test_mrr']
        # 重建列索引
        result_data = pd.DataFrame(data_dict,columns=column_list)
        # 保存数据
        result_data.to_csv(log_dir+'model_%s_%s_%s_%d.csv'%(candi_type,data_type,lang_type,model_id),encoding='utf-8', sep='|')


log_dir= './log/'


if __name__ == '__main__':
    process_logdata(log_dir)
