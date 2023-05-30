import os
import argparse
import numpy as np

from saihnn_tolabel import C2labelbySaiHnn
from textsa_tolabel import C2labelbyTextSa
from codesa_tolabel import C2labelbyCodeSa

import sys

sys.path.append("..")
sys.path.append("../label_models/")

from label_models import configs

from label_models.model_saihnn import SaiHnn
from label_models.model_textsa import TextSa
from label_models.model_codesa import CodeSa

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

import warnings
warnings.filterwarnings("ignore")


 # 不同模型打标签的结果
def final_analays_staqc(label_path, human_path, final_path):
    with open(label_path, 'r') as f:
        pre = eval(f.read())
        f.close()
    ids = pre[0]
    codemf_lable = pre[1]
    textsa_lable = pre[2]
    codesa_lable = pre[3]
    human_lable_1 = []
    with open(human_path, 'r') as f:
        human = eval(f.read())
        f.close()
    for i in range(0, len(human[0])):
        if (human[1][i] == 1):
            human_lable_1.append(human[0][i])

    total_final = []
    count = 0
    for i in range(0, len(ids)):
        if (codesa_lable[i] == 1 and textsa_lable[i] == 1 and codemf_lable[i] == 1):
            if ids[i] in human[0]:
                continue
            else:
                total_final.append(ids[i])
                count += 1
    total_final = total_final + human_lable_1
    f = open(final_path, "w")
    print(len(total_final))
    for i in range(0, len(total_final)):
        f.writelines(str(total_final[i]))
        f.writelines('\n')
    f.close()


# 在最终标签语料中，找到human中的语料，替换成human中标签

def final_analays_large(label_path, human_path, single_path, save_path):
    with open(label_path, 'r') as f:
        pre = eval(f.read())
        f.close()
    ids = pre[0]
    codemf_lable = pre[1]
    textsa_lable = pre[2]
    codesa_lable = pre[3]
    human_lable_1 = []
    with open(human_path, 'r') as f:
        human = eval(f.read())
        f.close()
    for i in range(0, len(human[0])):
        if (human[1][i] == 1):
            human_lable_1.append(human[0][i])

    total_final = []
    count = 0
    for i in range(0, len(ids)):
        if (codesa_lable[i] == 1 and textsa_lable[i] == 1 and codemf_lable[i] == 1):
            if ids[i] in human[0]:
                continue
            else:
                total_final.append(ids[i])
                count += 1
    with open(single_path, 'r') as f:
        single = eval(f.read())
        f.close()
    single_ids = []
    for i in range(0, len(single)):
        single_ids.append(single[i][0])
    print(len(single_ids))
    # total_final = total_final+human_lable_1+single_ids
    total_final = total_final + human_lable_1
    print(count)

    f = open(save_path, "w")
    print(total_final[1])
    for i in range(0, len(total_final)):
        f.writelines(str(total_final[i]))
        f.writelines('\n')
    f.close()


def parse_args():
    # 创建对象
    parser = argparse.ArgumentParser("Train and Test Model")
    # 语言类型选择
    parser.add_argument("--lang_type", type=str, default="sql",help="dataset set")

    # 数据处理位置
    parser.add_argument("--data_path", type=str, default="../hqfso_tool/corpus", help="format path")
    # 人工标注位置
    parser.add_argument("--human_path", type=str, default="../data_hnn/", help="label path")

    # saihuman模型位置
    parser.add_argument("--saihnn_path", type=str, default="../label_params/new/bivhnn/", help="label path")
    # textsa模型位置
    parser.add_argument("--textsa_path", type=str, default="../label_params/new/textsa/", help="label path")
    # codesa模型位置
    parser.add_argument("--codesa_path", type=str, default="../label_params/new/codesa/", help="label path")

    # 其他类型参数
    parser.add_argument("--verbose", action="store_true", default=True, help="Be verbose")

    return parser.parse_args()



def staqc_tolabel(format_name,mutiple_name,final_name):

    args = parse_args()

    # 格式数据
    format_data = args.data_path + '/format_tolabel/' + format_name
    # 标注数据
    mutiple_label =  args.data_path + '/' + mutiple_name
    # 最终标注
    final_label = args.data_path + '/' + final_name

    human_name = '%s_hnn_labeled_by_human.txt' % args.lang_type
    human_label = args.human_path + human_name

    vocab_path = args.data_path + '/embeddings/'


    drop1 = drop2 = drop3 = drop4 = drop5 = np.round(0.25, 2)

    ##############################################

    saihnn_params = args.saihnn_path + args.lang_type
    saihnn_confs = configs.get_config_u2l(args.lang_type, saihnn_params, vocab_path)

    ##### Define model ######
    logger.info('Build SAIHNN Model')

    saihnn_model = eval('SaiHnn')(saihnn_confs)

    saihnn_c2label = C2labelbySaiHnn(saihnn_confs)

    saihnn_model.params_adjust(dropout1=drop1, dropout2=drop2, dropout3=drop3, dropout4=drop4, dropout5=drop5,
                                 regularizer=round(0.0004, 4), seed=42)
    saihnn_model.build()

    if args.lang_type == 'python':
        saihnn_c2label.load_model(saihnn_model, 1166, 0.5, 0.45, 0.55, 0.45, 0.0006)
        saihnn_c2label.u2l_saihnn(saihnn_model, format_data, mutiple_label)

    if args.lang_type == 'sql':
        saihnn_c2label.load_model(saihnn_model, 86, 0.25, 0.25, 0.25, 0.25, 0.0004)
        saihnn_c2label.u2l_saihnn(saihnn_model, format_data, mutiple_label)

    ##############################################

    textsa_params = args.textsa_path + args.lang_type
    textsa_confs = configs.get_config_u2l(args.lang_type, textsa_params, vocab_path)

    ##### Define model ######
    logger.info('Build TEXTSA Model')

    textsa_model = eval('TextSa')(textsa_confs)

    textsa_c2label = C2labelbyTextSa(textsa_confs)

    textsa_model.params_adjust(dropout1=drop1, dropout2=drop2, dropout3=drop3, dropout4=drop4, dropout5=drop5,
                               regularizer=round(0.0004, 4), seed=42)
    textsa_model.build()

    if args.lang_type == 'python':
        textsa_c2label.load_model(textsa_model, 1079, 0.5, 0.5, 0.5, 0.5, 1.0002)
        textsa_c2label.u2l_textsa(textsa_model, format_data, mutiple_label)

    if args.lang_type == 'sql':
        textsa_c2label.load_model(textsa_model, 1033, 0.1, 0.1, 0.1, 0.1, 1.0002)
        textsa_c2label.u2l_textsa(textsa_model, format_data, mutiple_label)

    ####################### #######################

    codesa_params = args.codesa_path + args.lang_type
    codesa_confs = configs.get_config_u2l(args.lang_type, codesa_params, vocab_path)

    ##### Define model ######
    logger.info('Build CODESA Model')

    codesa_model = eval('CodeSa')(codesa_confs)

    codesa_c2label = C2labelbyCodeSa(codesa_confs)

    codesa_model.params_adjust(dropout1=drop1, dropout2=drop2, dropout3=drop3, dropout4=drop4, dropout5=drop5,
                               regularizer=round(0.0004, 4), seed=42)
    codesa_model.build()

    if args.lang_type == 'python':
        codesa_c2label.load_model(codesa_model, 138, 0.15, 0.15, 0.15, 0.15, 101)
        codesa_c2label.u2l_codesa(codesa_model, format_data, mutiple_label)

    if args.lang_type == 'sql':
        codesa_c2label.load_model(codesa_model, 1111, 0.1, 0.1, 0.1, 0.1, 101)
        codesa_c2label.u2l_codesa(codesa_model, format_data, mutiple_label)

    final_analays_staqc(mutiple_label, human_label, final_label)



def large_tolabel(format_name,mutiple_name,single_name,final_name):

    args = parse_args()

    # 格式数据
    format_data = args.data_path + '/format_tolabel/' + format_name
    # 标注数据
    mutiple_label =  args.data_path + '/' + mutiple_name
    # 单选标注
    single_label =  args.data_path + '/' + single_name
    # 最终标注
    final_label =  args.data_path + '/' + final_name

    human_name = '%s_hnn_labeled_by_human.txt' % args.lang_type
    human_label = args.human_path + human_name

    vocab_path = args.data_path + '/embeddings/'


    drop1 = drop2 = drop3 = drop4 = drop5 = np.round(0.25, 2)

    ##############################################

    saihnn_params = args.saihnn_path + args.lang_type
    saihnn_confs = configs.get_config_u2l(args.lang_type, saihnn_params, vocab_path)

    ##### Define model ######
    logger.info('Build SAIHNN Model')

    saihnn_model = eval('SaiHnn')(saihnn_confs)

    saihnn_c2label = C2labelbySaiHnn(saihnn_confs)

    saihnn_model.params_adjust(dropout1=drop1, dropout2=drop2, dropout3=drop3, dropout4=drop4, dropout5=drop5,regularizer=round(0.0004, 4), seed=42)
    saihnn_model.build()

    if args.lang_type == 'python':
        saihnn_c2label.load_model(saihnn_model, 1166, 0.5, 0.45, 0.55, 0.45, 0.0006)
        saihnn_c2label.u2l_saihnn(saihnn_model, format_data, mutiple_label)

    if args.lang_type == 'sql':
        saihnn_c2label.load_model(saihnn_model, 86, 0.25, 0.25, 0.25, 0.25, 0.0004)
        saihnn_c2label.u2l_saihnn(saihnn_model, format_data, mutiple_label)

    ##############################################

    textsa_params = args.textsa_path + args.lang_type
    textsa_confs = configs.get_config_u2l(args.lang_type, textsa_params, vocab_path)

    ##### Define model ######
    logger.info('Build TEXTSA Model')

    textsa_model = eval('TextSa')(textsa_confs)

    textsa_c2label = C2labelbyTextSa(textsa_confs)

    textsa_model.params_adjust(dropout1=drop1, dropout2=drop2, dropout3=drop3, dropout4=drop4, dropout5=drop5,
                               regularizer=round(0.0004, 4), seed=42)
    textsa_model.build()

    if args.lang_type == 'python':
        textsa_c2label.load_model(textsa_model, 1079, 0.5, 0.5, 0.5, 0.5, 1.0002)
        textsa_c2label.u2l_textsa(textsa_model, format_data, mutiple_label)

    if args.lang_type == 'sql':
        textsa_c2label.load_model(textsa_model, 1033, 0.1, 0.1, 0.1, 0.1, 1.0002)
        textsa_c2label.u2l_textsa(textsa_model, format_data, mutiple_label)

    ####################### #######################

    codesa_params = args.codesa_path + args.lang_type
    codesa_confs = configs.get_config_u2l(args.lang_type, codesa_params, vocab_path)

    ##### Define model ######
    logger.info('Build CODESA Model')

    codesa_model = eval('CodeSa')(codesa_confs)

    codesa_c2label = C2labelbyCodeSa(codesa_confs)

    codesa_model.params_adjust(dropout1=drop1, dropout2=drop2, dropout3=drop3, dropout4=drop4, dropout5=drop5,
                               regularizer=round(0.0004, 4), seed=42)
    codesa_model.build()

    if args.lang_type == 'python':
        codesa_c2label.load_model(codesa_model, 138, 0.15, 0.15, 0.15, 0.15, 101)
        codesa_c2label.u2l_codesa(codesa_model, format_data, mutiple_label)

    if args.lang_type == 'sql':
        codesa_c2label.load_model(codesa_model, 1111, 0.1, 0.1, 0.1, 0.1, 101)
        codesa_c2label.u2l_codesa(codesa_model, format_data, mutiple_label)

    final_analays_large(mutiple_label, human_label, single_label, final_label)




args = parse_args()

format_name = '%s_corpus_qid2index_blocks_to_multiple_qids_format_tolabel.pkl' % args.lang_type
mutiple_name ='%s_corpus_qid2index_blocks_to_multiple_qids_labeled.txt' % args.lang_type
single_name = '%s_corpus_qid2index_blocks_to_single_qids_labeled.txt' % args.lang_type
final_name =  '%s_corpus_qid2index_blocks_labeled_final.txt' % args.lang_type


# 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# 可用显卡
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


if __name__ == '__main__':
    #staqc_tolabel(format_name,mutiple_name,final_name)
    large_tolabel(format_name,mutiple_name,single_name,final_name)









