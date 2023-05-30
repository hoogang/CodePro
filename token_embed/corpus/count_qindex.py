import pickle


# 读取语料数据集
with open('python_saihnn_corpus_qid2index_blocks_unlabeled.pickle', "rb") as f:
    #  存储为字典 有序
    # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, [qcont] 查询上下文, [acont] 回复上下文, 块长度，标签]
    saihnn_corpus_lis = pickle.load(f)
    print(len(saihnn_corpus_lis))

with open('python_codemf_corpus_qid2index_blocks_labeled.pickle', "rb") as f:
    #  存储为字典 有序
    # # [(id, index), [[c]] 代码，[q] 查询, [qcont] 查询上下文, [acont] 回复上下文, 块长度，标签]
    codemf_corpus_lis = pickle.load(f)
    print(len(codemf_corpus_lis))


# 读取语料数据集
with open('sql_saihnn_corpus_qid2index_blocks_unlabeled.pickle', "rb") as f:
    #  存储为字典 有序
    # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, [qcont] 查询上下文, [acont] 回复上下文, 块长度，标签]
    saihnn_corpus_lis = pickle.load(f)
    print(len(saihnn_corpus_lis))

with open('sql_codemf_corpus_qid2index_blocks_labeled.pickle', "rb") as f:
    #  存储为字典 有序
    # # [(id, index), [[c]] 代码，[q] 查询, [qcont] 查询上下文, [acont] 回复上下文, 块长度，标签]
    codemf_corpus_lis = pickle.load(f)
    print(len(codemf_corpus_lis))