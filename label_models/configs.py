

#跑模型运行的配置
def get_config(lang_type,params_path,vocab_path):
    conf = {
        'params_path': params_path,
        'buckets':[(2, 10, 22, 72), (2, 20, 34, 102), (2, 40, 34, 202), (2, 100, 34, 302)],
        'data_params':{

            #train data
            "train_path": '../data_hnn/new_embeddings/%s/hnn_%s_train_struc2vec_embeding_300.pkl'%(lang_type,lang_type),
            #valid data
            'valid_path': '../data_hnn/new_embeddings/%s/hnn_%s_valid_struc2vec_embeding_300.pkl'%(lang_type,lang_type),
            #test data
            'test_path': '../data_hnn/new_embeddings/%s/hnn_%s_test_struc2vec_embeding_300.pkl'%(lang_type,lang_type),

            'code_pretrain_emb_path': '%s/%s_word_vocab_final.pkl'%(vocab_path,lang_type),
            'text_pretrain_emb_path': '%s/%s_word_vocab_final.pkl'%(vocab_path,lang_type)
        },               
        'train_params': {
            'batch_size': 100,
            'nb_epoch':150,
            # 'optimizer':5 'adam',
            #'optimizer': Adam(clip_norm=0.1),

            'n_eval': 100,
            'evaluate_all_threshold': {
                'mode': 'all',
                'top1': 0.4,
            },
            # epoch that the model is reloaded from . If reload=0, 然后从头开始训练
            'reload':0,
            'dropout1': 0,
            'dropout2': 0,
            'dropout3': 0,
            'dropout4': 0,
            'dropout5': 0,
            'regularizer': 0,

        }
    }
    return conf



#打标签运行的配置
def get_config_u2l(lang_type,params_path,vocab_path):
    conf = {
        'params_path': params_path,
        'buckets':[(2, 10, 22, 72), (2, 20, 34, 102), (2, 40, 34, 202), (2, 100, 34, 302)],
        'data_params':{

            # train data
            "train_path": '../data_hnn/new_embeddings/%s/hnn_%s_train_struc2vec_embeding_300.pkl'%(lang_type,lang_type),
            # valid data
            'valid_path': '../data_hnn/new_embeddings/%s/hnn_%s_valid_struc2vec_embeding_300.pkl'%(lang_type,lang_type),
            # test data
            'test_path': '../data_hnn/new_embeddings/%s/hnn_%s_test_struc2vec_embeding_300.pkl'%(lang_type,lang_type),

            #原始Staqc打标签的词向量地址
            #'code_pretrain_emb_path':  '../data_hnn/new_embeddings/%s/%s_word_vocab_final.pkl' % (lang_type,lang_type),
            #'text_pretrain_emb_path':  '../data_hnn/new_embeddings/%s/%s_word_vocab_final.pkl' % (lang_type,lang_type)

            #sql：large语料-打标签的词向量地址
            #'code_pretrain_emb_path': '../istaqc_crawl/corpus/embeddings/sql_word_vocab_final.pkl',
            #'text_pretrain_emb_path': '../istaqc_crawl/corpus/embeddings/sql_word_vocab_final.pkl'

            #Python：large语料-打标签的词向量地址
            #'code_pretrain_emb_path': '../istaqc_crawl/corpus/embeddings/python_word_vocab_final.pkl',
            #'text_pretrain_emb_path': '../istaqc_crawl/corpus/embeddings/python_word_vocab_final.pkl'

            'code_pretrain_emb_path': '%s/%s_word_vocab_final.pkl' % (vocab_path, lang_type),
            'text_pretrain_emb_path': '%s/%s_word_vocab_final.pkl' % (vocab_path, lang_type)

        },
        'train_params': {
            'batch_size': 100,
            'nb_epoch':150,
            #'optimizer':5 'adam',
            #'optimizer': Adam(clip_norm=0.1),

            'n_eval': 100,
            'evaluate_all_threshold': {
                'mode': 'all',
                'top1': 0.4,
            },
            # epoch that the model is reloaded from . If reload=0, 然后从头开始训练
            'reload':0,
            'dropout1': 0,
            'dropout2': 0,
            'dropout3': 0,
            'dropout4': 0,
            'dropout5': 0,
            'regularizer': 0,

        }
    }
    return conf




