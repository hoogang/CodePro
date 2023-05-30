# -*- coding: utf-8 -*-

import re



import sys
sys.path.append("../../")

# 解析结构
#sqlang解析
from crawl_process.parsers import sqlang_structured

#python解析
from crawl_process.parsers import python_structured

#javang解析
from crawl_process.parsers import javang_structured


def codetag_parse(lang_type,pro_line):

    if pro_line and pro_line != '-100':

        # 匹配任意字符（包括换行符）
        p = r'\[c\]((?!\[c\]|\[/c\])[\s\S])*\[/c\]'

        # 抽取代码标签 re.I(re.IGNORECASE): 忽略大小写
        matchCode = re.finditer(p, pro_line)
        # 代码标签 ['<code>string<code>']
        codeTag = [i.group() for i in matchCode]

        if codeTag != []:  # 有代码标签
            #############纯代码处理##########
            p = re.compile(r'\[c\]([\s\S]*?)\[/c\]')
            codeStr = [p.findall(i)[0] for i in codeTag]

            # 判断代码类型
            codeTok = [['-100'] if j == '-1000' else (python_structured.python_code_parse(j)
                    if python_structured.python_code_parse(j) != '-1000' and python_structured.python_code_parse(j) !=
                    [] else ['-100'])  for j in codeStr] if lang_type == 'python' else (
                    [['-100'] if j == '-1000' else (sqlang_structured.sqlang_code_parse(j)
                    if sqlang_structured.sqlang_code_parse(j) != '-1000' and sqlang_structured.sqlang_code_parse(j) !=
                    [] else ['-100']) for j in codeStr]  if lang_type == 'sql' else
                    [['-100'] if j == '-1000' else (javang_structured.javang_code_parse(j)
                    if javang_structured.javang_code_parse(j) != '-1000' and javang_structured.javang_code_parse(j) !=
                    [] else ['-100']) for j in codeStr] )

            # 合并代码令牌
            cTokStr = [' '.join(tok) for tok in codeTok]
            #############纯代码处理##########

            ############纯文本处理############
            # 文本替换
            for n in  range(len(codeTag)):
                pro_line = re.sub(p, 'ctag_%d_ctag'%(n+1),pro_line,count=1)
            # 解析查询
            cutWords = python_structured.python_context_parse(pro_line) if lang_type == 'python' else \
                        (sqlang_structured.sqlang_context_parse(pro_line) if lang_type == 'sql'
                        else javang_structured.javang_context_parse(pro_line))
            # 分词拼接
            textCuter = ' '.join(cutWords)
            ############纯文本处理############

            # 文本代码重组
            for (c, n) in zip(cTokStr, range(len(cTokStr))):
                textCuter = textCuter.replace('ctag_%d_ctag'%(n+1), c)
            # 分词 保留符号和单纯数字
            tokens = textCuter.split(' ')

            return tokens

        else: #无代码标签
              #解析查询文本
            tokens = python_structured.python_context_parse(pro_line) if lang_type == 'python' else \
                    (sqlang_structured.sqlang_context_parse(pro_line) if lang_type == 'sql' else
                     javang_structured.javang_context_parse(pro_line))

            return tokens
    else:
        return ['-100']


def querytag_parse(lang_type, pro_line):

    if pro_line and pro_line != '-100':

        # 匹配任意字符（包括换行符）
        p = r'\[q\]((?!\[q\]|\[/q\])[\s\S])*\[/q\]'
        # 抽取代码标签
        matchCode = re.finditer(p, pro_line)
        # 代码标签 ['<code>string<code>']
        queryTag = [i.group() for i in matchCode]

        if queryTag != []:  # 有代码标签
            #############纯查询处理##########
            p = re.compile(r'\[q\]([\s\S]*?)\[/q\]')
            queryStr = [p.findall(i)[0] for i in queryTag]

            # 判断查询类型
            queryTok = [python_structured.python_context_parse(j) if python_structured.python_context_parse(j) != []
                        else ['-100'] for j in queryStr] if lang_type == 'python' else\
                       ([sqlang_structured.sqlang_context_parse(j) if sqlang_structured.sqlang_context_parse(j) != []
                        else ['-100'] for j in queryStr] if lang_type == 'sql' else
                       [javang_structured.javang_context_parse(j) if javang_structured.javang_context_parse(j) != []
                        else ['-100'] for j in queryStr])

            # 合并查询词语
            qTokStr = [' '.join(tok) for tok in queryTok]
            #############纯查询处理##########

            ############纯程序处理############
            # 文本替换
            for i in range(len(queryTag)):
                pro_line = re.sub(p, 'qtag_%d_qtag' %(i+1), pro_line, count=1)
            # 解析代码
            cutTokens = python_structured.python_code_parse(pro_line) if python_structured.python_code_parse(
                        pro_line) != '-1000' and python_structured.python_code_parse(
                        pro_line) != [] else ['-100'] \
                        if lang_type == 'python' else (
                        sqlang_structured.sqlang_code_parse(pro_line) if sqlang_structured.sqlang_code_parse(
                        pro_line) != '-1000' and sqlang_structured.sqlang_code_parse(
                        pro_line) != [] else ['-100'] \
                        if lang_type == 'sql' else
                        javang_structured.javang_code_parse(pro_line) if javang_structured.javang_code_parse(
                        pro_line) != '-1000' and javang_structured.javang_code_parse(
                        pro_line) != [] else ['-100'])
            # 分词拼接
            textCuter = ' '.join(cutTokens)
            ############纯程序处理############

            # 文本代码重组
            for (q, n) in zip(qTokStr, range(len(qTokStr))):
                textCuter = textCuter.replace('qtag_%d_qtag' % (n+1), q)
            # 分词 保留符号和单纯数字
            tokens = textCuter.split(' ')

            return tokens

        else:#无代码标签
             #解析查询文本
            tokens = python_structured.python_context_parse(pro_line) if lang_type == 'python' \
                     else (sqlang_structured.sqlang_context_parse(pro_line) if lang_type == 'sql'
                     else javang_structured.javang_context_parse(pro_line))

            return tokens
    else:
        return ['-100']



#程序分词测试
if __name__ == '__main__':

    print(codetag_parse('python','[c]groups = []\nuniquekeys = []\nfor k, g in groupby(data, keyfunc):\n groups.append(list(g)) # Store group iterator as a list\n uniquekeys.append(k)[/c] [c]-1000[/c] [c]-1000[/c] [c]-1000[/c] [c]from itertools import groupby\nthings = [("animal", "bear"), ("animal", "duck"), ("plant", "cactus"), ("vehicle", "speed boat"), ("vehicle", "school bus")]\nfor key, group in groupby(things, lambda x: x[0]):\n for thing in group:\n print "A %s is a %s." % (thing[1], key)\n print " "[/c] [c]-1000[/c] [c]-1000[/c] [c]lambda x: x[0][/c] [c]-1000[/c] [c]-1000[/c] [c]-1000[/c] [c]for key, group in groupby(things, lambda x: x[0]):\n listOfThings = " and ".join([thing[1] for thing in group])\n print key + "s: " + listOfThings + "."[/c]'))

    print(codetag_parse('python','[c]d = {"x": 1, "y": 2, "z": 3} \n for key in d: \n print (key, "corresponds to", d[key])[/c]'))

    print(codetag_parse('sql','The canonical way is to use the built-in cursor iterator. \n [c]curs.execute("select * from people")\nfor row in curs:\n print row\n[/c] \n \n You can use [c]fetchall()[/c] to get all rows at once. \n [c]for row in curs.fetchall():\n print row\n[/c] \n It can be convenient to use this to create a Python list containing the values returned: \n [c]curs.execute("select first_name from people")\nnames = [row[0] for row in curs.fetchall()]\n[/c] \n This can be useful for smaller result sets, but can have bad side effects if the result set is large. \n \n You have to wait for the entire result set to be returned to\nyour client process. \n You may eat up a lot of memory in your client to hold\nthe built-up list. \n It may take a while for Python to construct and deconstruct the\nlist which you are going to immediately discard anyways. \n \n \n If you know there"s a single row being returned in the result set you can call [c]fetchone()[/c] to get the single row. \n [c]curs.execute("select max(x) from t")\nmaxValue = curs.fetchone()[0]\n[/c] \n \n Finally, you can loop over the result set fetching one row at a time. In general, there"s no particular advantage in doing this over using the iterator. \n [c]row = curs.fetchone()\nwhile row:\n print row\n row = curs.fetchone()\n[/c]'))

    print(querytag_parse('python','def conv_only(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], **conv_kwargs):\n def network_fn(X):\n out = tf.cast(X, tf.float32) / 255.\n with tf.variable_scope("convnet"):\n for num_outputs, kernel_size, stride in convs:\n out = layers.convolution2d(out,\n num_outputs=num_outputs,\n kernel_size=kernel_size,\n stride=stride,\n activation_fn=tf.nn.relu,\n **conv_kwargs)\n return out\n return network_fn'))

    print(querytag_parse('python','def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):\n    [q]\n    Recognizes faces in given image using a trained KNN classifier\n\n    :param X_img_path: path to image to be recognized\n    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.\n    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.\n    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance\n           of mis-classifying an unknown person as a known one.\n    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].\n        For faces of unrecognized persons, the name "unknown" will be returned.\n    [/q]    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:\n        raise Exception("Invalid image path: {}".format(X_img_path))\n\n    if knn_clf is None and model_path is None:\n        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")\n\n    [q] Load a trained KNN model (if one was passed in)[/q]    if knn_clf is None:\n        with open(model_path, "rb") as f:\n            knn_clf = pickle.load(f)\n\n    [q] Load image file and find face locations[/q]    X_img = face_recognition.load_image_file(X_img_path)\n    X_face_locations = face_recognition.face_locations(X_img)\n\n    [q] If no faces are found in the image, return an empty result.[/q]    if len(X_face_locations) == 0:\n        return []\n\n    [q] Find encodings for faces in the test iamge[/q]    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)\n\n    [q] Use the KNN model to find the best matches for the test face[/q]    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)\n    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]\n\n    [q] Predict classes and remove classifications that aren"t within the threshold[/q]    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)'))