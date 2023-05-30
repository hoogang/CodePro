

import re

from parsers.sqlang_structured import SqlangParser
from parsers.python_structured import PythonParser

def tokenize_python_code(code):
  tokenized_code, bool_failed_var, bool_failed_token= PythonParser(code)

  return tokenized_code, bool_failed_var, bool_failed_token


def tokenize_sqlang_code(code, bool_remove_comment=True):
  query = SqlangParser(code, regex=True)
  typedCode = query.parseSql()
  tokens = [re.sub('\s+', ' ', x.strip()) for x in typedCode]

  if bool_remove_comment:
    tokens_remove_comment = []
    for token in tokens:
      if token[0:2] == "--":
        pass
      else:
        tokens_remove_comment.append(token)
    tokens = tokens_remove_comment

  return tokens, 0, 0

def tokenize_code_corpus(qid_to_code, pl):
  # not tokenizable
  failed_token_qids = set()
  # not parsable to have vars
  failed_var_qids = set()

  qid_to_tokenized_code = dict()

  count = 0
  for qid, code in qid_to_code.items():
    count += 1
    if count % 1000 == 0:
      print(count)

    # unicode --> ascii
    code = code.strip()
    if len(code) == 0:
      tokenized_code = [""]

    else:
      if pl == "python":
        tokenized_code, bool_failed_var, bool_failed_token = tokenize_python_code(code)
      elif pl == "sql":
        tokenized_code, bool_failed_var, bool_failed_token = tokenize_sqlang_code(code)
      else:
        raise Exception("Invalid programming language! (Support python and sql only.)")

      if bool_failed_token:
        failed_token_qids.add(qid)
        print("failed tokenization qid: %s" % str(qid))
      if bool_failed_var:
        failed_var_qids.add(qid)

    # save
    qid_to_tokenized_code[qid] = tokenized_code

  print("Total size: %d. Fails: %d." % (len(qid_to_tokenized_code), len(failed_token_qids)))

  return qid_to_tokenized_code, failed_var_qids, failed_token_qids

def main():

    code_corpus1 = \
      {
    	"1": "def clamp(n, minn, maxn):\n  return max(min(maxn, n), minn)",
    	"2": "In [7]: df = DataFrame({'A' : list('aabbcd'), 'B' : list('ffghhe')})\n\nIn [8]:"
          " df\nOut[8]: \n   A  B\n0  a  f\n1  a  f\n2  b  g\n3  b  h\n4  c  h\n5  d  e\n\nIn "
          "[9]: df.dtypes\nOut[9]: \nA    object\nB    object\ndtype: object\n\nIn [10]: "
          "df.apply(lambda x: x.astype('category'))       \nOut[10]: \n   A  B\n0  a  f\n1  a  "
          "f\n2  b  g\n3  b  h\n4  c  h\n5  d  e\n\nIn [11]: df.apply(lambda x:"
          " x.astype('category')).dtypes\nOut[11]: \nA    category\nB    category\ndtype: object\n",

        "3":' #include <netinet/in.h>\n#include <arpa/inet.h>\n#import "Base64.h"\n'
      }

    code_corpus2 = \
      {
      "1": "| ID | COLOUR |\n|----|--------|\n|  1 |   Blue |\n|  4 |  Green |\n|  "
                        "5 | Orange |\n|  6 |   Teal |\n|  3 | Yellow |\n|  2 |    Red |\n",
      "2": "Contacts\n----------------------------------\nID          AuditLogID  CreatedOn\n"
                        "----------- ----------- ----------\n10          1           2015-01-02\n11          3"
                        "           2015-05-06\n\nAddresses\n----------------------------------\nID          "
                        "AuditLogID  CreatedOn\n----------- ----------- ----------\n20          4           "
                        "2014-02-01\n21          5           2010-01-01\n\nItems\n----------------------------------\n"
                        "ID          AuditLogID  CreatedOn\n----------- ----------- ----------\n30          2           "
                        "2015-03-04\n31          6           2011-03-04\n",
       "3": "ID     STATUS  CONVERSATION_ID   MESSAGE_ID    DATE_CREATED\n3         2         "
                        "2                95         May, 05 2012 \n2         2         1                87         "
                        "March, 03 2012 \n",
      "4": "INSERT INTO tournTypes VALUES\n(1,2),\n(1,3),\n(2,3),\n(3,1)\n\nINSERT INTO leagueTypes VALUES\n(16,2,0), -- 16 teams, 2 divisions, teams only play within own division\n(8,1,0),\n(28,4,1)\n\nINSERT INTO playoffTypes VALUES\n(8,0), -- 8 teams, single elimination\n(4,0),\n(8,1)\n\nINSERT INTO Schedule VALUES\n('Champions league','2015-12-10','2016-02-10',1),\n('Rec league','2015-11-30','2016-03-04-,2)'\n",

      "5": "$videos = Carousel::find(2)->videos; //finds all videos associated with carousel having id of 2\n\nreturn $videos;\n",
      "6": " SQL> create table mytbl (data_col varchar2(200));\n Table created\n SQL> insert into mytbl values('\u5728\u804c'); \n 1 row inserted.\n SQL> commit;\n Commit complete.\n SQL> select * from mytbl where data_col like '%\u5728\u804c%';\n DATA_COL                                                                                                                                                                                               \n -----------\n \u5728\u804c \n\n SQL> SELECT * FROM nls_database_parameters where parameter='NLS_CHARACTERSET';\n PARAMETER                      VALUE                                  \n ------------------------------ ----------------------------------------\n NLS_CHARACTERSET               AL32UTF8   \n",
      "7":'@prev_sn := SerialNumber,\n@prev_toner := Remain_Toner_Black\n',
      "8": 'WITH QtyCTE AS (\n  SELECT  [Category] = c.category_name\n          , [RootID] = c.category_id\n          , [ChildID] = c.category_id\n  FROM    Categories c\n  UNION ALL \n  SELECT  cte.Category\n          , cte.RootID\n          , c.category_id\n  FROM    QtyCTE cte\n          INNER JOIN Categories c ON c.father_id = cte.ChildID\n)\nSELECT  cte.RootID\n        , cte.Category\n        , COUNT(s.sales_id)\nFROM    QtyCTE cte\n        INNER JOIN Sales s ON s.category_id = cte.ChildID\nGROUP BY cte.RootID, cte.Category\nORDER BY cte.RootID\n'
      }


    qid_to_tokenized_code, failed_var_qids, failed_token_qids = tokenize_code_corpus(code_corpus1, "python")

    print(qid_to_tokenized_code)

    qid_to_tokenized_code, failed_var_qids, failed_token_qids = tokenize_code_corpus(code_corpus2, "sql")

    print(qid_to_tokenized_code)



if __name__ == "__main__":
  main()
