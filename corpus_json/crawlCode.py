
import json
import base64
import pandas as pd
import numpy  as np
from urllib import request


# 保证以CSV保存字符串
def filter_inva_char(line):
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    # 去除\r
    line = line.replace('\r', ' ')
    # 切换换行
    line = line.replace('\n', '\\n')
    return line


def get_results(headers, owner, repo, path, ref):
    url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}'.format(owner=owner, repo=repo, path=path, ref=ref)
    print("原始url：",url)
    req = request.Request(url, headers=headers)
    response = request.urlopen(req).read()
    result = json.loads(response.decode())
    return result


def main():

    headers = {'User-Agent': 'Mozilla/5.0',
               'Authorization': 'token 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',
               'Content-Type': 'application/json',
               'Accept': 'application/vnd.github+json'
               }

    csndf_data = pd.read_csv("annotationStore.csv",encoding='utf-8',header=0)
    url_list = csndf_data["GitHubUrl"]


    code = []
    for i in range(len(csndf_data)):
        url_raw = url_list[i]
        print("原始url：", url_raw)
        s = url_raw.split("/")
        owner = s[3]
        repo = s[4]
        path = '/'.join(s[7:]).split("#")[0]
        ref = s[6]

        str_getIndex  = [int(x.lstrip('L')) for x in url_raw.split("#")[1].split('-')]

        try:
            results = get_results(headers, owner, repo, path, ref)
        except:
            print("爬虫解析错误", i)
            code.append('')
            continue

        r = results["content"]
        result = str(base64.b64decode(r), "utf-8")
        string_line = result.splitlines(True)
        result = ' '.join(string_line[str_getIndex[0]-1:str_getIndex[1]])

        if str_getIndex[1]-str_getIndex[0]<=2:
            code.append('')
        else:
            code.append(filter_inva_char(result.strip()))

        print(f"{i}条数据处理结束".format(i=i))

    col_name = csndf_data.columns.tolist()

    # 构建集
    csndf_data.insert(col_name.index('Query') + 1, 'Code', code)

    print('原始收集数据的大小', csndf_data.shape)  #4006

    csndf_data['Code'].replace(r'^\s*$', value=np.nan,regex=True,inplace=True)

    csndf_data.dropna(subset=['Code'], inplace=True)

    # 重新0-索引
    csndf_data = csndf_data.reset_index(drop=True)
    print('丢弃无效数据的大小', csndf_data.shape)   #3912

    # 单引号置换双引号 便于解析 (主要是代码）
    for name in ['Query', 'Code']:
        csndf_data[name] = csndf_data[name].str.replace('\'', '\"')

    csndf_data = csndf_data[['Language','Query','Code','Relevance','GitHubUrl']]
    
    csndf_data.to_csv("csn_annotation_pred.csv",  encoding='utf-8', sep='|')

if __name__ == '__main__':
    main()



