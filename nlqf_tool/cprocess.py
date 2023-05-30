import re

# 骆驼命名法
import inflection

# 词性还原
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
wnler = WordNetLemmatizer()

# 词干提取
from nltk.corpus import wordnet


def revert_abbrev(line):
    patterns = [
        (r'won\"t', 'will not'),
        (r'can\"t', 'cannot'),
        (r'i\"m', 'i am'),
        (r'ain\"t', 'is not'),
        (r'(\w+)\"ll', '\g<1> will'),
        (r'(\w+)n\"t', '\g<1> not'),
        (r'(\w+)\"ve', '\g<1> have'),
        (r'(\w+)\"s', '\g<1> is'),
        (r"(\w+)\"re", '\g<1> are'),
        (r'(\w+)\"d', '\g<1> would')]

    match_patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    for p,r in match_patterns:
        (line, count) = re.subn(p, r, line)

    return line


# 获取词性
def get_wordpos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def process_mark(line):
    line = line.replace('\'','\"')
    # 句子预处理
    line = revert_abbrev(line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = line.replace('\n', ' ')
    line = line.replace('\t', ' ')
    line = re.sub(' +', ' ', line)
    line = line.strip()
    # 骆驼命名转下划线
    line = inflection.underscore(line)
    return line


def process_words(line):
    # 找单词
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)
    # 替换小数
    decimal = re.compile(r"\d+(\.\d+)+")
    line = re.sub(decimal, 'TAGINT', line)
    # 替换字符串
    string = re.compile(r'\"[^\"]+\"')
    line = re.sub(string, 'TAGSTR', line)
    # 替换十六进制
    decimal = re.compile(r"0[xX][A-Fa-f0-9]+")
    line = re.sub(decimal, 'TAGINT', line)
    # 替换数字 56
    number = re.compile(r"\s\d+\s?")
    line = re.sub(number, ' TAGINT ', line)
    # 替换字符 6c60b8e1
    # 后缀匹配
    other = re.compile(r"(?<![A-Z|a-z|_|])\d+[A-Za-z]+")
    line = re.sub(other, 'TAGOER', line)
    cut_words = line.split(' ')
    # 全部小写化
    cut_words = [x.lower() for x in cut_words]
    # 词性标注
    word_tags = pos_tag(cut_words)
    tags_dict = dict(word_tags)
    word_list = []
    for word in cut_words:
        word_pos = get_wordpos(tags_dict[word])
        if word_pos in ['a', 'v', 'n', 'r']:
            # 词性还原
            word = wnler.lemmatize(word, pos=word_pos)
        # 词干提取(效果最好）
        word = wordnet.morphy(word) if wordnet.morphy(word) else word
        word_list.append(word)
    return word_list


#############################################################################
def filter_invachar(line):
    # 去除非常用符号；防止解析有误
    line = re.sub('[^(0-9|a-z|A-Z|\-|_|\'|\"|\-|\(|\)|\n)]+', ' ', line)
    # 包括\r\t也清除了
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    return line


def processing(line):
    line =  filter_invachar(line)
    line =  process_mark(line)
    words = process_words(line)
    # 分完词后,再去掉 括号
    for i in range(0, len(words)):
        if re.findall('[\(\)]', words[i]):
            words[i] = ''
    # 列表里包含 '' 或 ' '
    words = [x.strip() for x in words if x.strip() != '']
    # 解析可能为空
    line = ' '.join(words) if words!=[] else False
    return line

if __name__ == '__main__':
    processing('How to save an image-file on server through Java applet?')
    processing('return @test \n')

        
