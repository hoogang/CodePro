
import ruler
import modeler
import torch
import json

# vocab_path = './resource/word_vocab.json'
# model_path = './resource/java_vae.model'

vocab_path = './vae/data/java/word_vocab.json'
model_path = './vae/run/java/model/model.ckpt'


with open(vocab_path, 'r') as f:
    word_vocab = json.load(f)

model_file = torch.load(model_path)

rule_list = ['contain_any_javadoc_tag','not_contain_any_letter']

def my_rule1(comment_str):
    return len(comment_str) < 10

def my_rule2(comment_str):
    if comment_str.startswith('Return'):
        return comment_str.replace('Return', '')
    else:
        return True

my_rule_dict = {
    'length_less_than_10':my_rule1,
    'detach_return':my_rule2
}


raw_comments = ['======','create a remote conection','create a remote conection','return @test','convert int to string']

print(raw_comments)

comments,idx = ruler.filter(raw_comments, selected_rules=rule_list,defined_rule_dict=my_rule_dict)
print('rule filter:',comments,idx)
#['create a remote conection', 'convert int to string'] [1,3]


comments,idx =  modeler.filter(raw_comments, word_vocab, model_file, with_cuda=True, comment_len=20,num_workers=1, max_iter=1000)
print('model filter:',comments,idx)
# ['create a remote conection', 'convert int to string'] [1,3]
