
import re
import torch
import tqdm
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from nltk.stem import WordNetLemmatizer

# from cprocess import processing
#
# class CommentDataset(Dataset):
#     PAD = 0
#     UNK = 1
#     BOS = 2
#     EOS = 3
#
#     def __init__(self, comment, word_vocab, comment_len,is_docstring=False):
#         super(CommentDataset, self).__init__()
#         self.comment = []
#
#         if isinstance(comment,str):
#             with open(comment, 'r') as f:
#                 lines = f.readlines()
#         else:
#             lines = comment
#
#         lines = [processing(i) for i in lines] if is_docstring else lines
#
#         for q in lines:
#             words = q.split()[:comment_len]
#             if is_docstring:
#                 words = [word_vocab.get(w, self.UNK) for w in words] + [self.EOS]
#             else:
#                 words = [int(w) for w in words] + [self.EOS]
#             padding = [self.PAD for _ in range(comment_len - len(words)+2)]
#             words.extend(padding)
#             self.comment.append(words)
#
#
#     def __len__(self):
#         return len(self.comment)
#
#     def __getitem__(self, item):
#         return torch.tensor(self.comment[item]),torch.tensor(self.comment[item])


class CommentDataset(Dataset):
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    def __init__(self, comment, word_vocab, comment_len,is_docstring=False):
        super(CommentDataset, self).__init__()
        lemmatizer = WordNetLemmatizer()

        self.comment = []


        if isinstance(comment, str):
            with open(comment, 'r') as f:
                lines = f.readlines()
        else:
            lines = comment

        lines = [' '.join(re.findall(r"[\w]+|[^\s\w]", i.lower())) for i in lines] if is_docstring else lines

        for q in lines:
            words = q.split()[:comment_len]
            if is_docstring:
                if len(words) > 0 and word_vocab.get(words[0], self.UNK) == self.UNK:
                    words[0] = lemmatizer.lemmatize(words[0], pos='v')
                words = [word_vocab.get(w, self.UNK) for w in words] + [self.EOS]
            else:
                words = [int(w) for w in words] + [self.EOS]
            padding = [self.PAD for _ in range(comment_len - len(words) + 2)]
            words.extend(padding)
            self.comment.append(words)

    def __len__(self):
        return len(self.comment)

    def __getitem__(self, item):
        return torch.tensor(self.comment[item]),torch.tensor(self.comment[item])


def compute_loss(model, data_loader, vocab_size, device):
    loss_values = []

    loss_weight = torch.ones((vocab_size)).to(device)
    loss_weight[0] = 0
    loss_weight[1] = 0

    loss = torch.nn.CrossEntropyLoss(weight=loss_weight)
    
    for data, target in tqdm.tqdm(data_loader):
        with torch.no_grad():
            data = data.to(device)
            target = target.to(device)
            m, l, z, decoded = model.forward(data)
            decoded = decoded.view(-1, vocab_size)
            loss_value = loss(decoded.view(-1, vocab_size), target.view(-1))
            loss_values.append(loss_value.item())

    return np.array(loss_values)


def filter(comments, word_vocab, model, with_cuda=True, comment_len=20,num_workers=1,max_iter=1000,dividing_point=None):

    if not isinstance(comments, list):
        raise TypeError('comments must be a list')
    
    if len(comments) < 2:
        raise ValueError('The length of comments must be greater than 1')

    vocab_size = len(word_vocab)

    corpus = CommentDataset(comments, word_vocab, comment_len, is_docstring=True)

    data_loader = DataLoader(dataset=corpus, batch_size=1, shuffle=False, num_workers=num_workers)

    model.eval()
    device = torch.device("cuda:0" if with_cuda else "cpu")
    model = model.to(device)

    losses = compute_loss(model,data_loader, vocab_size, device)

    # 返回值从小到大的索引排序
    idx = np.argsort(losses)

    losses = np.array(losses)[idx].reshape(-1, 1)

    if dividing_point is None:
        gmm = GaussianMixture(n_components=2, covariance_type='full', max_iter=max_iter).fit(losses)
        predicted = gmm.predict(losses)
        return_type = predicted[np.argmin(losses)]
        idx_new = idx[predicted == return_type]
        idx_new.sort()
    elif isinstance(dividing_point, int):
        idx_new = idx[:int(len(idx)*dividing_point)]
    elif isinstance(dividing_point, function):
        idx_new = dividing_point(idx,losses)

    filtered_comments = [comments[i] for i in idx_new]

    return filtered_comments, idx_new
