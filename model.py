"""
构建模型
Mymodel为感知机模型
CNN和CNN1d为textCNN模型，区别在于前者用2dconv，后者用1dconv
"""
import torch.nn as nn
import config
import torch.nn.functional as F
import torch
import sklearn

class myModel(nn.Module):
    def __init__(self):
        super(myModel,self).__init__()
        #self.embedding = nn.Embedding(num_embeddings=len(config.ws),embedding_dim=200,padding_idx=config.ws.PAD)
        self.embedding = nn.Embedding(num_embeddings=18570, embedding_dim=300, padding_idx=0)
        self.fc = nn.Linear(config.max_len * 300, 5)

    def forward(self, x):
        """
        :param x:[batch_size,max_len]
        :return:
        """
        x = self.embedding(x) #input embeded :[batch_size,max_len,200]
        #变形
        x = x.view(x.size(0),-1)
        #全连接
        out = self.fc(x)
        #print(out)
        return F.softmax(out, dim=-1)
        #return F.log_softmax(out,dim=-1)


class CNN(nn.Module):
    def __init__(self, vocab_size=10585*2, embedding_dim=100, num_filter=2,
                 filter_sizes=[1,2,3], output_dim=5, dropout=0.2, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=num_filter, kernel_size=(fs, embedding_dim))
                                    for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * num_filter, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, text):
        #embedded = self.dropout(self.embedding(text))  # [batch size, sent len, emb dim]
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)  # [batch size, 1, sent len, emb dim]
        # 升维是为了和nn.Conv2d的输入维度吻合，把channel列升维。
        # num_filters代表通道数
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved = [batch size, num_filter, sent len - filter_sizes+1]
        # 有几个filter_sizes就有几个conved
        pooled = [F.max_pool1d(conv,conv.shape[2]).squeeze(2) for conv in conved]  # [batch,num_filter]
        x_cat=torch.cat(pooled, dim=1)
        cat = self.dropout(x_cat)
        # cat = [batch size, num_filter * len(filter_sizes)]# 把 len(filter_sizes)个卷积模型concate起来传到全连接层。
        return self.fc(cat)

class CNN1d(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
#text = [batch_size,sentence,embedding]

    def forward(self, text):
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.permute(0, 2, 1)
        # embedded = [batch size, emb dim, sent len]
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)

