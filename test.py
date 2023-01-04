'''
测试模型的好坏
'''

import torch
import torch.nn.functional as F
from word_sequence import WordSequence
from dataset import get_dataloader
import torch


#加载训练好的模型
model = torch.load('myModel_ws8000_3epoch.pth')

test_loss = 0
correct = 0
co_num = 0
model.eval()

test_dataloader = get_dataloader(train=False)

with torch.no_grad():
    for idx, (input, target) in enumerate(test_dataloader):
        output = model(input)
        #test_loss += F.nll_loss(output, target, reduction = 'sum')
        pred = torch.max(output,dim=-1,keepdim=False)[-1]
        correct=pred.eq(target.data).sum()
        co_num += correct
    test_loss = test_loss/len(test_dataloader.dataset)
    accuracy = co_num/len(test_dataloader.dataset)
    print(accuracy)