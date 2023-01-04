"""进行模型的训练"""
from word_sequece import WordSequence
from dataset import get_dataloader
from torch.optim import Adam
from tqdm import tqdm
import torch.nn.functional as F
import torch
#from model import CNN
#from model import CNN1d
#import sklearn.svm as svm

my_model = torch.load('myModel_ws8000_2epoch.pth')
#my_model = CNN1d(vocab_size=10585*2, embedding_dim=128, n_filters=2, filter_sizes=[1,2,3], output_dim=5, dropout=0.2, pad_idx=0)
#my_model = svm.SVC(kernel="rbf", decision_function_shape="ovr")
optimizer = Adam(my_model.parameters())

def train(epoch):
    train_dataloader = get_dataloader(train=True)
    bar = tqdm(train_dataloader, total=len(train_dataloader))
    for idx, (input, target) in enumerate(bar):
        optimizer.zero_grad()
        output = my_model(input)
        #loss = F.nll_loss(output,target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        bar.set_description("epcoh:{}  idx:{}   loss:{:.6f}".format(epoch,idx,loss.item()))


if __name__ == '__main__':
    for i in range(1):
        train(i)

    torch.save(my_model, 'myModel_ws8000_3epoch.pth')
