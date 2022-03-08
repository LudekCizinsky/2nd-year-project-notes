from collections import defaultdict
from termcolor import colored
import torch
import torch.nn as nn
from .common import MyDataset


LOGCL = "blue"

class LSTMCL(torch.nn.Module):
    def __init__(self, nwords, K, emb_dim, hidden_dim):
        super(LSTMCL, self).__init__()
        self.E = torch.nn.Embedding(nwords, emb_dim)
        self.lstm = torch.nn.LSTM(
            emb_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(2*hidden_dim, K)
    
    def forward(self, X):

        emb = self.E(X)
        dp1 = torch.nn.Dropout(0.2)
        emb = dp1(emb)
        yhat, _ = self.lstm(emb)
        dp2 = torch.nn.Dropout(0.3)
        out = self.fc(dp2(yhat))
        sm = nn.Softmax(dim=2)
        out = sm(out)
        out = out[:, -1, :]

        return out

class LangClassifier:

  def __init__(self):
    self.model = None

    self.X_train = None
    self.y_train = None
    self.X_dev = None
    self.y_dev = None

    self.MX = None
    self.MY = None
    
    self.embdim = 100
    self.lstmdim = 50
    self.lr = 0.01
    self.epochs = 5
    self.bsize = 64

  def train_eval(self):
    y_hat = self.predict(self.X_train)
    acc = torch.sum(self.y_train == y_hat)/y_hat.size(0)
    print(f"> Training Accuracy: {acc}")

  def dev_eval(self, filename):

    print(colored('[Evaluation on dev set started]', "magenta" , attrs=['bold']))

    print(colored('[Loading and transforming dev data]', LOGCL, attrs=['bold']))
    rawdevx, rawdevy = self._load_data(filename)
    self.X_dev, _ = self._raw2idx(rawdevx, mx=32, M=self.MX)
    self.y_dev, _ = self._raw2idx(rawdevy, mx=1, M=self.MY, flatten=True)
    n, m = self.X_dev.size()
    print(f"> Resulting dimension of loaded data: {n} x {m}")
    print("> Here are first two lines in the training:\n")
    for l in self.X_dev[:2]:
      print(l)
      print()
    print()
    print("> Here are first two lines in the labels:\n")
    for l in self.y_dev[:2]:
      print(l)
      print()
    print()
 
    print(colored('[Making predictions]', LOGCL, attrs=['bold']))
    y_hat = self.predict(self.X_dev)
    acc = torch.sum(self.y_dev == y_hat)/y_hat.size(0)
    print(f"> Dev Accuracy: {acc}")

  def predict(self, X):

    yhat = self.model.forward(X)
    return torch.argmax(yhat, 1)


  def fit(self, filename):
    
    print(colored('[Training started]', "magenta" , attrs=['bold']))

    print(colored('[Loading and transforming training data]', LOGCL, attrs=['bold']))
    rawtrainx, rawtrainy = self._load_data(filename)
    self.X_train, self.MX = self._raw2idx(rawtrainx, mx=32)
    self.y_train, self.MY = self._raw2idx(rawtrainy, mx=1, flatten=True)
    n, m = self.X_train.size()
    print(f"> Resulting dimension of loaded data: {n} x {m}")
    print("> Here are first two lines in the training:\n")
    for l in self.X_train[:2]:
      print(l)
      print()
    print()
    print("> Here are first two lines in the labels:\n")
    for l in self.y_train[:2]:
      print(l)
      print()
    print()
 
    print(colored('[Preparing model]', LOGCL, attrs=['bold']))
    torch.manual_seed(0)
    self.model = LSTMCL(
        nwords=len(self.MX),
        K=len(self.MY),
        emb_dim=self.embdim,
        hidden_dim=self.lstmdim
    )
    print("> Done!\n")

    print(colored('[Learning optimal parameters]', LOGCL, attrs=['bold']))
    traindata = MyDataset(self.X_train, self.y_train)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(self.epochs):

      print(f">> Epoch {epoch + 1}")
      trainloader = torch.utils.data.DataLoader(traindata, batch_size=self.bsize, shuffle=True)
      running_loss = 0.0
      
      for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        xx, yy = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        out = self.model.forward(xx)
        
        loss = criterion(out, yy)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        bn = 50 # After how many batches do you want to print the loss
        if i % bn  == (bn - 1):    
          print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / bn:.3f}')
          running_loss = 0.0

    self.train_eval()
    print('> Done!\n')

  def _load_data(self, path):
    text = []
    labels = []
    for lineIdx, line in enumerate(open(path)):
        tok = line.strip().split('\t')
        labels.append(tok[0])
        text.append(tok[1].split(' '))
    return text, labels

  def _raw2idx(self, raw, mx, M=None, flatten=False):
    
    if M is None:
      M = defaultdict()
      M.default_factory = M.__len__

    res = [] 
    for x in raw:

      if isinstance(x, str):
        x = [x]

      t = (mx - len(x))
      if t < 0:
        t = 0

      tmpx = [] 
      for i, w in enumerate(x):
          if i == mx:
            break
          try:
            tmpx.append(M[w])
          except KeyError:
            tmpx.append(M["PAD"])
      tmpx = tmpx + [M["PAD"]]*t
      res.append(tmpx)
   
    M.default_factory = None

    if flatten:
      tmp = []
      for x in res:
        tmp.extend(x)
      res = tmp
    
    return torch.LongTensor(res), M

