#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("Loading libs...")
import sys
import math
from collections import defaultdict
import gensim.models
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings('ignore')
print("> Done!\n")



# -------- General code
def get_yes_no(text):
  
  while True:
    ans = input(text) 
    if ans in ["y", "n"]:
      return ans

def load_embed(filename):

  return gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)


# -------- Lecture 7 detailed code
def getemb():
  print("Loading twitter embeddings...")
  te = load_embed("twitter.bin")
  print("> Loading successfully done.\n")
  #print("> Here is an embedding of the word cat:")
  #print(te['cat'])
  #print()

  print("Loading google news embeddings...")
  ge = load_embed("googlenews.bin")
  print("> Loading successfully done.\n")
  #print("> Here is an embedding of the word cat:")
  #print(ge['cat'])
  #print()

  return te, ge

def cosine(v1, v2):
  
  assert len(v1) == len(v2), "The provided vectors are not of the same len"
  num = sum([v1[i]*v2[i] for i in range(len(v1))])
  den = (sum([i**2 for i in v1])**.5)*(sum([j**2 for j in v2])**.5)

  return 1 - num/den

def wordnet_sim(w1, w2):

  w1 = wordnet.synsets(w1)[0]
  w2 = wordnet.synsets(w2)[0]
  return w1.wup_similarity(w2)

def show_word_sim(te, ge):

  print("Comparing similarity of cat and dog...")
  print(f"> Distance measure: {te.distance('cat', 'dog')}")
  print(f"> Cosine measure mine: {cosine(te['cat'], te['dog'])}")
  print()
  
  print("Measuring similarity between selected word pairs...")
  words_pairs = [('soccer', 'football'), ('ice', 'snow'), ('sun', 'warm'), ('big', 'huge'), ('cute', 'beautiful')]
  for pair in words_pairs:
    w1, w2 = pair
    print(f'> Pair: {pair}')
    print(f'>> Sim according to twitter emb.: {te.similarity(w1, w2)}')
    print(f'>> Sim according to google news emb.: {ge.similarity(w1, w2)}')
    print(f'>> Sim according to wordnet: {wordnet_sim(w1, w2)}')
    print()

def get_most_sim(m, neg, pos):
  return m.most_similar(positive=pos, negative=neg, topn=10)

def show_analogies(te, ge):

 print("Showing semantic analogies...")
 print(">> Country-capital (given):")
 print(f">>> Twitter {get_most_sim(te, neg=['Denmark'], pos=['England', 'Copenhagen'])}")
 print(f">>> Google News {get_most_sim(ge, neg=['Denmark'], pos=['England', 'Copenhagen'])}")
 print()
 print(">> Object-shape (mine):")
 print(f">>> Twitter {get_most_sim(te, neg=['door'], pos=['button', 'square'])}")
 print(f">>> Google News {get_most_sim(ge, neg=['door'], pos=['button', 'square'])}")
 print()

 print("Showing syntactic analogies...")
 print(">> Superlatives (given):")
 print(f">>> Twitter {get_most_sim(te, neg=['nice'], pos=['good', 'nicer'])}")
 print(f">>> Google News {get_most_sim(ge, neg=['nice'], pos=['good', 'nicer'])}")
 print()
 print(">> Present continuous tense (mine):")
 print(f">>> Twitter {get_most_sim(te, neg=['go'], pos=['run', 'going'])}")
 print(f">>> Google News {get_most_sim(ge, neg=['go'], pos=['run', 'going'])}")
 print()


def transform(txt, pad, win):
  
  print("Started extraction of data for CBOW...")
  words = txt.split(" ")
  print(f">> Size of the text: {len(words)} words")
  n = len(words)
  M = defaultdict()
  M.default_factory = M.__len__
  data = []
  lbls = []

  for i, w in enumerate(words):

    # prev
    pi = i - win
    prev = []
    while pi < 0:
      prev.append(M[pad])
      pi += 1
    if pi != i:
      prev.extend([M[v] for v in words[pi: i]])

    # nxt
    ni = i + win
    nxt = []
    while ni >= n:
      nxt.append(M[pad])
      ni -= 1
    if ni != i:
      nxt.extend([M[v] for v in words[i+1:ni+1]])

    # Combine
    data.append(prev + nxt)
    lbls.append(M[w])
  
  M.default_factory = None
  print(f">> Total vocab size: {len(M)}")
  print("> Done.\n")

  return torch.LongTensor(data), torch.LongTensor(lbls), M

def do_cbow_extraction():
  
  pad = "<PAD>"
  test = False
  if test:
    tiny_corpus = ["this is an example", "this is a longer example sentence", "I love deep learning"] 
    txt = " ".join(tiny_corpus)
    X, y, M = transform(txt, pad, win=2) 
  
  with open("sample.txt") as f:
    txt = " ".join(f.readlines()) 
  X, y, M = transform(txt, pad, win=2) 
  
  return X, y, M

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        assert x.shape[0] == y.shape[0] # assuming shape[0] = dataset size
        self.x = x
        self.y = y


    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class CBOW(nn.Module):

    def __init__(self, emb_dim, vocab_dim):
        super(CBOW, self).__init__()
        self.W = torch.nn.Embedding(vocab_dim, emb_dim)
        self.W1 = torch.nn.Linear(emb_dim, vocab_dim)
 
    def forward(self, X):
         
        emb = torch.sum(self.W(X), 1)
        out = self.W1(emb)                 
        return out
    
    def fit(self, X, y, epochs):
      
      print("Started training...") 
      traindata = MyDataset(X, y)
      optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
      criterion = torch.nn.CrossEntropyLoss()
      for epoch in range(epochs):
        print(f">> Epoch {epoch + 1}")
        trainloader = torch.utils.data.DataLoader(traindata, batch_size=100, shuffle=True)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
          # get the inputs; data is a list of [inputs, labels]
          xx, yy = data

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          out = self.forward(xx)
          loss = criterion(out, yy)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()  
          if i % 100  == 99:    # print every 100 mini-batches
              print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
              running_loss = 0.0

      print('> Finished Training.')


def estimate_emb(X=None, y=None, M=None):


  if X is None:
    pad = "<PAD>"
    with open("sample.txt") as f:
      txt = " ".join(f.readlines()) 
    X, y, M = transform(txt, pad, win=2) 
  
  embed_dim = 64
  print(f"Len of V: {len(M)}")
  md = CBOW(embed_dim, len(M))
  md.fit(X, y, 2)

  print("\nExtracting embeddings parameters...")
  weights = md.W.weight.tolist()
  print(f"Len of weights: {len(weights)}")
  MI = {value: key for key, value in M.items()}
  r = f"{len(M)} {embed_dim}\n"
  c = 0 
  for i, w in enumerate(weights):
    word = MI[i].strip()
    w = " ".join([str(j) for j in w])
    r += word + " " + w + "\n"
    c += 1
  with open("embeds.txt", "w") as f:
    f.write(r) 
  print("> Done.\n")


def show_sim_words():
  
  print("Loading computed vectors to gensim...")
  oe = gensim.models.KeyedVectors.load_word2vec_format('embeds.txt', binary=False)
  print("> Done.\n")

  print("Most similar words to the word 'life':")
  for w, p in oe.most_similar('life', topn=5):
    print(f">> Word: {w} | Prob: {p}")
  print()
  print("Most similar words to the word 'learning':")
  for w, p in oe.most_similar('learning', topn=5):
    print(f">> Word: {w} | Prob: {p}")


# -------- High level code to run the given tasks in each lecture
def lecture7():
  
  te, ge = getemb()

  ans = get_yes_no("-- Do you want to see word similarities? ")
  if ans == "y":
    show_word_sim(te, ge) 

  ans = get_yes_no("-- Do you want to see analogies? ")
  if ans == "y":
    show_analogies(te, ge)
  
  ans = get_yes_no("-- Do you want to see the extraction of data for CBOW? ")
  if ans == "y":
    X, y, M = do_cbow_extraction()
  else:
    X, y, M = None, None 

  ans = get_yes_no("-- Do you want to estimate the word embeddings? ")
  if ans == "y":
    estimate_emb(X, y, M)

  ans = get_yes_no("-- Do you want to see the similar words using the trained embeddings? ")
  if ans == "y":
    show_sim_words()

def lecture8():
  print("See the sol.md file as the tasks were answered there, no coding required.")

if __name__ == "__main__":
  
  which = sys.argv[1]

  if which == 'l7':
    lecture7()
  elif which == 'l8':
    lecture8()
  else:
    raise ValueError("Undefined lecture!")

