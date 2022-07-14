#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import random
import re
from functools import partial

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# My personal library
sys.path.insert(0, "/Users/ludekcizinsky/Dev/personal/nano-learn")
from nnlearn.feature_extraction.text import CountVectorizer 

labels = ["ADJ", "ADV", "NOUN", "VERB", "PROPN", "INTJ", "ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ", "PUNCT", "SYM", "X"] 

def tokenizer(text, token):
      
    tokens = re.findall(token, text) 
    tmp = []
    skip = False
    for i in range(1, len(tokens) - 1):

        if not skip:
            prev, curr, nxt = tokens[i - 1], tokens[i], tokens[i + 1]
            prohibited = ["'", " "]
            is_relevant = prev not in prohibited and curr in ["'"] and nxt not in prohibited
            
            if is_relevant:
                tmp.append(curr + nxt)
                skip = True
            else:
                tmp.append(curr)
        else:
            skip = False
    
    tokens = [tokens[0]] + tmp + [tokens[-1]]  
    
    return tokens


def get_yes_no(text):
  
  while True:
    ans = input(text) 
    if ans in ["y", "n"]:
      return ans

def load_ann(path):

  with open(path) as f:
    lines = f.readlines()
    pos = [line.strip().split(" ")[1] for line in lines if line.strip()]
  return np.array(pos)

def generate_random_ann(p, D):

  res = []
  for d in D:
    if random.random() < p:
      res.append(d)
    else:
      options = list(set(labels) - set([d]))
      res.append(random.choice(options))

  return np.array(res)

def annotation_quality():

  d1 = "luci.pos.ann.conll" # input("Specify path relative to your current location to the first dataset: ")
  d2 = "other.pos.ann.conll" # input("Specify path relative to your current location to the second dataset: ")

  ann1 = load_ann(d1)
  ann2 = generate_random_ann(.8, ann1)  # load_ann(d2)
  acc = sum(ann1 == ann2)/ann1.shape[0]
  print(f"- Accuracy score: {acc}")

  print(f"- Kohen's Kappa: {kohen_kappa(ann1, ann2, labels)}")

  print("- Different annotations:")
  for i in range(len(ann1)):
    if ann1[i] != ann2[i]:
      print(f"{i}: Ann1: {ann1[i]} x Ann2: {ann2[i]}")


def kohen_kappa(ann1, ann2, labels):

  n = ann1.shape[0]
  po = sum(ann1 == ann2)/n

  cm = confusion_matrix(ann1, ann2)
  pe = 0
  for k in range(cm.shape[0]):
    pe += (cm[k, :].sum()/n)*(cm[:, k].sum()/n)

  return (po - pe)/(1 - pe)


def load_langid(path):
    text = []
    lbs = []
    for line in open(path):
        tok = line.strip().split('\t')
        lbs.append(tok[0])
        text.append(tok[1])
    return text, lbs


def extract_fts(raw_docs, analyzer="word", tknz=None, ngram_range=None, model=None):
  
  # Get the sparse matrix and vocabulary
  if not model:
    model = CountVectorizer(tokenizer=tknz, ngram_range=ngram_range, analyzer=analyzer)
    voc, X = model.fit_transform(raw_docs)
  else:
    voc, X = model.transform(raw_docs)
 
  return voc, X, model


def bow_performance():

  # Define what constitutes token
  token = " |[\.]{3}|[\w]+|@[\w]+|[\w]+-[\w]+|[\w]+|[\.?!,();:—']"

  # Initialize tokenizer
  tknz = partial(tokenizer, token=token)

  print("Extracting features...")
  # train
  wooki_train_text, y_train = load_langid('langid-data/wookipedia_langid.train.tok.txt')
  voc_train, X_train, model = extract_fts(wooki_train_text, tknz=tknz)
  
  # dev
  wooki_dev_text, y_dev = load_langid('langid-data/wookipedia_langid.dev.tok.txt')
  voc_dev, X_dev, _ = extract_fts(wooki_dev_text, model=model)
  print("Done!")

  # Train the models
  print("Training models...")
  NB = MultinomialNB()
  LR = LogisticRegression(solver='liblinear')
  NB.fit(X_train, y_train)
  LR.fit(X_train, y_train)
  print("Done!")

  # Try the performance on wookipedia dataset
  y_hat_nb = NB.predict(X_dev) 
  y_hat_lr = LR.predict(X_dev)
  print("> Wookipedia dataset performance")
  print(f"NB acc: {accuracy_score(y_dev, y_hat_nb)} | LR acc: {accuracy_score(y_dev, y_hat_lr)}")

  # Bulba dataset performance
  bulba_dev_text, y_dev = load_langid('langid-data/bulbapedia_langid.dev.tok.txt')
  voc_dev, X_dev, _ = extract_fts(bulba_dev_text, model=model)
  y_hat_nb = NB.predict(X_dev) 
  y_hat_lr = LR.predict(X_dev)
  print("> Bulba dataset performance")
  print(f"NB acc: {accuracy_score(y_dev, y_hat_nb)} | LR acc: {accuracy_score(y_dev, y_hat_lr)}")

  # Compute confusion matrix
  print("> Confusion matrix for LR model evaluated on bulba dataset")
  print(LR.classes_)
  print(confusion_matrix(y_dev, y_hat_lr))

  # Inspect the most interesting features for both models
  words = model.get_feature_names()
  nb_w = np.mean(NB.coef_, axis=0)
  nb_top_i = (-nb_w).argsort()[:5]
  print(f"Here are top words for NB: {words[nb_top_i]}")
  lr_w = np.mean(LR.coef_, axis=0)
  lr_top_i = (-lr_w).argsort()[:5]
  print(f"Here are top words for LR: {words[lr_top_i]}")


def ngram_char_perm():

  print("Extracting features...")
  # train
  wooki_train_text, y_train = load_langid('langid-data/wookipedia_langid.train.tok.txt')
  voc_train, X_train, model = extract_fts(wooki_train_text, analyzer="char",
      ngram_range=(1,3))
  
  # dev
  wooki_dev_text, y_dev = load_langid('langid-data/wookipedia_langid.dev.tok.txt')
  voc_dev, X_dev, _ = extract_fts(wooki_dev_text, model=model)
  print("Done!")

  # Train the models
  print("Training models...")
  NB = MultinomialNB()
  LR = LogisticRegression(solver='liblinear')
  NB.fit(X_train, y_train)
  LR.fit(X_train, y_train)
  print("Done!")

  # Try the performance on wookipedia dataset
  y_hat_nb = NB.predict(X_dev) 
  y_hat_lr = LR.predict(X_dev)
  print("> Wookipedia dataset performance")
  print(f"NB acc: {accuracy_score(y_dev, y_hat_nb)} | LR acc: {accuracy_score(y_dev, y_hat_lr)}")

  # Bulba dataset performance
  bulba_dev_text, y_dev = load_langid('langid-data/bulbapedia_langid.dev.tok.txt')
  voc_dev, X_dev, _ = extract_fts(bulba_dev_text, model=model)
  y_hat_nb = NB.predict(X_dev) 
  y_hat_lr = LR.predict(X_dev)
  print("> Bulba dataset performance")
  print(f"NB acc: {accuracy_score(y_dev, y_hat_nb)} | LR acc: {accuracy_score(y_dev, y_hat_lr)}")

  # Compute confusion matrix
  print("> Confusion matrix for LR model evaluated on bulba dataset")
  print(LR.classes_)
  print(confusion_matrix(y_dev, y_hat_lr))

  # Inspect the most interesting features for both models
  words = model.get_feature_names()
  nb_w = np.mean(NB.coef_, axis=0)
  nb_top_i = (-nb_w).argsort()[:5]
  print(f"Here are top words for NB: {words[nb_top_i]}")
  lr_w = np.mean(LR.coef_, axis=0)
  lr_top_i = (-lr_w).argsort()[:5]
  print(f"Here are top words for LR: {words[lr_top_i]}")


def lecture3():

  # Annotations
  ans = get_yes_no("-- Do you want to see the annotation exercise? [y/n] ")
  if ans == "y":
    with open("luci.pos.ann.conll") as f:
      lines = f.readlines() 
    for line in lines: print(line)

  # Quality
  ans = get_yes_no("-- Do you want to see the quality of annotations? [y/n] ")
  if ans == "y":
    annotation_quality()


def lecture4():

  ans = get_yes_no("-- Do you want to see the results for the theoretical exercise? [y/n] ")
  if ans == "y":
    print("Visit this site to see it under week 2: https://deepnote.com/project/SYP--pD6EPwxQJqHOzuVtsapEA/%2Fnotebook.ipynb")

  ans = get_yes_no("-- Do you want to see the performance of NB and LR on unigram features? [y/n] ")
  if ans == "y":
    bow_performance()

  ans = get_yes_no("-- Do you want to see the performance of NB and LR on the trigram features? [y/n] ")
  if ans == "y":
    ngram_char_perm()


if __name__ == '__main__':

  which = sys.argv[1]
  if which == "l3":
    lecture3()
  elif which == "l4":
    lecture4()

