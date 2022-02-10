#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import random
import re
from functools import partial

import numpy as np
from sklearn.metrics import confusion_matrix

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
  d2 = "test.pos.ann.conll" # input("Specify path relative to your current location to the second dataset: ")

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

def extract_fts(raw_docs, lbs, langs, model=None):
  
  # Define what constitutes token
  token = " |[\.]{3}|[\w]+|@[\w]+|[\w]+-[\w]+|[\w]+|[\.?!,();:—']"

  # Filter out irrelevant docs if needed
  tmp = []
  for i in range(len(lbs)):
    if lbs[i] in langs:
      tmp.append(raw_docs[i])
  raw_docs = tmp

  # Initialize tokenizer
  tknz = partial(tokenizer, token=token)

  # Get the sparse matrix and vocabulary
  if not model:
    model = CountVectorizer(tknz)
    voc, X = model.fit_transform(raw_docs)
  else:
    voc, X = model.transform(raw_docs)
 
  return voc, X, model

def lecture3():

  # Annotations
  ans = get_yes_no("Do you want to see the annotation exercise? [y/n] ")
  if ans == "y":
    with open("luci.pos.ann.conll") as f:
      lines = f.readlines() 
    for line in lines: print(line)

  # Quality
  ans = get_yes_no("Do you want to see the quality of annotations? [y/n] ")
  if ans == "y":
    annotation_quality()

  # Feature extraction
  ans = get_yes_no("Do you want to extract features? [y/n] ")
  if ans == "y":

    # train
    wooki_train_text, wooki_train_labels = load_langid('langid-data/wookipedia_langid.train.tok.txt')
    voc_train, X_train, model = extract_fts(wooki_train_text, wooki_train_labels, langs=["en"])
    
    # dev
    wooki_dev_text, wooki_dev_labels = load_langid('langid-data/wookipedia_langid.dev.tok.txt')
    voc_dev, X_dev, _ = extract_fts(wooki_dev_text, wooki_dev_labels, langs=["en"], model=model)
    
    print(f"Feature extraction done successfully.")

 
if __name__ == '__main__':

  which = sys.argv[1]
  if which == "l3":
    lecture3()
  elif which == "l4":
    lecture4()

