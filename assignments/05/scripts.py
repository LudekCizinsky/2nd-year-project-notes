#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from termcolor import colored
print(colored('------------- Loading libraries', 'green', attrs=['bold']))
import codecs
import sys
import datetime
import torch
from libs.lec9 import RnnPosTagger
from libs.lec10 import LangClassifier

LOGCL = 'blue'
print("> Done!\n")


# -------- General code
def get_yes_no(text):
  
  while True:
    ans = input(text) 
    if ans in ["y", "n"]:
      return ans

# -------- Detailed code for lecture 9
# See libs/lec9.py

# -------- Detailed code for lecture 10
# See libs/lec10.py

# -------- High level code to run the given tasks in each lecture
def lecture9():

  now = datetime.datetime.now()
  tm = now.strftime("%Y-%m-%d %H:%M:%S")

  print(colored(f'------------- Start of the review of ex. 9: {tm}\n', "red" , attrs=['bold']))
  tg = RnnPosTagger()
  tg.fit('pos-data/da_ddt-ud-train.conllu')
  tg.dev_eval('pos-data/da_arto-dev.conll')
  print()
  print(colored('------------- End of exercise 9', "red" , attrs=['bold']))

def lecture10():

  now = datetime.datetime.now()
  tm = now.strftime("%Y-%m-%d %H:%M:%S")

  print(colored(f'------------- Start of the review of ex. 10: {tm}\n', "red" , attrs=['bold']))
  lc = LangClassifier()
  lc.fit('topic-data/train.txt')
  lc.dev_eval('topic-data/dev.txt')
  print()
  print(colored('------------- End of exercise 10', "red" , attrs=['bold']))

if __name__ == "__main__":
  
  which = sys.argv[1]

  if which == 'l9':
    lecture9()
  elif which == 'l10':
    lecture10()
  else:
    raise ValueError("Undefined lecture!")

