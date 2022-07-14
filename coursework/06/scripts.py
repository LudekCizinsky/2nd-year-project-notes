#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from termcolor import colored
print(colored('------------- Loading libraries', 'green', attrs=['bold']))
import codecs
import sys
import datetime
from libs.lec11 import lec11
from libs.lec12 import lec12
import warnings
warnings.filterwarnings('ignore')

LOGCL = 'blue'
print("> Done!\n")


# -------- General code
def get_yes_no(text):
  
  while True:
    ans = input(text) 
    if ans in ["y", "n"]:
      return ans

# -------- Detailed code for lecture 11
# See libs/lec11.py

# -------- Detailed code for lecture 12
# See libs/lec12.py

# -------- High level code to run the given tasks in each lecture
def lecture11():

  now = datetime.datetime.now()
  tm = now.strftime("%Y-%m-%d %H:%M:%S")

  print(colored(f'------------- Start of the review of ex. 11: {tm}\n', "red" , attrs=['bold']))
  lec11()
  print(colored('------------- End of exercise 11', "red" , attrs=['bold']))

def lecture12():

  now = datetime.datetime.now()
  tm = now.strftime("%Y-%m-%d %H:%M:%S")

  print(colored(f'------------- Start of the review of ex. 12: {tm}\n', "red" , attrs=['bold']))
  lec12()
  print(colored('------------- End of exercise 12', "red" , attrs=['bold']))

if __name__ == "__main__":
  
  which = sys.argv[1]

  if which == 'l11':
    lecture11()
  elif which == 'l12':
    lecture12()
  else: 
    raise ValueError("Undefined lecture!")

