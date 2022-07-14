#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys

text1 = """
'Curiouser and curiouser!' cried Alice (she was so much surprised, that for the moment she quite forgot how to speak good English); 'now I'm opening out like the largest telescope that ever was! Good-bye, feet!' (for when she looked down at her feet, they seemed to be almost out of sight, they were getting so far off). 'Oh, my poor little feet, I wonder who will put on your shoes and stockings for you now, dears? I'm sure I shan't be able! I shall be a great deal too far off to trouble myself about you: you must manage the best way you can; —but I must be kind to them,' thought Alice, 'or perhaps they won't walk the way I want to go! Let me see: I'll give them a new pair of boots every Christmas...'
"""

tweet = "@robv New vids coming tomorrow #excited_as_a_child, can't w8!!" 

text2 = """
Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch is the longest official one-word placename in U.K. Isn't that weird? I mean, someone took the effort to really make this name as complicated as possible, huh?! Of course, U.S.A. also has its own record in the longest name, albeit a bit shorter... This record belongs to the place called Chargoggagoggmanchauggagoggchaubunagungamaugg. There's so many wonderful little details one can find out while browsing http://www.wikipedia.org during their Ph.D. or an M.Sc.
"""

def tokenizer(token, text):
      
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


def sentence_segment1(match_regex, tokens):

    current = []
    sentences = [current]
    for tok in tokens:
        current.append(tok)
        if re.match(match_regex, tok):
            current = []
            sentences.append(current)
    if not sentences[-1]:
        sentences.pop(-1)
    return sentences

def sentence_segment2(prev, nxt, tokens):
    sent = [tokens[0]]
    result = []
    for i in range(1, len(tokens)-1):
       p, curr, n = tokens[i-1], tokens[i], tokens[i+1] 
       if re.match(prev, p) and curr == " " and re.match(nxt, n):
         sent.append(curr)
         result.append(sent)
         sent = []
       else:
         sent.append(curr)
    
    if len(sent) > 0:
      result.append(sent)
    
    return result

def lecture1():

    # ----------------- Token definitions 
    # With 3 dots together
    token1 = " |[\.]{3}|[\w]+-[\w]+|[\w]+|[\.?!,();:—']"
    
    # Dots separetely
    token2 = " |[\w]+-[\w]+|[\w]+|[\.?!,();:—']" 

    # Twitter token definition
    token3 = " |[\.]{3}|[\w]+|@[\w]+|[\w]+-[\w]+|[\w]+|[\.?!,();:—']"

    # ----------------- Segmentors definitions
    segment1 = "\."
    segment2 = "[\.?!] [A-Z]"

    # ----------------- Run tokenizer
    print("----------- Token definition with dots together")
    for token in tokenizer(token1, text1): print(token)
    print("---------- Token definition without dots together")
    for token in tokenizer(token2, text1): print(token)
    print("---------- Token definition for twitter")
    for token in tokenizer(token3, tweet): print(token)

    # ----------------- Run segmenter
    print("---------- Default segmenter")
    for sentence in sentence_segment1(segment1, tokenizer(token3, text2)): print(sentence)
    print()
    print("---------- Improved segmenter")
    prev = segment2.split(" ")[0]
    nxt = segment2.split(" ")[1] 
    for sentence in sentence_segment2(prev,nxt,tokenizer(token3, text2)): print(sentence)


if __name__ == '__main__':

  which = int(sys.argv[1])
  if which == 1:
    lecture1()

