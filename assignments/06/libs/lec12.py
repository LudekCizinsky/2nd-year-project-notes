import elmoformanylangs
from scipy.spatial.distance import cosine
import pickle
from termcolor import colored
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
LOGCL = "blue"


def getTopN(inputSent, model, tokzr, topn=1):

    maskId = tokzr.convert_tokens_to_ids(tokzr.mask_token)
    tokenIds = tokzr(inputSent).input_ids
    if maskId not in tokenIds:
        return 'please include ' + tokzr.mask_token + ' in your input'
    maskIndex = tokenIds.index(maskId)
    logits = model(torch.tensor([tokenIds])).logits
    return tokzr.convert_ids_to_tokens(torch.topk(logits, topn, dim=2).indices[0][maskIndex])


def lec12(): 

  print(colored('[Exercise 3: Subword tokenization]', "magenta" , attrs=['bold']))
  print(colored('[Loading pre-trained tokenizer]', LOGCL , attrs=['bold']))
  tokzr = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
  print("> Done!\n")

  print(colored('[Testing tokenizer]', LOGCL , attrs=['bold']))
  print(f"> {tokzr.tokenize('My name is Ludek Cizinsky and I am from Czech republic.')}")
  print(f"> {tokzr.tokenize('Moje jméno je Luděk Čižinský a jsem z České republiky.')}")
  print("> Done!\n")
  
  print(colored('[Loading another tokenizer]', LOGCL , attrs=['bold'])) 
  model = AutoModelForMaskedLM.from_pretrained('bert-base-cased')
  tokzr = AutoTokenizer.from_pretrained('bert-base-cased')
  print("> Done!\n")

  print(colored('[Testing analogies]', LOGCL , attrs=['bold']))
  print(f"> Copenhagen is capital of Denmark and London is capital of [MASK].")
  s1 = getTopN('Copenhagen is capital of Denmark and London is capital of [MASK].', model, tokzr, 5)
  print(f">> {s1}\n")

  print(f"> He is a boy and she is a [MASK].")
  s2 = getTopN('He is a boy and she is a [MASK].', model, tokzr, 5)
  print(f">> {s2}\n")
  
  print(f"> He is nice, but the other guy is even [MASK].")
  sy1 = getTopN('He is nice, but the other guy is even [MASK].', model, tokzr, 5)
  print(f">> {sy1}\n")
  
  print(f"> Dane is from Denmark and Czech is from [MASK].")
  sy2 = getTopN('Dane is from Denmark and Czech is from [MASK].', model, tokzr, 5)
  print(f">> {sy2}\n")


  print(colored('[Examining gender biases]', LOGCL , attrs=['bold']))
  print(f"> A truck driver stopped to take a break and to eat [MASK] lunch.")
  g1 = getTopN('> A truck driver stopped to take a break and to eat [MASK] lunch.', model, tokzr, 5)
  print(f">> {g1}\n")

  print(f"> He wore a blue color whereas she wore [MASK]")
  g2 = getTopN('> He wore a blue color whereas she wore a [MASK]', model, tokzr, 5)
  print(f">> {g2}\n")


  


