import logging
import elmoformanylangs
from scipy.spatial.distance import cosine
import pickle
from termcolor import colored

logging.getLogger('allennlp.common.params').disabled = True
logging.getLogger('allennlp.nn.initializers').disabled = True
logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.WARNING)
logging.getLogger('urllib3.connectionpool').disabled = True

LOGCL = "blue"

def lec11(): 

  print(colored('[Exercise 1: Getting Elmo embeddings for target words]', "magenta" , attrs=['bold']))
  t = 'laugh'
  sent = [
      ['Amy', 'did', 'not', 'laugh', 'this', 'was', 'work', 'concentration', 'achievement'],
      ['She', 'began', 'to', 'laugh'],
      ['And', 'the', 'old_man', 'had', 'given', 'a', 'sly', 'and', 'wicked', 'laugh', 'and', 'said', 'Hell', 'yes']
  ]
  
  print(colored('[Loading pre-trained ELMO model]', LOGCL , attrs=['bold']))
  e = elmoformanylangs.Embedder('libs/elmo.en/')
  print("> Done!\n")

  print(colored('[Comparing cosine distance]', LOGCL , attrs=['bold']))
  emb = e.sents2elmo(sent)
  e0 = emb[0][sent[0].index(t)]
  e1 = emb[1][sent[1].index(t)]
  e2 = emb[2][sent[2].index(t)]
  
  d1 = cosine(e0, e1)
  d2 = cosine(e0, e2)
  print(f"> e0 vs e1: {d1}")
  print(f"> e0 vs e2: {d2}\n")

  print(colored('[Exercise 2: Word sense disambiguation]', "magenta" , attrs=['bold']))
  print(colored('[Loading pre-trained ELMO model]', LOGCL , attrs=['bold']))
  embeds_list, labels_list = pickle.load(open('libs/semcor/semcor_dev.elmo.pickle', 'rb')) 
  print("> Done!\n")

  print(colored('[Evaluation of the model using cosine distance]', LOGCL , attrs=['bold']))
  correct = 0
  for x, y in zip(embeds_list, labels_list):
    e1, e2, e3 = x
    d1 = cosine(e1, e2)
    d2 = cosine(e1, e3)
    
    if d1 < d2:
      yhat = '2'
    else:
      yhat = '1' 
    correct += int(yhat == y)
  res = correct/len(labels_list)
  print(f"> Accuracy: {res}\n")

