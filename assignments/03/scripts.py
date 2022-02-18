import sys
from collections import defaultdict
import numpy as np
import codecs

# -------- General code
def read_conll_file(file_name): 
  """
  Code from: Rob van der Goot
  Read in conll file
  
  :param file_name: path to read from
  :yields: list of words and labels for each sentence
  """
  current_words = []
  current_tags = []

  for line in codecs.open(file_name, encoding='utf-8'):
      line = line.strip()

      if line:
          if line[0] == '#':
              continue # skip comments
          tok = line.split('\t')
          if len(tok) == 2:
            word = tok[0]
            tag = tok[1]
          else:
            word = tok[0]
            tag = "masked"

          current_words.append(word)
          current_tags.append(tag)
      else:
          if current_words:  # skip empty lines
              yield((current_words, current_tags))
          current_words = []
          current_tags = []

  # check for last one
  if current_tags != [] and not raw:
      yield((current_words, current_tags))

def load_data(filename):

  X, y = [], []
  for words, labels in read_conll_file(filename):
    X.append(words)
    y.append(labels)
  return X, y

def get_yes_no(text):
  
  while True:
    ans = input(text) 
    if ans in ["y", "n"]:
      return ans

# -------- Lecture 5 detailed code
class PosTagger:

  """Part of speech tagger predicts tags for the given sequence of words.

  This part of speech tagger implements Hidden Markov Model where
  the decoding part is solved by Viterbi algorithm.

  Parameters
  ----------
  lp_sm : float, optional
    Laplace smoothing value

  Attributes
  ----------
  em : np.ndarray
    Emission matrix with dimensions: |T| x |V|
  tm : np.ndarray
    Transition matrix.
  pi : 
    Initial distribution.
  mt : dict
    Map tag to index in matrix.
  mw : dict
    Map word to index in matrix.
  T : list
    List with tags
  n : int
    Size of T.
  V : list
    Vocabulary where the last item is "OOV" - out of vocabulary token
  m : int
    Size of V.
  """

  def __init__(self, lp_sm=.1):
    self.lp_sm = lp_sm
    self.em = None 
    self.tm = None
    self.pi = None 
    self.mt = None
    self.mw = None
    self.T = None
    self.n = None 
    self.V = None
    self.m = None

  def fit(self, X, y):
    """
    Computes/retrieves:
    - transition matrix
    - emission matrix
    - initial distribution pi
    - vocab and tags
    Using these components, you can compute
    the most likely sequence of tags (hidden states)
    using Viterbi algorithm.

    Parameters
    ----------
    X : list
      Nested list where each inner list contains words/tokens
    y : list
      Nested list where each inner list contains gold pos labels
    """

    cnt = defaultdict(int) # store counts
    V, T = set(), set() # Vocabulary, POS tags
    
    # Compute the necessary counts to construct the matrices
    pi = defaultdict(int)
    total = 0
    for words, labels in zip(X, y):
      n = len(labels)
      for i in range(n):
        w, t = words[i], labels[i] # word, pos tag
        cnt[(t, w,)] += 1 # for emission
        if i < (n-1):
          t2 = labels[i + 1]
          cnt[(t, t2,)] += 1 # for transition
        cnt[t] += 1 # for both emission and transition
        V.add(w) # new word to vocab
        T.add(t) # new tag to tag set

      # to compute start + tag
      pi[labels[0]] += 1
      total += 1
    
    # Determine core properties - size of vocab and tags 
    self.T, self.V = list(T), list(V) + ["OOV"]
    self.n, self.m = len(self.T), len(self.V)

    # Construct corresponding matrices
    self.tm = np.empty((self.n, self.n))
    self.em = np.empty((self.n, self.m))
    for i in range(self.n):
      t = self.T[i]
      # Emission
      for j in range(self.m):
        w = self.V[j]
        cnt_w_t = cnt[(t, w,)]
        if w != "OOV":
          self.em[i, j] = (cnt_w_t + .1)/(cnt[t] + self.m*.1)
        else:
          self.em[i, j] = 1/self.m

      # Transition
      for j in range(self.n):
        t2 = self.T[j]    
        self.tm[i, j] = (cnt[(t, t2,)] + .1)/(cnt[t] + self.n*.1)

    # Add mappings
    self.mt = {self.T[i]: i for i in range(self.n)}
    self.mw = {self.V[i]: i for i in range(self.m)}
 
    # Compute pi
    self.pi = np.array([(pi[self.T[i]] + .1)/(total + self.n*.1) for i in range(self.n)])


  def _tmprob(self, fr, to):
    i = self.mt[fr]
    j = self.mt[to]
    return self.tm[i, j]
  
  def _wposprob(self, t, w):
    i = self.mt[t]
    j = self.mw[w]
    return self.em[i, j]

  def _step(self, x, vtb, wi, ti):
    res = []
    for tii in range(self.n):
      tm = self.tm[tii, ti]
      em = self.em[ti, x[wi]]
      r = vtb[tii, wi-1]*tm*em 
      res.append(r)
    
    return np.array(res)

  def predict(self, X):
    """Predicts the most likely sequence of tags.

    Parameters
    ----------
    X : list
      Nested list where each inner list contains words/tokens

    Returns
    -------
    y_hat : list
      Nested list where each inner list contains predicted pos labels.
    """
    
    res = []
    for x in X:
      
      # Remapping of words to indices
      x = [self.mw.get(w, self.mw["OOV"]) for w in x]

      # Initialization step 
      wl = len(x) 
      vtb = np.empty((self.n, wl))
      backp = np.empty((self.n, wl))
      for i in range(self.n):
        vtb[i, 0] = self.pi[i]*self.em[i, x[0]]
        backp[i, 0] = None
       
      # Build the most optimal path
      for wi in range(1, wl):
        for ti in range(self.n):
          mu = self._step(x, vtb, wi, ti)   
          backp[ti, wi] = i = np.argmax(mu)
          vtb[ti, wi] = mu[i]
      
      # Find the best path
      y_hat = [int(np.argmax(vtb[:, wl - 1]))]
      i = y_hat[0]
      for j in range(wl - 1, 0, -1):
        i = int(backp[i, j])
        y_hat.append(i)
      y_hat = y_hat[::-1] # reverse the order of the list

      # Remapping of indices to predicted labels
      y_hat = [self.T[i] for i in y_hat]
      res.append(y_hat)
 
    return res

  def evaluate(self, y_true, y_pred, how="accuracy_score"):
    """Evaluate performance of the tagger.

    Parameters
    ----------
    y_true : list
      2d list where each inner item represents gold pos labels.
    y_pred : list
      2d list where each inner item represents predicted pos labels.
    how : str, optional
      Which metric you want to use for evaluation.

    Returns
    -------
    score : float
      Requested metric.
    """

    if how == "accuracy_score":
      total = 0
      correct = 0
      for tr, pr in zip(y_true, y_pred):
        total += len(tr)
        correct += sum(np.array(tr) == np.array(pr)) 
      score = correct/total
    else:
      raise ValueError(f"Metric {how} is not implemented.")

    return score


def train_tagger():

  print("Loading training data...")
  X_train, y_train = load_data("pos-data/da_arto-train.conll") 
  print("> Done!\n")
   
  print("Training tagger...")
  tg = PosTagger()
  tg.fit(X_train, y_train)
  print("> Done!\n")
  
  return tg
 
def check_trained_tagger(tg):
 
  print("Check of trained tagger:")
  # - Transition
  ratio = tg._tmprob(fr="ADJ", to="NOUN")/tg._tmprob(fr="NOUN", to="ADJ") 
  print("> ADJ NOUN is " + str(round(ratio, 4)) + "x more probable than NOUN ADJ\n") 
  # - Emission
  w = "hvor"
  print(f"> Emmision probs for the word {w} sorted in decreasing order:")
  tmp = []
  for t in tg.T:
    p = tg._wposprob(t, w)
    tmp.append((t, w, p)) 
  tmp = sorted(tmp, key=lambda x: x[2], reverse=True)
  for v in tmp:
    t, w, p = v
    print(f">> {round(p, 4)} {t} {w}")
  print()


def check_dev_perf(tg):

  print("Loading dev data...")
  X_dev, y_dev = load_data("pos-data/da_arto-dev.conll")
  print("> Done!\n")

  print("Accuracy on dev set:")
  y_hat = tg.predict(X_dev)
  print(f"> {tg.evaluate(y_dev, y_hat)}")

def run_tagger_on_test(tg):

  print("Loading test data...")
  X_test, _ = load_data("pos-data/da_arto-test-masked.conll")
  print("> Done!\n")

  print("Predicting labels for test data...")
  y_hat = tg.predict(X_test)
  out = ""
  for y in y_hat:
    out += "\n".join(y) + "\n\n"
  with open("y.out", "w") as f:
    f.write(out) 
  print(f"> See the predicted labels in the file called y.out")


# -------- High level code to run the given tasks in each lecture
def lecture5():

  ans = get_yes_no("-- Do you want to train POS tagger? ")
  if ans == "y":
    tg = train_tagger()
  else:
    return

  ans = get_yes_no("-- Do you want to check if the tagger was trained as expected? ")
  if ans == "y":
    check_trained_tagger(tg)

  ans = get_yes_no("-- Do you want to check tagger's dev performance? ")
  if ans == "y":
    check_dev_perf(tg)

  ans = get_yes_no("-- Do you want to run tagger on test data? ")
  if ans == "y":
    run_tagger_on_test(tg)


def lecture6():
  pass


if __name__ == "__main__":
  
  which = sys.argv[1]

  if which == 'l5':
    lecture5()
  else:
    lecture6()

