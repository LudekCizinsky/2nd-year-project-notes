# About
To check this week's exercise, execute from this folder following command (linux based terminal):

```
chmod +x scripts.py
./run.sh
```

Below, I leave some comments to the given exercises.

# Lecture 11
## Exercise 1: Getting Elmo embeddings for target words
The answer in this exercise is that the `sentence 1` is more likely to have
a different sense than the second sentence since I got the following results:

```
[Comparing cosine distance]
2022-03-13 21:35:10,834 INFO: 1 batches, avg len: 11.0
> e0 vs e1: 0.16099363565444946
> e0 vs e2: 0.12716662883758545
```

And here is the code:

```py
t = 'laugh'
sent = [
    ['Amy', 'did', 'not', 'laugh', 'this', 'was', 'work', 'concentration', 'achievement'],
    ['She', 'began', 'to', 'laugh'],
    ['And', 'the', 'old_man', 'had', 'given', 'a', 'sly', 'and', 'wicked', 'laugh', 'and', 'said', 'Hell', 'yes']
]

e = elmoformanylangs.Embedder('libs/elmo.en/')

emb = e.sents2elmo(sent)
e0 = emb[0][sent[0].index(t)]
e1 = emb[1][sent[1].index(t)]
e2 = emb[2][sent[2].index(t)]

d1 = cosine(e0, e1)
d2 = cosine(e0, e2)
```

## Exercise 2: Word sense disambiguation

For this exercise, I got the following result:

```
[Loading pre-trained ELMO model]
> Done!

[Evaluation of the model using cosine distance]
> Accuracy: 0.64
```

```py
embeds_list, labels_list = pickle.load(open('libs/semcor/semcor_dev.elmo.pickle', 'rb')) 

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
```

