# About
To check this week's exercise, execute from this folder following command (linux based terminal):

```
./run.sh
```

Below, I leave some comments to the given exercises.

# Lecture 7
## 1. Word similarities
In this exercise, I was supposed to look up similarity of pair of words
according to the `google news` and `twitter` embeddings. Here are example
results:

```
> Pair: ('soccer', 'football')
>> Sim according to twitter emb.: 0.8725245594978333
>> Sim according to google news emb.: 0.731354832649231
>> Sim according to wordnet: 0.88

> Pair: ('ice', 'snow')
>> Sim according to twitter emb.: 0.6272193789482117
>> Sim according to google news emb.: 0.5391555428504944
>> Sim according to wordnet: 0.25

> Pair: ('sun', 'warm')
>> Sim according to twitter emb.: 0.6113071441650391
>> Sim according to google news emb.: 0.3973939120769501
>> Sim according to wordnet: 0.16666666666666666
```

Clearly, `twitter` embeddings seems to return better results. For the rest of
examples, run the command line.

## 2. Analogies
In this exercise, the task was to test how well we can extract different
analogies using the given word embeddings. In this case, `google news` seemed to
perform better as for example when finding the analogy for: `Denmark : Copenhagen x England - ?` I got the following results:

```
>>> Twitter [('Dublin', 0.791156530380249), ('London', 0.7881399989128113), ('Glasgow', 0.779718816280365), ('Edinburgh', 0.7671084403991699), ('Antwerp', 0.75947505235672), ('Aberystwyth', 0.7533636689186096), ('Birmingham', 0.7525423169136047), ('Melbourne', 0.7460692524909973), ('Leeds', 0.7449815273284912), ('Brighton', 0.7437199950218201)]

>>> Google News [('London', 0.4688379466533661), ('Twickenham', 0.4599652588367462), ('Headingley', 0.4539841413497925), ('Ashes_decider', 0.4507167339324951), ('Duncan_Fletcher', 0.4472123384475708), ('Englands', 0.4469376504421234), ('Leeds', 0.44511252641677856), ('Brian_Ashton', 0.4276672601699829), ('Andrew_Flintoff', 0.42736995220184326), ('Cardiff', 0.4255978465080261)]
```

For the rest, run the command line.

## 3., 4. and 5. Learning the word embeddings

In these tasks, the goal was to learn word embeddings from the given corpus. The
biggest challenge of course was first correctly tokenize the raw text. Next, it
was also important to decided on the architecture of the neural network. We only
had just one hidden layer which is actually not that much. Finally, it was also
important to consider which optimizer to use as well as the batch size as all of
these parameters can influence the overall performance. Again, use the command
line, or check the `scripts.py` to see the details of the training.

# Lecture 8
For this assignment, we are going to take a closer look at the ``Convolutional Neural Networks for Sentence Classification'' paper from Yoon Kim, which can be found [here](https://aclanthology.org/D14-1181/). 

Read the paper, and then answer the following questions:

a) Which type of pooling is applied after the convolution operations?

b) How many filters (kernels) are applied to the input sentence?

c) What dimensions do(es) the filter(s) have?

d) What do these dimensions correspond to?

e) The channels of a CNN are different ways of representing the input (e.g. for an RGB image: red, green, blue). Which different views does Kim use for sentence classification?

---

a) `Max over time pooling` which is just taking the maximum from the input vector `v` representing the feature map.

b) In total, `4 kernels` are applied since there are 4 resulting feature maps. In other words, each filter produces one feature map. 

c) Each filter is of `h \times k` dimensions where `h` is a size of the
window and `k` is the dimension of the input embeddings. For example,
Mikolov's word2vec has 300 dimensions. 

d) See the answer in c).

e) He adds another channels which are then trainable as opposed to the single
channel with pretrained word embeddings. 

