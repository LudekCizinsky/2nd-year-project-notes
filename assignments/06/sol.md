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

Using the following code:

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

# Lecture 12

## Exercise 3: Subword tokenization
I gave as an input the following sentences:

```py
print(f"> {tokzr.tokenize('My name is Ludek Cizinsky and I am from Czech republic.')}")
print(f"> {tokzr.tokenize('Moje jméno je Luděk Čižinský a jsem z České republiky.')}")
```

And obtained the following tokenization:

```
[Testing tokenizer]
> ['My', 'name', 'is', 'Lu', '##dek', 'Ci', '##zin', '##sky', 'and', 'I', 'am', 'from', 'Czech', 'republic', '.']
> ['Mo', '##je', 'jméno', 'je', 'Lu', '##dě', '##k', 'Č', '##i', '##ži', '##nský', 'a', 'jsem', 'z', 'České', 'republiky', '.']
> Done!
```

I would use it works well for both English and Czech, in both languages the only
problem is my name. For the word analogies, I got the following results:

```
[Testing analogies]
> Copenhagen is capital of Denmark and London is capital of [MASK].
>> ['England', 'Britain', 'London', 'Europe', 'UK']

> He is a boy and she is a [MASK].
>> ['girl', 'boy', 'woman', 'man', 'child']

> He is nice, but the other guy is even [MASK].
>> ['worse', 'better', 'hotter', 'harder', 'younger']

> Dane is from Denmark and Czech is from [MASK].
>> ['Slovakia', 'Austria', 'Czechoslovakia', 'Germany', 'Poland']
```

As can be seen it works well even when I included comma after the mask. Finally,
here is a couple of examples examining gender biases:

```
[Examining gender biases]
> A truck driver stopped to take a break and to eat [MASK] lunch.
>> ['his', 'some', 'their', 'for', 'my']

> He wore a blue color whereas she wore [MASK]
>> ['.', 'red', ';', 'green', 'white']
```

We can see that there are certain biases present.

## Exercise 4: Train a BERT model

The shape of the output should be a 1 dimensional tensor with the first
dimension being equal to the `number of classes` present within the training
dataset. To train the model, we can run the following from the folder of this
exercise:

```
python libs/bert/bert-topic.py libs/bert/topic-data/train.txt libs/bert/topic-data/dev.txt
```

To run the training on `HPC`, we first need to move the data to the cluster. We
first `ssh` to the cluster and:

```
ssh luci@hpc.itu.dk
mkdir bert
mkdir bert/topic-data
mkdir bert/log
```

We then `cd` locally into the `libs` folder and run:

```
scp bert/bert-topic.py luci@hpc.itu.dk:/home/luci/bert/main.py
scp bert/myutils.py luci@hpc.itu.dk:/home/luci/bert/myutils.py
scp bert/topic-data/train.txt luci@hpc.itu.dk:/home/luci/bert/topic-data/train.txt
scp bert/topic-data/dev.txt luci@hpc.itu.dk:/home/luci/bert/topic-data/dev.txt
scp bert/topic-data/test.txt luci@hpc.itu.dk:/home/luci/bert/topic-data/test.txt
```

Finally, we need to create a job to be run on the cluster. We can define the job
to be run as follows (execute from the bert folder):

```bash
#!/bin/bash

#SBATCH --job-name=simple-gpu    # Job name
#SBATCH --output=job.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=8        # Schedule 8 cores (includes hyperthreading)
#SBATCH --gres=gpu               # Schedule a GPU, or more with gpu:2 etc
#SBATCH --time=01:00:00          # Run time (hh:mm:ss) - run for one hour max

echo "[Running on $(hostname)]"
python3 main.py
```

Now, we need to make sure that we have loaded all the needed packages. Since we
are using `torch` we need to load `Anaconda` and activate virtual environment:

```
module load Anaconda3/2021.05
conda init bash
bash
conda create -n bert pytorch cudatoolkit=11.3 -c pytorch
conda activate bert
```
