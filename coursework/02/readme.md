# About
To check this week's exercise, execute from this folder following command (linux based terminal):

```
./run.sh
```

Below, I leave some comments to the given exercises.

# Lecture 3
## 1. Annotation 
Annotate the dataset called `english.conll` which can be found in the pos-data
folder.

**Solution**
See the file called `luci.pos.ann.conll`.

## 2. Annotation quality

Complete the following tasks:
a) Calculate the accuracy between you and the other annotator, how often did you agree?
b) Now implement Cohen’s Kappa score, and calculate the Kappa for your annotation sample. In which range does you Kappa score fall?
c) Take a closer look at the cases where you disagreed with the other annotator; are these disagreements due to ambiguity, or are there mistakes in the annotation?

**Solution**
See the implementations in `scripts.py`. Kappa score is 0.8 so we have a moderate
agreement. I believe the disagreements are rather due to ambiguity. 

## 3. Words as features
In this exercise, turn the given data into unigram features.

**Solution**
See the implementation in `scripts.py`. I am also using my own lib which can be
found [here](https://github.com/LudekCizinsky/nano-learn).

# Lecture 4
## 4. Naive Bayes Classifier (pen and paper)
See the solution [here](https://deepnote.com/project/SYP--pD6EPwxQJqHOzuVtsapEA/%2Fnotebook.ipynb).

## 5. Naive Bayes with BOW in sklearn and 6. Discriminative Classifier with BOW
The task was to train NB and LR on one dataset and then evaluate on two dev
sets. The method of feature extraction was bow.

**Solution**
For both models I got similar results. On wookipedia dataset LR performed slightly worse than NB and for bulba dataset, it was vice versa. In addition, it is important to emphasize, than I used count of words andnot just simply binary features. In addition, there was a slight performance drop for bulba dataset, which is reasonable as the models were trained on wookipedia datasets. 

## 7. Character n-grams

As the name suggests, here the goal was to implement character ngrams and see if
there will be any diffence in models' performance.

**Solution**
I have not seen any signifficant changes in terms of model performance. Again,
for implementation see the `scripts.py`.

## Analysis

Analyze the models using confusion matrix and weights for each features.

**Solution**
From confusion matrix, we can see that the model has the biggest problem with
misclassyfying en and nl with da. Interestingly, each model has different top
5 words.

