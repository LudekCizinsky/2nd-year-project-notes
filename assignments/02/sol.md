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

