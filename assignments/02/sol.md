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
a) Exercise 4.1 from J&M

**Solution**
From the information given, we know that to estimate posterior probabilities for
each class we can write:

<img src="https://render.githubusercontent.com/render/math?math={\color{white}P(y = pos | x) ~ 0.09*0.07*0.29*0.04*0.08 = .0000058 // P(y = neg | x) ~ 0.16*0.06*0.06*0.15*0.11 = .000009
}">

For this reason we can conclude that our naive bayes classifier would predict
the `negative class`.

b) Exercise 4.2 from J&M

**Solution**
In general, we need to do the following two steps:
1. Compute prior probability for each class
2. Compute class conditional, i.e. $p(x|y)$ where `x` is the set of words from
   the given test sentence and `y` is given class.
3. We then multiply the results for respective classes together and predict the
   class with highest likelihood.

First, we start with class priors. These can be estimated by using frequency of the given class within the whole training dataset. Therefore, we can write the following:

$$
P(comedy) = 2/5 //
P(action) = 3/5
$$

Now, to estimate the class conditionals using the Naive Bayes and Laplace
smoothing, we need to determine the size of the vocabulary vector and also count
the number of words in training dataset for each class. The size of vocab vector
is:

```
len([fun, couple, love, fast, furious, shoot, fly]) =  7
```

Number of words for comedy is 9 and for comedy it is 11. Now, we can just
compute for each word the class conditional probability:

$$
P(fast | comedy) = 2/(9 + 7) //
P(couple | comedy) = 3/(9 + 7)
P(shoot | comedy) = 1/(9 + 7)
P(fly | comedy) = 2/(9 + 7)
$$

and similarly for action:

$$
P(fast | action) = 3/(9 + 11) //
P(couple | action) = 1/(9 + 11)
P(shoot | action) = 5/(9 + 11)
P(fly | action) = 2/(9 + 11)
$$ 

Therefore, we can write, that:

$$
P(comedy | test_text) ~ .00007 
P(action | test_text) ~ .00011
$$

Therefore the conclusion is that the predicted class would be `action`.

