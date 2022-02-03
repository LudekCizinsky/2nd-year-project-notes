# Lecture 1
To see the scripts results simply run:

```
python scripts 1
```

##  Regular expressions

### Task 1
Write a regular expression (regex or pattern) that matches any of the following words: `cat, sat, mat`.

**Solution**
```
[acmst]{3}
```

### Task 2
Write a regular expression that matches numbers.

**Solution**

```
[0-9\.,]+
```

### Task 3
Expand the previous solution to match Danish prices indications.

```
[0-9\.,]+(( kr)|( DKK))?
```

## Tokenization

### Task 1
The tokenizer clearly makes a few mistakes. Where?

**Solution**

- That the sign `'` is not considered as a separate token and instead it is connected with a word.
- We missed `!`, `,`, `:`, `;`, `()`, `-`

### Task 2

Write a tokenizer to correctly tokenize the text.

**Solution**

```
 |[.]{3}|([\w]+-[\w]+)|[\w]+|[.?!,();:—']
```

### Task 3
Should one separate 'm, 'll, n't, possessives, and other forms of contractions from the word? Implement a tokenizer that separates these, and attaches the ' to the latter part of the contraction.

**Solution**
See scripts file.

### Task 4
Should elipsis (the last word) be considered as three .s or one ...? Design a regular expression for both solutions.

**Solution**
See the scripts the file.

## Twitter tokenization

### Task 1
As you might imagine, tokenizing tweets differs from standard tokenization. There are 'rules' on what specific elements of a tweet might be (mentions, hashtags, links), and how they are tokenized. The goal of this exercise is not to create a bullet-proof Twitter tokenizer but to understand tokenization in a different domain. In the next exercises, we will focus on the followingtweet:
```
tweet = "@robv New vids coming tomorrow #excited_as_a_child, can't w8!!"
```

What does 'correctly' mean, when it comes to Twitter tokenization? What is the correct tokenization of the tweet above?

**Solution**
I would say that there are two main parts:
- handle correctly user handles
- handle correctly hash tags as separate tokens

Therefore correct tokenization of the above tweet would be:

```
[@robv, New, vids, coming, tomorrow, #excited_as_a_child, can, 't, w8, !, !]
```

### Task 2
Try your tokenizer from the previous exercise (Question 4). Which cases are going wrong? Make sure your tokenizer handles the above tweet correctly.

**Solution**
It is not capable of handling correctly user and hashtag. See the fix in scripts.


### Task 3 - 4
Will your tokenizer correctly tokenize emojis? Think of at least one other example where your tokenizer will behave incorrectly.

**Solution**
It will not recognize emojis as this was not part of the regex definition of the
token. For example, it would not correctly recognize digit such as "1,000".


## Sentence segmentation
Sentence segmentation is not a trivial task either. There might be some cases where your simple sentence segmentation wouldn't work properly. First, make sure you understand the following sentence segmentation code used in the lecture:

```py
import re

def sentence_segment(match_regex, tokens):
    """
    Splits a sequence of tokens into sentences, splitting wherever the given matching regular expression
    matches.

    Parameters
    ----------
    match_regex the regular expression that defines at which token to split.
    tokens the input sequence of string tokens.

    Returns
    -------
    a list of token lists, where each inner list represents a sentence.

    >>> tokens = ['the','man','eats','.','She', 'sleeps', '.']
    >>> sentence_segment(re.compile('\.'), tokens)
    [['the', 'man', 'eats', '.'], ['She', 'sleeps', '.']]
    """
    current = []
    sentences = [current]
    for tok in tokens:
        current.append(tok)
        w would you deal with all URLs effectively?
        if match_regex.match(tok):
            current = []
            sentences.append(current)
    if not sentences[-1]:
        sentences.pop(-1)
    return sentences
```

In the following code, there is a variable `text` containing a small text and A regular expression-based segmenter:
```py
text = """
Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch is the longest official one-word placename in U.K. Isn't that weird? I mean, someone took the effort to really make this name as complicated as possible, huh?! Of course, U.S.A. also has its own record in the longest name, albeit a bit shorter... This record belongs to the place called Chargoggagoggmanchauggagoggchaubunagungamaugg. There's so many wonderful little details one can find out while browsing http://www.wikipedia.org during their Ph.D. or an M.Sc.
"""
token = re.compile('Mr.|[\w\']+|[.?]+')

tokens = token.findall(text)
sentences = sentence_segment(re.compile('\.'), tokens)
for sentence in sentences:
    print(sentence)
```

### Task 1
Improve the segmenter so that it segments the text in the way you think it is correct.

**Solution**
See the scripts file.

### Task 2
How would you deal with all urls effectivelly?

**Solution**
I would just used some solution that I would find on the Internet, such as this [one](https://mathiasbynens.be/demo/url-regex).

### Task 3
Can you think of other problematic cases not covered in the example?

**Solution**
Yes, for example if we would be dealing with a scientific text where there are
mathematical symbols present.

# Lecture 2
## Linux Command Line for NLP: Conll Format

### Task 1
Search in the `da_arto.conll` file (in the assingment1 directory) for first names. You can assume that first names always have the label B-PER, and that the string "B-PER" does not occur in the first column.

**Solution**
We can use the following command:
```
grep -n 'B-PER' da_arto.conll
```

### Task 2
How many names occur in the data?

**Solution**
We can use the following command:
```
grep -c 'B-PER' da_arto.conll
```

And we get `96`. If we only want unique names, we can can use:
```
grep 'B-PER' da_arto.conll | sort | uniq | wc -l
```
and we get `87`.

### Task 3
How can we make sure that we do not match the string "B-PER" occuring in the first column?

**Solution**
We can use `awk` command to select second column and then match the given
pattern in that column:
```
awk -F\t '$2=="B-PER"' da_arto.conll
```

### Task 4
How can we clean away the labels, so that we have only a list of names left? (hint: pipe the result of the previous command into a split)

**Solution**
We can again use `awk` for that:
```
awk -F\t '$2=="B-PER"' da_arto.conll | awk '{print $1}'
```

### Task 5
How many of the names you found start with an uppercased character?

**Solution**

There is in total 49 unique names starting with uppercase letter:
```
awk -F\t '$2=="B-PER"' da_arto.conll | awk '{print $1}' | grep -E "^[A-Z]" | sort | uniq | wc -l
```

