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
- We missed `!`, `,`, `:`, `;`,  `()`

### Task 2

Write a tokenizer to correctly tokenize the text.

**Solution**

```
[\w-]+|[.?!,();:]
```

