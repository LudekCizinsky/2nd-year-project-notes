import re

text = """
'Curiouser and curiouser!' cried Alice (she was so much surprised, that for the moment she quite forgot how to speak good English); 'now I'm opening out like the largest telescope that ever was! Good-bye, feet!' (for when she looked down at her feet, they seemed to be almost out of sight, they were getting so far off). 'Oh, my poor little feet, I wonder who will put on your shoes and stockings for you now, dears? I'm sure I shan't be able! I shall be a great deal too far off to trouble myself about you: you must manage the best way you can; —but I must be kind to them,' thought Alice, 'or perhaps they won't walk the way I want to go! Let me see: I'll give them a new pair of boots every Christmas...'
"""

tweet = "@robv New vids coming tomorrow #excited_as_a_child, can't w8!!" 

def tokenizer(token, text):
      
    tokens = re.findall(token, text) 
    tmp = []
    skip = False
    for i in range(1, len(tokens) - 1):

        if not skip:
            prev, curr, nxt = tokens[i - 1], tokens[i], tokens[i + 1]
            prohibited = ["'", " "]
            is_relevant = prev not in prohibited and curr in ["'"] and nxt not in prohibited
            
            if is_relevant:
                tmp.append(curr + nxt)
                skip = True
            elif curr != " ":
                tmp.append(curr)
        else:
            skip = False
    
    tokens = [tokens[0]] + tmp + [tokens[-1]] 
    
    for token in tokens: print(token)

if __name__ == '__main__':

    # With 3 dots together
    token = " |[\.]{3}|[\w]+-[\w]+|[\w]+|[\.?!,();:—']"
    
    # Dots separetely
    token2 = " |[\w]+-[\w]+|[\w]+|[\.?!,();:—']" 

    # Twitter token definition
    token_twitter = " |#[\w]+|@[\w]+|[\w]+-[\w]+|[\w]+|[\.?!,();:—']"

    # Run tokenizer 
    tokenizer(token_twitter, text)

