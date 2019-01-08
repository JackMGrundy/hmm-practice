import random
import sys
import string
import numpy as np

"""
Simple generation of Robert Frost like lines using a Markov Chain
"""

def removePunctuation(corpus):
    """
    Given a corpus string, remove punctuation
    """
    corpus = corpus.translate(str.maketrans('','',string.punctuation))
    return(corpus)


def sentences(corpus):
    """
    sentences: a formatting function not to be used outside of this problem. Remove punctuation from a 
            corpus and split based on new lines.
    """
    corpus = removePunctuation(corpus)
    corpus = corpus.replace("\n\n", "\n").split("\n")
    return(corpus)


def updateDict(d, k, v):
    """
    updateDict: check if key k exists in dictionary d. If it doesn't add it with an empty list value. 
                Append v to k's list. 

    Args:
        d (dict)
        k (immutable): a dictionary key
        v: a value

    Returns:
        Nothing
    """
    if k not in d:
        d[k] = []
    d[k].append(v)



def processText(corpus, o):
    """
    processText: take a corpous and order o indicating degree of n-grams. Returns a set of objects
                for generating new random text. 

    Args:
        corpus (string): string of text
        o (int [1: inf]): max number of ngrams to 

    Returns:
        s (dict: str -> [0, 1]): dictionary mapping word x to the P(A=X|A is the first word in a sentence)
        t (dict: (o dimensional tuple) -> {a: p(a), b: p(b), ...}) dictionary mapping o dimensional tuples of words
                to a dictionary pairing next words with probabilities. For example {(a, dog): {"barks": 0.1, "fetches":0.9}}
        d (list of dict): list of length o-1. Each element is a dictionary mapping single words to a dictionary pairing single 
                words with probabilities. The nth dictionary describes the probability various words being the n+1 word in a sentence 
                given the preceding word.
    """
    s = {}
    d = { x:{} for x in range(o-1) }
    t = {}

    corpus = sentences(corpus)
    for sentence in corpus:
        words = sentence.split(" ")

        # Count first words
        s[words[0]] = s.get(words[0], 0.) + 1

        # Distributions of second words
        for i in range(o-1):
            if i==len(words)-1: break # Not enough words
            wordA = words[i]
            wordB = words[i+1]
            updateDict(d[i], wordA, wordB)

        # distribution of next word given o preceding words
        for i in range(o, len(words)+1):
            if i==len(words):
                k = words[i-o:i]
                v = "END"
            else:
                k = words[i-o:i]
                v = words[i]
            k = tuple(k)
            updateDict(t, k, v)

    # Normalize start word probabilities
    numSentences = len(corpus)
    for k, v in s.items():
        s[k] = s[k]/numSentences

    return(s, t, d)



def generateText(corpus, n, o):
    """
    generateText: takes in an example corpus and produces n output strings similar to the input text

    Args:
        corpus (string): a string of ASCII text
        n (int): a number of sentences to generate
        o (int): dimension of n-grams to consider while generating text

    Returns:
        res (list): a list of n sentences similar to the corpus

    """
    s, t, d = processText(corpus, o)

    # Produce n sentences
    for j in range(n):
        res = []
        
        # First word
        firstWord = np.random.choice(np.array(list(s.keys())), 1, replace=True, p=np.array(list(s.values())))[0]
        res.append(firstWord)

        # 2 to o words
        nextWord = firstWord
        for i in range(o-1):
            nextWord = random.choice(d[i][nextWord])
            res.append(nextWord.lower())

        # o to end words
        while nextWord != "END":
            nextWords = tuple(res[len(res)-o:])
            # For larger n-grams, it's likely we haven't seen that combination
            if nextWords not in t.keys(): break
            nextWord = random.choice(t[tuple(nextWords)])
            res.append(nextWord.lower())
        
        print(" ".join(res[0:len(res)-1]))


if __name__=="__main__":
    corpus = open("robert_frost.txt").read()
    generateText(corpus, 4, 2)
# EOF