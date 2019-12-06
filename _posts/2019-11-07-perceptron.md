---
title: "Machine Learning: Hindi-English-Code-Mixed-Stemmer
date: 2019-11-07
tags: [machine learning, data science, natural language processing, nlp]
header:
  image: "/images/banner.jpg"
excerpt: "Machine Learning, NLP, Stemmer"
---


# Hindi-English-Code-Mixed-Stemmer
An unsupervised stemmer for Natural Language Processing Tasks on Hinglish Language ( Hindi + English words )

#### Pretext:

English is widely acknowledged as the world’s most successful language. It has developed over the 20th century into a global lingua franca, the most widely used language on the internet, and the clear leader in education and research. But new competitors are emerging to displace English, especially in rich multilingual contexts such as India.

The language Hinglish involves a hybrid mixing of Hindi and English within conversations, individual sentences and even words. An example: “She was bhunno-ing the masala-s_ jub_ phone ki ghuntee bugee.” Translation: “She was frying the spices when the phone rang”. It is gaining popularity as a way of speaking that demonstrates you are modern, yet locally grounded.

India has the second largest English-speaking population in the world (at 125m), while many speak multiple languages. English fluency is socially prestigious and important for job success and upward mobility.

> Acquiring fluent English (or any language) requires rich and consistent language exposure. In India, this is largely limited to the urban upper classes. Together, these two factors – limited English access and the desirability of becoming an English speaker – could mean that communication styles which are more available to the masses, such as Hinglish, grow faster than English.


### What is stemming?
Stemming is the process of reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words known as a lemma. Stemming is important in natural language understanding (NLU) and natural language processing (NLP). Stemming is a part of linguistic studies in morphology and artificial intelligence (AI) information retrieval and extraction. Stemming and AI knowledge extract meaningful information from vast sources like big data or the Internet since additional forms of a word related to a subject may need to be searched to get the best results. Stemming is also a part of queries and Internet search engines.


## Code for a Hinglish Stemmer:
```
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import os
try:
    import nltk
except:
    print("First install NLTK using pip install nltk command")
    exit()
try:
    import gensim
    from gensim import corpora, models, similarities
except:
    print("First install Gensim using pip install nltk command")
    exit()


class Stemmer:
    w2vModel = None
    sensitivity = 10

    # constructor
    def __init__(self, modelLocation= "w2vModel", w2vModel = None):
        try:
            self.w2vModel = gensim.models.Word2Vec.load(modelLocation)
        except:
            print("Could not locate the w2vModel file in the directory : "+(modelLocation))
            print("Try to load the w2vModel and try again")



    #  ----------- Stemming functions -----------


    # takes a word and removes the repeated occurance of characters in that word
    # outputs word without repeat consecutive occurance of the word
    def RepetitionStemmer(self, word):
        # find repeted occurence of letters in a word
        # remove the occurence
        i=0
        newWord = ''
        while(i <len(word)):
            c = word[i]
            newWord+=c
            while(i<len(word) and word[i] == c):
                i=i+1

        return newWord

    # takes a word2vec model, word and nWords(to run most similar on - higher the better but slower)
    # output the list of words similar to that word ( including that word passed through repetition stemmer)
    def WordEmbeddingStemmer(self, w2vModel, word, nWords = 10):

        try:
            similarWordsList =[w2vModel.wv.most_similar(word, topn = nWords )[i][0] for i in range(10)]
        except:
            return self.RepetitionStemmer(word)

        word = self.RepetitionStemmer(word)

        outputList = []
        for similarWord in similarWordsList:
            stemmSimilarWord = self.RepetitionStemmer(similarWord)
            w0 = word
            w1 = word[:-1]
            sw0 = stemmSimilarWord
            sw1 = stemmSimilarWord[:-1]

            if (sw0 in w0) or( w0 in sw0) or (sw1 in w0) or (w1 in sw0):
                if(len(stemmSimilarWord)<len(word)):
                    outputList.append(stemmSimilarWord)
                else:
                    outputList.append(word)
        if len(outputList) == 0:
            outputList.append(word)
        return outputList[0]

    # stemmers
    def stemWord(self, word):
        return self.WordEmbeddingStemmer(self.w2vModel, word)

    def stemListOfWords(self, listOfWords):
        return [self.WordEmbeddingStemmer(self.w2vModel, word) for word in listOfWords]

    def stem2dListOfWords(self, listOfWords2d):
        output = []
        for sentenceOfWords in listOfWords2d:
            output.append([self.WordEmbeddingStemmer(self.w2vModel, word) for word in sentenceOfWords])
        return output
```


### Requirements
* Gensim
* NLTK

Usage:

```
import stemmer
```

#### Single Word Stemmer

```
myStemmer = stemmer.Stemmer()
output = myStemmer.stemWord("ladkaa")
```
> output : 'ladka'

#### List of Words Stemmer
```
output = myStemmer.stemListOfWords(["ladkii", "ladkaaaa", "firaaangii"])
```
> output: ['ladki', 'ladka', 'firangi']

#### 2D List of Words Stemmer
```    
output = myStemmer.stem2dListOfWords([["merii","merraa"], ["terii", "terraaa", "aaajjjaa"]])
```
> output: [['meri', 'mera'], ['teri', 'tera', 'aja']]


##### Use Credits:
Ashish Gupta
Github: www.github.com/ashishgupta1350
You are free to use and distribute this in anyway you like.
