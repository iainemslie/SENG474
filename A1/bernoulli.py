#!/usr/bin/env python3
import argparse
import math

def main():
    #Input for training
    infile  = open("traindata.txt", "r")
    train_data = infile.read()
    infile = open("trainlabels.txt", "r")
    train_labels= infile.read()
    #Input for applying
    infile = open("testdata.txt", "r")
    test_data= infile.read()


    item = TrainBernoulliNB(train_labels, train_data)
    #Get the values returned by the tuple from TrainBernoulliNB
    vocab = item[0]
    prior = item[1]
    condprob = item[2]

    #print(condprob.get(("cannot",1)))

    score = ApplyBernoulliNB(train_labels, vocab, prior, condprob, test_data)

def TrainBernoulliNB(train_labels, train_data):
    condprob = {}
    num_classes = CountClasses(train_labels)
    prior = [0, 0]
    vocab = ExtractVocab(train_data)
    num_docs = CountDocs(train_data)
    for c in range(num_classes):
        docs_in_class = CountDocsInClass(train_labels, c)
        prior[c] = docs_in_class/num_docs
        for term in vocab:
            Nct = CountDocsInClassContainingTerm(train_data, c, term)
            condprob.update({(term,c):((Nct+1)/(num_classes+2))})
    item = (vocab, prior, condprob)
    return item

def ApplyBernoulliNB(train_labels, vocab, prior, condprob, test_data):
    score = [0, 0]
    Vd = ExtractTermsFromDoc(train_labels, test_data)
    for c in range(len(prior)):
        score[c] = math.log10(prior[c])
        for term in vocab:
            if term in Vd:
                score[c] += math.log10(condprob.get((term,1)))
            else:
                if(condprob.get((term,1)) < 1):
                    score[c] += math.log10(1 - condprob.get((term,1)))
    return ArgMax(score)

#Puts each unique word from the training data into a set
def ExtractVocab(train_data):
    vocab = set()
    sentences = train_data.splitlines()
    for line in sentences:
        sentence_split = line.split()
        for word in sentence_split:
            vocab.add(word)
    return vocab

#Returns the number of unique lines in the training data
def CountDocs(train_data):
    num_lines = 0
    sentences = train_data.splitlines()
    for line in sentences:
        num_lines = num_lines+1
    return num_lines

def CountDocsInClass(train_labels,c):
    num_docs = 0
    x = train_labels.splitlines()
    for item in x:
        if int(item) == c:
            num_docs = num_docs + 1
    return num_docs

#Returns the number of different classes in the labels document
def CountClasses(train_labels):
    classes = set()
    cl = train_labels.splitlines()
    for line in cl:
        classes.add(line)
    return len(classes)

#Returns the number of words in the document matching the term
def CountDocsInClassContainingTerm(train_data, c, term):
    count = 0
    sentences = train_data.splitlines()
    for line in sentences:
        sentence_split = line.split()
        for word in sentence_split:
            if word == term:
                count = count+1
    return count

def ExtractTermsFromDoc(train_labels, test_data):
    vocab = set()
    sentences = test_data.splitlines()
    for line in sentences:
        sentence_split = line.split()
        for word in sentence_split:
            vocab.add(word)
    return vocab

def ArgMax(score):
    max = 0
    for c in range(len(score)):
        if score[c] > max:
            print (max)
            max = score[c]
    return max

if __name__ == "__main__":
    main()
