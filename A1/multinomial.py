#!/usr/bin/env python3
import argparse
import math

def main():
    #Input for training
    infile  = open("traindata.txt", "r")
    train_data = infile.read()
    infile = open("trainlabels.txt", "r")
    train_labels= infile.read()

    train_result = TrainMultinomialNB(train_labels, train_data)
    vocab = train_result[0]
    prior = train_result[1]
    condprob = train_result[2]

    infile  = open("traindata.txt", "r")
    test_data = infile.read()
    infile = open("trainlabels.txt", "r")
    test_labels= infile.read()

    output = []

    sentences = test_data.splitlines()
    for doc in sentences:
        apply_result = ApplyMultinomialNB(train_labels, vocab, prior, condprob, doc)
        output.append(apply_result)

    outfile = open("result.txt", "w")
    for item in output:
        outfile.write(str(item)+"\n")

def TrainMultinomialNB(train_labels, train_data):
    condprob = {}
    Tct = {}
    prior = [0, 0]
    num_classes = CountClasses(train_labels)
    Vocab = ExtractVocab(train_data)
    Num_docs = CountDocs(train_data)
    for c in range(num_classes):
        Nc = CountDocsInClass(train_labels,c)
        prior[c] = (Nc/Num_docs)
        textc = ConcatenateTextOfAllDocsInClass(train_labels, train_data, c)
        for term in Vocab:
            T = CountTokensOfTerm(textc, term)
            Tct.update({(term, c):(T)})
        for term in Vocab:
            result = Tct.get((term,c))
            condprob.update({(term,c):result+1/(len(textc)+len(Vocab))})
    return (Vocab, prior, condprob)


def ApplyMultinomialNB(train_labels, vocab, prior, condprob, doc):
    score = [0, 0]
    W = ExtractTokensFromDoc(vocab, doc)
    num_classes = CountClasses(train_labels)
    for c in range(num_classes):
        score[c] = math.log10(prior[c])
        for term in W:
            score[c] += math.log10(condprob.get((term,c)))
    return ArgMax(score)

def ArgMax(score):
    if score[0] > score[1]:
        return 0
    elif score[1] >= score[0]:
        return 1

def ExtractVocab(train_data):
    vocab = set()
    sentences = train_data.splitlines()
    for line in sentences:
        sentence_split = line.split()
        for word in sentence_split:
            vocab.add(word)
    return vocab

def CountDocs(train_data):
    num_lines = 0
    sentences = train_data.splitlines()
    for line in sentences:
        num_lines = num_lines+1
    return num_lines

def CountClasses(train_labels):
    classes = set()
    cl = train_labels.splitlines()
    for line in cl:
        classes.add(line)
    return len(classes)

def CountDocsInClass(train_labels,c):
    num_docs = 0
    x = train_labels.splitlines()
    for item in x:
        if int(item) == c:
            num_docs = num_docs + 1
    return num_docs

def ConcatenateTextOfAllDocsInClass(train_data, train_labels, c):
    text = []
    sentences = train_data.splitlines()
    cl = train_labels.splitlines()
    for l1, l2 in zip(sentences, cl):
        if int(l1) == c:
            sentence = l2.split()
            for word in sentence:
                text.append(word)
    return text

def CountTokensOfTerm(textc, term):
    term_count = 0
    for item in textc:
        if item == term:
            term_count = term_count + 1
    return term_count

def ExtractTokensFromDoc(vocab, doc):
    tokens = []
    words = doc.split()
    for word in words:
        if word in vocab:
            tokens.append(word)
    return tokens

if __name__ == "__main__":
    main()
