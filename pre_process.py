# encoding: utf-8 

import pandas as pd
import codecs
import sys
import spacy

def tokenize(input_file):
    token_lines = []
    for line in training_file:
        token_lines.append(line.split())
    return token_lines

def separate_syntagma_pairs(sentence_list):
    pairs = []
    for line in sentence_list:
        temp_list = []
        for item in line:
            temp_list.append(item.split("\\"))
        pairs.append(temp_list)
    return pairs

def pos_tagger(sentence_list):
    pos_tagged = []
    nlp = spacy.load('pt_core_news_sm')
    for sentence in sentence_list:
        pos_tagged_sentence = []
        for item in sentence:
            token = nlp(item[0])[0]
            item.insert(1, token.pos_)
            pos_tagged_sentence.append(item)
        pos_tagged.append(pos_tagged_sentence)
    return pos_tagged


if len(sys.argv) < 2:
    print("Usage: python pre_process.py training_set_name")
    sys.exit()

training_file = codecs.open(sys.argv[1], 'r', encoding='utf-8-sig')
token_lines = tokenize(training_file)
training_file.close()
pairs = separate_syntagma_pairs(token_lines)
pos_tagged = pos_tagger(pairs)
print(pos_tagged)

