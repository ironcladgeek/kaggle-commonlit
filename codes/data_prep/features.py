import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import spacy
import textstat

nlp = spacy.load("en_core_web_sm")


def num_words(text):
    return textstat.lexicon_count(text)

def num_sentences(text):
    doc = nlp(text)
    return sum([1 for sent in doc.sents])

def avg_sentences_len(text):
    return len(text) / num_sentences(text)

def avg_num_syllables_per_word(text):
    return textstat.syllable_count(text) / num_words(text)

def avg_num_words_per_sentence(text):
    return num_words(text) / num_sentences(text)

def avg_words_len(text):
    return len(text) / num_words(text)

def flesch_reading(text):
    return textstat.flesch_reading_ease(text)

def flesch_kincaid(text):
    return textstat.flesch_kincaid_grade(text)

def automated_readability(text):
    return textstat.automated_readability_index(text)

def dale_chall_readability(text):
    return textstat.dale_chall_readability_score(text)


def naive_features(self,
             dataset,
             text_col_name,
             text_len=True,
             avg_sentences_len=True,
             num_sentences=True,
             num_words=True,
             avg_num_words_per_sentence=True,
             avg_words_len=True,
             avg_num_syllables_per_word=True,
             avg_num_complex_words_per_sentence=True,
             avg_words_frequency=True,
             flesch_reading_ease=True,
             flesch_kincaid_grade=True,
             automated_readability_index=True,
             dale_chall_readability_score=True):
    pass
