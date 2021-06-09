import pandas as pd
from tqdm import tqdm
import spacy
import textstat

tqdm.pandas()


class NaiveFeatures:
    # TODO: provide doc string
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def _num_words(text):
        return textstat.lexicon_count(text)

    @staticmethod
    def _num_digits(text):
        return sum(c.isnumeric() for c in text)

    def _num_sentences(self, text):
        doc = self.nlp(text)
        return sum([1 for sent in doc.sents])

    def _avg_sentences_len(self, text):
        return len(text) / self._num_sentences(text)

    def _avg_num_syllables_per_word(self, text):
        return textstat.syllable_count(text) / self._num_words(text)

    def _avg_num_words_per_sentence(self, text):
        return self._num_words(text) / self._num_sentences(text)

    def _avg_words_len(self, text):
        return len(text) / self._num_words(text)

    @staticmethod
    def _flesch_reading(text):
        return textstat.flesch_reading_ease(text)

    @staticmethod
    def _flesch_kincaid(text):
        return textstat.flesch_kincaid_grade(text)

    @staticmethod
    def _automated_readability(text):
        return textstat.automated_readability_index(text)

    @staticmethod
    def _dale_chall_readability(text):
        return textstat.dale_chall_readability_score(text)


    def generate(self,
                 dataset,
                 text_col_name,
                 text_len=True,
                 avg_sentences_len=True,
                 num_sentences=True,
                 num_words=True,
                 num_digits=True,
                 avg_num_words_per_sentence=True,
                 avg_words_len=True,
                 avg_num_syllables_per_word=True,
                 flesch_reading_ease=True,
                 flesch_kincaid_grade=True,
                 automated_readability_index=True,
                 dale_chall_readability_score=True):

        # asserts
        assert type(dataset) == pd.core.frame.DataFrame
        assert type(text_col_name) == str
        assert text_col_name in dataset.columns

        df = dataset.copy()

        if text_len:
            print('Calculating text length ...')
            df['text_len'] = df[text_col_name].progress_apply(lambda x: len(x))
        if avg_sentences_len:
            print('\nCalculating average sentences length ...')
            df['avg_sentences_len'] = df[text_col_name].progress_apply(lambda x: self._avg_sentences_len(x))
        if num_sentences:
            print('\nCalculating number of sentences ...')
            df['num_sentences'] = df[text_col_name].progress_apply(lambda x: self._num_sentences(x))
        if num_words:
            print('\nCalculating number of words ...')
            df['num_words'] = df[text_col_name].progress_apply(lambda x: self._num_words(x))
        if num_digits:
            print('\nCalculating number of digits ...')
            df['num_digits'] = df[text_col_name].progress_apply(lambda x: self._num_digits(x))
        if avg_num_words_per_sentence:
            print('\nCalculating average number of words per sentence ...')
            df['avg_num_words_per_sentence'] = df[text_col_name].progress_apply(lambda x: self._avg_num_words_per_sentence(x))
        if avg_words_len:
            print('\nCalculating average words length ...')
            df['avg_words_len'] = df[text_col_name].progress_apply(lambda x: self._avg_words_len(x))
        if avg_num_syllables_per_word:
            print('\nCalculating average number of syllables per word ...')
            df['avg_num_syllables_per_word'] = df[text_col_name].progress_apply(lambda x: self._avg_num_syllables_per_word(x))
        if flesch_reading_ease:
            print('\nCalculating flesch reading ease score ...')
            df['flesch_reading_ease']= df[text_col_name].progress_apply(lambda x: self._flesch_reading(x))
        if flesch_kincaid_grade:
            print('\nCalculating flesch kincaid grade score ...')
            df['flesch_kincaid_grade'] = df[text_col_name].progress_apply(lambda x: self._flesch_kincaid(x))
        if automated_readability_index:
            print('\nCalculating automated readability index score ...')
            df['automated_readability_index'] = df[text_col_name].progress_apply(lambda x: self._automated_readability(x))
        if dale_chall_readability_score:
            print('\nCalculating dale chall readability score ...')
            df['dale_chall_readability_score'] = df[text_col_name].progress_apply(lambda x: self._dale_chall_readability(x))

        return df







