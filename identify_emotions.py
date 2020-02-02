import operator
import spacy
from spacy.tokens import Token
from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import typing

nlp = spacy.load("en_core_web_md")


class EmoParser:
    """ Class encapsulates all configurations for the identification process.

        heuristics (dict) - a collection of settings for frequency and strength of
            identified emotion. Values were found by tweaking on examples.
        nltk_mapping (dict) - mapping from spacy to WordNet Part-of-speech constants.
        emotions (list) - emotions from Plutchik's wheel of emotions
        emotions_range (dict) - mapping emotions to valence. Commonly, Anger,
            Anticipation, Joy, and Trust are positive in valence, while Fear,
            Surprise, Sadness, and Disgust are negative in valence. Which
            is not a case here.
        remap_emotions (dict) - mapping opposite emotions.
        emotions_nlp (dict) - mapping emotions to spacy Token objects.

        emotions_scores (DataFrame) - DataFrame read from csv, that contains
            emotional scores for dictionary. Sirocco Opinion Extraction
            Framework - Model Files are used
            https://github.com/datancoffee/sirocco-mo/tree/master/src/main/resources/csdict
        id_columns (list) - indexing columns for emotions_scores.
        vader (SentimentIntensityAnalyzer) - get a sentiment intensity score.

        

    """
    def __init__(self, emotions_dict_file):
        self.heuristics = {
            "occur": 0.1,
            "emotion": 0.3,
        }
        # In WordNet, a satellite adjective--more broadly referred to as
        # a satellite synset--is more of a semantic label used elsewhere
        # in WordNet than a special part-of-speech in nltk.
        # adjectives are subcategorized into 'head' and 'satellite' synsets
        # within an adjective clutser
        # { Part-of-speech constants
        # https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html
        self.nltk_mapping = {
            "ADJ": "a",
            "ADJ_SAT": "s",
            "ADV": "r",
            "NOUN": "n",
            "VERB": "v",
        }

        self._set_up_emotions_radar()
        self._set_up_sentiment_analysis(emotions_dict_file)

    def _set_up_emotions_radar(self) -> None:
        self.emotions = [
            "joy",
            "trust",
            "fear",
            "surprise",
            "sadness",
            "disgust",
            "anger",
            "anticipation",
        ]
        self.emotions_range = {
            "pos": ["joy", "trust", "anticipation", "surprise"],
            "neg": ["anger", "sadness", "fear", "disgust"],
        }
        self.emotions_nlp = {emotion: nlp(emotion) for emotion in self.emotions}
        self.remap_emotions = {
            "joy": "sadness",
            "trust": "disgust",
            "fear": "anger",
            "surprise": "anticipation",
        }
        self.remap_emotions.update(dict((v, k) for k, v in self.remap_emotions.items()))

    def _set_up_sentiment_analysis(self, emotions_dict_file: str) -> None:
        self.vader = SentimentIntensityAnalyzer()

        emotions_scores = pd.read_csv(emotions_dict_file)
        self.id_columns = ["lemma[key]", "pos[key]"]
        emotions_scores = emotions_scores.rename(columns={"acceptance": "trust"})
        self.emotions_scores = emotions_scores[
            [
                col
                for col in emotions_scores.columns
                if col in self.emotions + self.id_columns
            ]
        ]

def _negate(token: Token) -> bool:
    num_of_negations = [1 for child in token.head.children if child.dep_ == "neg"]

    if len(num_of_negations) % 2 == 0:
        return False
    else:
        return True

def _get_dictionary_score(emp: EmoParser, valence: str, word: str, pos: str) -> typing.Dict[str, float]:
    local_scores = emp.emotions_scores[
            [
                col
                for col in emp.emotions_scores
                if col in emp.emotions_range[valence] + emp.id_columns
            ]
        ]
    relevant_scores = local_scores.loc[local_scores["lemma[key]"] == word]

    if len(relevant_scores) > 1:
            relevant_scores = relevant_scores.loc[
                relevant_scores["pos[key]"] == emp.nltk_mapping[pos]
            ]

    if len(relevant_scores) == 1:
        (word_emotions,) = relevant_scores.to_dict(orient="records")
        return {
            k: v
            for k, v in word_emotions.items()
            if k in emp.emotions_range[valence]
        }
    else:
        return {}

def _find_similar_emotion(emp: EmoParser, valence: str, token: Token) -> typing.Dict[str, float]:
    local_nlp = {
            k: v
            for k, v in emp.emotions_nlp.items()
            if k in emp.emotions_range[valence]
        }

    return {
                emotion: round(token.similarity(e_nlp), 2)
                for emotion, e_nlp in local_nlp.items()
            }

def _update_doc_emotions(emp: EmoParser, token: Token, doc_emotions) -> None:
    word = token.lemma_
    pos = token.pos_

    # looks like other PoS have less impact on score
    if pos not in emp.nltk_mapping:
        return

    polarity = emp.vader.polarity_scores(word)

    # skip words without sentiments
    if polarity["neu"]:
        return

    # select only emotions of prevalent valence
    (valence,) = [k for k, v in polarity.items() if v and k != "compound"]

    word_emotions = _get_dictionary_score(emp, valence, word, pos)

    # values is a view object, it is neither derived from a built-in list type,
    # nor has any common ancestor type with a list in the inheritance hierarchy
    # so mypy considers it a type mismatch.
    list_values: typing.Iterable = word_emotions.values()

    if word_emotions and sum(list_values):
        word_final_score = word_emotions
    else:
        word_final_score = _find_similar_emotion(emp, valence, token)

    if _negate(token):
        remapped_score = {
            emp.remap_emotions[emotion]: word_final_score.pop(emotion)
            for emotion in list(word_final_score)
        }
        word_final_score = remapped_score

    

    for emotion, res in word_final_score.items():
        doc_emotions[emotion].append(res)


def score_emotions(emotions_dict_file: str, text: str) -> typing.Dict[str, float]:
    """Get score for emotions from Plutchik's wheel of emotions for given text
    
    Args:
        emotions_dict_file (str) - path to Sirocco Opinion Extraction Framework Model Files
        text (text) - a text for which emotion scores should be obtained
    Returns:
        (dict) - emotions from Plutchik's wheel of emotions that were identified
    """
    emp = EmoParser(emotions_dict_file)

    doc = nlp(text)
    doc_emotions: typing.Dict[str, typing.List[float]] = {k: [] for k in emp.emotions}

    for token in doc:
        _update_doc_emotions(emp, token, doc_emotions)

    emotional_words = len([v for val in doc_emotions.values() for v in val])

    doc_score: typing.Dict[str, float] = {}
    for emotion in doc_emotions:
        results_list = doc_emotions[emotion]
        emotion_mean = np.mean(results_list)
        occur = round(len(results_list) / emotional_words, 2)
        if (
            occur > emp.heuristics["occur"]
            or np.mean(results_list) > emp.heuristics["emotion"]
        ):
            doc_score[emotion] = round(emotion_mean, 2)

    return doc_score


