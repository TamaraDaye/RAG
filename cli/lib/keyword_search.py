from lib.search_utils import load_data, load_stop_words
import string
from nltk.stem import PorterStemmer
from collections import defaultdict
import os


# word stemmer
stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.index_path = None
        self.docmap_path = None

    def _add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)

    def get_document(self, term):
        return sorted(list(self.index[term]))

    def save(self):
        # os.makedirs()
        pass

    def build(self):
        movies = load_data()
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            tokens = tokenize_text(f"{movie['title']} {movie['description']}")
            self._add_document(doc_id, tokens)


def clean_text(text):
    text = text.lower()
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )  # creates a dict mapping for string chars to be replaced
    return text


def tokenize_text(text):  # Create tokens of movie title
    text = clean_text(text)
    stop_words = list(load_stop_words())
    tokens = [
        stemmer.stem(tok) for tok in text.split() if tok and tok not in stop_words
    ]
    return tokens


# Compare each token in the token in query to every token from movie title if match found return True
def matching_token(query_tok, movie_tok):
    for tok in query_tok:
        for token in movie_tok:
            if tok in token:
                return True

    return False


def search_command(query, n):
    res = []
    data = load_data()
    query_tokens = tokenize_text(query)

    for movie in data:
        movie_tokens = tokenize_text(movie["title"])
        if matching_token(query_tokens, movie_tokens):
            res.append(movie)
        if len(res) == n:
            break

    return res
