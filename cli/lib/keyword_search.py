from typing import Dict, List
from lib.search_utils import load_data, load_stop_words, CACHE_PATH
import string
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
import pickle
import math

# word stemmer
stemmer = PorterStemmer()


class InvertedIndex:
    """
    An inverted index that maps tokens -> docs
    This class manages the inverted index by creating and index and a document map

    Attributes:
        index (dict[str, set]): This is the index map that stores each token and the set of doc_ids they occur
        docmap(dict[int, doc]): This is the document map it stores each documents id and the corresponding document
        index_path(Path): The file path to the index serialized data
        docmap_path(Path): The file path to the document serialized data
        term_frequency(dict[int, Counter]):a frequency counter of words for each document
    """

    def __init__(self):
        """Initializes inverted index object

        Note: Constructor method for InvertedIndex

        Args:
            self: new instance from creating from calling the class

        Returns:
            An inverted index
        """
        self.index = defaultdict(set)  # Maps token to doc
        self.docmap = {}  # Maps doc_id to entire doc
        self.index_path = CACHE_PATH / "index.pkl"  # file for index
        self.docmap_path = CACHE_PATH / "docmap.pkl"  # file for docs
        self.term_frequency_path = CACHE_PATH / "term_frequencies.pkl"
        self.term_frequency = defaultdict(Counter)

    def _add_document(self, doc_id: int, token: str):
        """Adds token and inserts the document id into its set of ids

        Note:
            This function takes a list of tokens for each token it adds the specified doc id to its set
            indicates each token in the input exists in the document with that doc_id

        Args:
            doc_id(int): unique id of document in database
            tokens(List[str]): The list of tokens obtained by cleaning and tokenizing the document

        Returns:
            None
        """
        tokens = tokenize_text(token)

        for token in tokens:
            self.index[token].add(doc_id)

        self.term_frequency[doc_id].update(tokens)

    def get_document(self, term: str) -> List[int]:
        """This obtains the list of docs(doc_ids) the input index occurs in

        Args:
            term(str): This is a token

        Returns:
            returns the set of document ids this particular token occurs
        """
        return sorted(list(self.index[term]))

    def get_tf(self, doc_id: int, term: str):
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Too much args")
        return self.term_frequency[doc_id].get(token[0], 0)

    def get_idf(self, term: str):
        token = tokenize_text(term)

        if len(token) != 1:
            raise ValueError("Too much args")

        total_doc_count = len(self.docmap)
        term_match_count = len(self.index[token[0]])
        return math.log((total_doc_count + 1) / (term_match_count + 1))

    def get_tf_idf(self, doc_id: int, term: str):
        token = tokenize_text(term)
        return self.get_idf(token[0]) * self.get_tf(doc_id, term)

    def get_bm25idf(self, term: str) -> float:
        token = tokenize_text(term)

        if len(token) != 1:
            raise ValueError("Too much args")

        N = len(self.docmap)
        df = len(self.index[token[0]])

        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def build(self):
        """Build the index from the input database"""
        movies = load_data()
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            tokens = f"{movie['title']} {movie['description']}"
            self._add_document(doc_id, tokens)

    def save(self):
        """
        Serializes the data to the binary and stores in cache
        """
        if not self.index or not self.docmap:
            print("Index or Docmap is missing. Rebuilding....")
            self.build()
        CACHE_PATH.mkdir(parents=True, exist_ok=True)
        artifacts = [
            (self.index_path, self.index),
            (self.docmap_path, self.docmap),
            (self.term_frequency_path, self.term_frequency),
        ]

        for file_path, data_object in artifacts:
            with open(file_path, "wb") as f:
                pickle.dump(data_object, f)

    def load(self):
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(self.term_frequency_path, "rb") as f:
            self.term_frequency = pickle.load(f)


def clean_text(text: str) -> str:
    """Cleans the input text by normalizing it making it all lower case and removing punctuation

    Args:
        text(str): This could be the query text or document text

    Returns:
        The newly cleaned text
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    return text


def tokenize_text(text: str) -> List[str]:  # Create tokens of movie title
    """Creates tokens from input text

    Args:
        text(str): Input text to be tokenized the text is cleaned first before tokenization

    Returns:
        A list of tokens

    """
    text = clean_text(text)
    stop_words = list(load_stop_words())
    tokens = [
        stemmer.stem(tok) for tok in text.split() if tok and tok not in stop_words
    ]
    return tokens


def search_command(query: str, n: int = 5) -> List[Dict[str, int | str]]:  # pyright: ignore[]
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    res = []
    seen = set()

    for qt in query_tokens:
        matching_doc = idx.get_document(qt)

        for doc_id in matching_doc:
            if doc_id in seen:
                continue

            seen.add(doc_id)
            doc = idx.docmap.get(doc_id)
            res.append({"id": doc_id, "title": doc["title"]})  # pyright: ignore[]
            if len(res) == n:
                return res


idx = InvertedIndex()
idx.load()


def build_command():
    idx.build()
    idx.save()


def tf_command(doc_id: int, term: str) -> int:
    return idx.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    return idx.get_idf(term)


def tf_idf_command(doc_id: int, term: str) -> float:
    tf_idf = idx.get_tf_idf(doc_id, term)
    return tf_idf


def bm25_idf_command(term: str) -> float:
    bm25_idf = idx.get_bm25idf(term)
    return bm25_idf
