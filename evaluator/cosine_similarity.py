#%%
#Cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from retriever import db_records


def calculate_cosine_similarity(text1: str, text2: str) -> int:
    """
     Transforms the query text (text1) and each record (text2) of the dataset into a vector using a vectorizer, and then calculates and returns the cosine similarity score between the two vectors.
    Args:
        text1: Text from dataset
        text2: LLM Output

    Returns:
        Cosine similarity (int)
    """
    vectorizer = TfidfVectorizer(
        stop_words='english',
        norm='l2',
        ngram_range=(1, 2),
        sublinear_tf=True,
        analyzer='word'
    )
    tfidf = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return similarity[0][0]


#print(db_records)

#%%
from generator import call_llm_with_full_text
query = "define a rag store"
request_from_llm = call_llm_with_full_text(itext=query)
print(request_from_llm)

#%%
print(calculate_cosine_similarity(" ".join(db_records), request_from_llm))

#%%
#Enhanced similarity

import spacy
import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet
from collections import Counter
import numpy as np

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def preprocess_text(text):
    doc = nlp(text.lower())
    lemmatized_words = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        lemmatized_words.append(token.lemma_)
    return lemmatized_words

def expand_with_synonyms(words):
    expanded_words = words.copy()
    for word in words:
        expanded_words.extend(get_synonyms(word))
    return expanded_words

def calculate_enhanced_similarity(text1, text2) -> int:
    """
    Calculates the cosine similarity between the pre-processed and augmented synonym vectors
    Args:
        text1: Text from dataset
        text2: LLM Output

    Returns:
        Cosine similarity (int)
    """
    # Preprocess and tokenize texts
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)

    # Expand with synonyms
    words1_expanded = expand_with_synonyms(words1)
    words2_expanded = expand_with_synonyms(words2)

    # Count word frequencies
    freq1 = Counter(words1_expanded)
    freq2 = Counter(words2_expanded)

    # Create a set of all unique words
    unique_words = set(freq1.keys()).union(set(freq2.keys()))

    # Create frequency vectors
    vector1 = [freq1[word] for word in unique_words]
    vector2 = [freq2[word] for word in unique_words]

    # Convert lists to numpy arrays
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    # Calculate cosine similarity
    cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    return cosine_similarity


#%%
print(calculate_enhanced_similarity(" ".join(db_records), request_from_llm))

#%%

#Наша база данных db_records представляет из себя список (list) предложений. Мы хотим найти предложение из базы данных, которое наиболее соотносится с запросом пользователя. Для этого просто посмотрим в каком предложении содержится больше всего слов из запроса

query = "define a rag store"

def find_best_match_keyword_search(query, db_records):
    best_score = 0
    best_record = None
    best_common_keywords = None

    # Split the query into individual keywords
    query_keywords = set(query.lower().split())

    # Iterate through each record in db_records
    for record in db_records:
        # Split the record into keywords
        record_keywords = set(record.lower().split())

        # Calculate the number of common keywords
        common_keywords = query_keywords.intersection(record_keywords)
        current_score = len(common_keywords)

        # Update the best score and record if the current score is higher
        if current_score > best_score:
            best_score = current_score
            best_record = record
            best_common_keywords = common_keywords

    return best_score, best_record, best_common_keywords

best_keyword_score, best_matching_record, best_common_keywords = find_best_match_keyword_search(query, db_records)

print(f"Best Keyword Score: {best_keyword_score}")
print(f"Common keywords: {best_common_keywords}")
print(f"Best match sentence: {best_matching_record}")


#%%
#Простой RAG
augmented_input = f"{query}: {best_matching_record}"
print(call_llm_with_full_text(augmented_input))

#%%
#Продвинутый RAG: векторный поиск и поиск на основе индекса

def find_best_match(text_input, records) -> tuple[int, str]:
    """
    Векторный поиск
    Args:
        text_input:
        records:

    Returns:

    """
    best_score = 0
    best_record = None
    for record in records:
        current_score = calculate_cosine_similarity(text_input, record)
        if current_score > best_score:
            best_score = current_score
            best_record = record
    return best_score, best_record

best_similarity_score, best_matching_record = find_best_match(query, db_records)

print(query,": ", best_matching_record)
similarity_score = calculate_enhanced_similarity(query, best_matching_record)
print(f"Enhanced Similarity:, {similarity_score:.3f}")

#%%
augmented_input = f"{query}: {best_matching_record}"
llm_response = call_llm_with_full_text(augmented_input)
print(llm_response)


#%%

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def setup_vectorizer(records):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(records)
    return vectorizer, tfidf_matrix

def find_best_match(query, vectorizer, tfidf_matrix):
    query_tfidf = vectorizer.transform([query])
    similarities = cosine_similarity(query_tfidf, tfidf_matrix)
    best_index = similarities.argmax()  # Get the index of the highest similarity score
    best_score = similarities[0, best_index]
    return best_score, best_index

vectorizer, tfidf_matrix = setup_vectorizer(db_records)

best_similarity_score, best_index = find_best_match(query, vectorizer, tfidf_matrix)
best_matching_record = db_records[best_index]

print(query,": ", best_matching_record)
similarity_score = calculate_enhanced_similarity(query, best_matching_record)
print(f"Enhanced Similarity:, {similarity_score:.3f}")
