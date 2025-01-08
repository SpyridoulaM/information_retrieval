#katevasma dataset
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('wiki_movie_plots_deduped.csv')
data = data.head(1000)
stemmer = PorterStemmer()

#katharisma tou dataset apo tis eggrafes stis opoies to plot einai keno
data = data.dropna(subset=['Plot'])

#apothikeush toy katharismenou dataset
data.to_csv('cleaned_movies_dataset.csv', index=False)

stopwords = nltk.corpus.stopwords.words("english")

def preprocess_text_with_stemming(text):
    #afairesi eidikwn xaraktirwn
    text = ''.join([char for char in text if char not in string.punctuation])  
    #tokenize se mikra 
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    #afairesi stopwords
    tokens = [token for token in tokens if token not in stopwords]
    #stemming
    tokens = [stemmer.stem(token) for token in tokens]
    #afairesi pollaplwn kenwn
    text = " ".join(tokens)
    return text

data['Processed_Plot'] = data['Plot'].apply(preprocess_text_with_stemming)

# Εμφάνιση των πρώτων γραμμών για επιβεβαίωση
# print(data.head())

#apothikeush toy katharismenou dataset
data.to_csv('preprocessed_movies_dataset_with_stemming.csv', index=False)

# Δημιουργία δομής ανεστραμμένου ευρετηρίου
inverted_index = defaultdict(list)

# Κατασκευή του ευρετηρίου
for idx, row in data.iterrows():
    doc_id = idx  # Το ID του εγγράφου (μπορεί να είναι η γραμμή)
    words = row['Processed_Plot'].split()  # Διάσπαση κειμένου σε λέξεις
    for word in words:
        if doc_id not in inverted_index[word]:  # Αποφυγή διπλών εγγραφών
            inverted_index[word].append(doc_id)

# print(dict(list(inverted_index.items())[:10]))  # Εμφάνιση των 10 πρώτων λέξεων
#fortwsh toy eurititiou se arxeio
with open('inverted_index.json', 'w') as f:
    json.dump(inverted_index, f)

#
with open('inverted_index.json', 'r') as f:
    inverted_index = json.load(f)

# Μηχανή αναζήτησης με CLI
def boolean_search(query, inverted_index):
    # Καθαρισμός και tokenization του ερωτήματος
    query = query.lower()  # Μετατροπή σε μικρά γράμματα
    tokens = word_tokenize(query)  # Διαίρεση σε tokens (λέξεις)
    tokens = [token for token in tokens if token not in stopwords]
    # Εφαρμογή stemming σε κάθε λέξη
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    result_set = set()

    # Αναζήτηση για AND, OR, NOT
    if "and" in stemmed_tokens:
        terms = [term for term in stemmed_tokens if term != "and"]
        result_set = set(inverted_index.get(terms[0], []))
        for term in terms[1:]:
            result_set &= set(inverted_index.get(term, []))  # Διατομή για AND
    elif "or" in stemmed_tokens:
        terms = [term for term in stemmed_tokens if term != "or"]
        for term in terms:
            result_set |= set(inverted_index.get(term, []))  # Ένωση για OR
    elif "not" in stemmed_tokens:
        terms = [term for term in stemmed_tokens if term != "not"]
        result_set = set(inverted_index.keys()) - set(inverted_index.get(terms[0], []))  # Διαφορά για NOT
    else:
        result_set = set(inverted_index.get(stemmed_tokens[0], []))  # Απλή αναζήτηση

    return result_set

# TF-IDF Search Function
def tfidf_search(query, data):
    # Συνδυασμός των κειμένων του dataset με το ερώτημα
    documents = data['Processed_Plot'].tolist()
    documents.append(preprocess_text_with_stemming(query))

    # Υπολογισμός των TF-IDF διανυσμάτων
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Υπολογισμός της ομοιότητας συνημιτόνου
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # Ταξινόμηση των εγγράφων βάσει ομοιότητας
    related_docs_indices = cosine_similarities.argsort()[::-1]

    # Επιστροφή των δεικτών των σχετικών εγγράφων και των βαθμολογιών τους
    return [(index, cosine_similarities[index]) for index in related_docs_indices if cosine_similarities[index] > 0]

# Απλή διεπαφή
print("Μηχανή Αναζήτησης (CLI)")
search_method = input("Επιλέξτε μέθοδο αναζήτησης ('boolean' ή 'tfidf'): ").strip().lower()

while True:
    query = input("Δώστε το ερώτημα (ή 'exit' για έξοδο): ")
    if query.lower() == "exit":
        print("Έξοδος από τη μηχανή αναζήτησης.")
        break

    if search_method == 'boolean':
        results = boolean_search(query, inverted_index)
        print(f"Βρέθηκαν {len(results)} σχετικά έγγραφα: {results}")
    elif search_method == 'tfidf':
        results = tfidf_search(query, data)
        print(f"Βρέθηκαν {len(results)} σχετικά έγγραφα:")
        for idx, score in results:
            print(f"Έγγραφο ID: {idx}, Βαθμολογία ομοιότητας: {score:.4f}")
    else:
        print("Μη έγκυρη μέθοδος αναζήτησης. Παρακαλώ επιλέξτε 'boolean' ή 'tfidf'.")
        break
