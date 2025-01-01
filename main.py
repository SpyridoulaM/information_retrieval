#katevasma dataset
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
import string
data = pd.read_csv('wiki_movie_plots_deduped.csv')
data = data.head(1000)
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
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
    
    tokens =[stemmer.stem(token) for token in tokens]

    #afairesi pollaplwn kenwn
    text = " ".join(tokens)
    return text

data['Processed_Plot'] = data['Plot'].apply(preprocess_text_with_stemming)


# Εμφάνιση των πρώτων γραμμών για επιβεβαίωση
#print(data.head())


data.to_csv('preprocessed_movies_dataset_with_stemming.csv', index=False)

from collections import defaultdict

# Δημιουργία δομής ανεστραμμένου ευρετηρίου
inverted_index = defaultdict(list)

# Κατασκευή του ευρετηρίου
for idx, row in data.iterrows():
    doc_id = idx  # Το ID του εγγράφου (μπορεί να είναι η γραμμή)
    words = row['Processed_Plot'].split()  # Διάσπαση κειμένου σε λέξεις
    
    for word in words:
        if doc_id not in inverted_index[word]:  # Αποφυγή διπλών εγγραφών
            inverted_index[word].append(doc_id)

#print(dict(list(inverted_index.items())[:10]))  # Εμφάνιση των 10 πρώτων λέξεων
#fortwsh toy eurititiou se arxeio
import json

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


# Απλή διεπαφή
print("Μηχανή Αναζήτησης (CLI)")
while True:
    query = input("Δώστε το ερώτημα (ή 'exit' για έξοδο): ")
    if query.lower() == "exit":
        print("Έξοδος από τη μηχανή αναζήτησης.")
        break
    results = boolean_search(query, inverted_index)
    print(f"Βρέθηκαν {len(results)} σχετικά έγγραφα: {results}")



