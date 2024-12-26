#katevasma dataset
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
import string
data = pd.read_csv('wiki_movie_plots_deduped.csv')
print(data)

#katharisma tou dataset apo tis eggrafes stis opoies to plot einai keno
data = data.dropna(subset=['Plot'])

#apothikeush toy katharismenou dataset
data.to_csv('cleaned_movies_dataset.csv', index=False)

stopwords = nltk.corpus.stopwords.words("english")

def preprocess_text(text):
    #afairesi eidikwn xaraktirwn
    text = ''.join([char for char in text if char not in string.punctuation])  

    #tokenize se mikra 
    text = text.lower() 
    tokens = nltk.word_tokenize(text)

    #afairesi stopwords
    tokens = [token for token in tokens if token not in stopwords]

    #afairesi pollaplwn kenwn
    text = " ".join(tokens)
    return text

data['Processed_Plot'] = data['Plot'].apply(preprocess_text)
print(data[['Plot', 'Processed_Plot']].head(6))
data.to_csv('preprocessed_movies_dataset.csv', index=False)
