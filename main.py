
#katevasma dataset
import pandas as pd
data = pd.read_csv('wiki_movie_plots_deduped.csv')
print(data)

#katharisma tou dataset apo tis eggrafes stis opoies to plot einai keno
data = data.dropna(subset=['Plot'])

#apothikeush toy kathariismenou dataset
data.to_csv('cleaned_movies_dataset.csv', index=False)







