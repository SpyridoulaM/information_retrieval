

import pandas as pd
data = pd.read_csv('wiki_movie_plots_deduped.csv')
print(data)


data = data.dropna(subset=['Plot'])


data.to_csv('cleaned_movies_dataset.csv', index=False)







