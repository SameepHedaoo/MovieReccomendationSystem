import numpy as np
import pandas as pd
import difflib  #for spelling mistakes 
from sklearn.feature_extraction.text import TfidfVectorizer  #convert textual data into numerical
from sklearn.metrics.pairwise import cosine_similarity  #for similarity

# Loading data from csv to pandas
movies_data = pd.read_csv(r'C:\Users\Sameep Hedaoo\Desktop\Py\Movie Recommendation\movies.csv')
# print(movies_data.head())
# print(movies_data.head())

#Selecting relevant features for recommendation
selected_features = ['genres','keywords','tagline','cast','director']
# print(selected_features)

#Replacing the null valuess with null string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

#combining all the 5 selected features,storing all the features together

combined_features = movies_data['genres']+movies_data['keywords']+movies_data['tagline']+movies_data['cast']+movies_data['director']

#converting the text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

#getting the similarity scores using cosine Similiarity
similarity = cosine_similarity(feature_vectors)
# print(similarity.shape)

#getting the movie name from the user
movie_name = input("Enter the movie name:")

#creating a list of all the movies given in the dataset
list_of_titles = movies_data['title'].tolist()
# print(list_of_titles)

#finding the close match for the movie name given by the user
find_close_match = difflib.get_close_matches(movie_name,list_of_titles)
print(find_close_match)
close_match = find_close_match[0]
print(close_match)

#finding the index of title of movie
index_of_movie = movies_data[movies_data.title == close_match]['index'].values[0]
# print(index_of_movie)

#getting a list of similar movies
similarity_score = list(enumerate(similarity[index_of_movie]))
# print(similarity_score)

#sorting the movies based on their similarity score
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1],reverse = True)
# print(sorted_similar_movies)
i = 1
#print the name of similar movies based on index
for movie in sorted_similar_movies:
  
  index = movie[0] 
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if(i<=10):
    print(i,'.',title_from_index)
    i+=1

