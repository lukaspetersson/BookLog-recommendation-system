print("LOG: Getting data")
import pandas as pd
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('dataset/ratings.csv')
train, test = train_test_split(dataset, test_size=0.2, random_state=42)
n_users = len(dataset.user_id.unique())
n_books = len(dataset.book_id.unique())

print("LOG: Creating Embedding Model")
from keras.layers import Input, Embedding, Flatten, Dot, Dense
from keras.models import Model, load_model
book_input = Input(shape=[1], name="Book-Input")
book_embedding = Embedding(n_books+1, 5, name="Book-Embedding")(book_input)
book_vec = Flatten(name="Flatten-Books")(book_embedding)
user_input = Input(shape=[1], name="User-Input")
user_embedding = Embedding(n_users+1, 5, name="User-Embedding")(user_input)
user_vec = Flatten(name="Flatten-Users")(user_embedding)
prod = Dot(name="Dot-Product", axes=1)([book_vec, user_vec])
#model = Model([user_input, book_input], prod)
#model.compile('adam', 'mean_squared_error')

##Training the model
#print("LOG: Training the model")
#history = model.fit([train.user_id, train.book_id], train.rating, epochs=10, verbose=1)
#model.save('regression_model.h5')

##Restoring Existing Model
print("LOG: Restoring existing model")
model = load_model("regression_model.h5")

#Visualizing embeddings
print("LOG: Visualizing embeddings")
# Extract embeddings
book_em = model.get_layer('Book-Embedding')
book_em_weights = book_em.get_weights()[0]

from sklearn.decomposition import PCA
import seaborn as sns
pca = PCA(n_components=2)
pca_result = pca.fit_transform(book_em_weights)
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1])

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tnse_results = tsne.fit_transform(book_em_weights)
sns.scatterplot(x=tnse_results[:,0], y=tnse_results[:,1])

#Making Recommendations
print("LOG: Making Recommendations")
# Creating dataset for making recommendations for the first user
import numpy as np
book_data = np.array(list(set(dataset.book_id)))
user = np.array([1 for i in range(len(book_data))])
predictions = model.predict([user, book_data])
predictions = np.array([a[0] for a in predictions])
recommended_book_ids = (-predictions).argsort()[:5]

print("LOG: Print Recommendations")
print(recommended_book_ids)
print(predictions[recommended_book_ids])

books = pd.read_csv("dataset/books.csv")
books.head()

print(books[books["id"].isin(recommended_book_ids)])

print("LOG: final")
