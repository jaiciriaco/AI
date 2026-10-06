import pandas as pd

# Extraer las columnas de los géneros de películas que usaremos para el one-hot encoding
genre_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.genre"

# Leer los géneros desde u.genre
genres = pd.read_csv(
    genre_url,
    sep="|",
    header=None,
    names=["genre", "genre_id"],
    engine="python"
)

# Crear un diccionario de los géneros
genres_dict = dict(zip(genres["genre_id"], genres["genre"]))

print("Géneros disponibles:", list(genres_dict.values()), "\n")


# Extraer los datos de las películas
item_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"

# Leer los datos de las películas desde u.item
movies = pd.read_csv(
    item_url,
    sep="|",
    header=None,
    encoding="latin-1",
    engine="python"
)

# Seleccionar las columnas relevantes para el one-hot encoding
genre_cols = list(genres_dict.keys())
num_genres = len(genre_cols)

# Reorganizar las columnas para que el título esté primero seguido de los géneros
movies = movies[[1] + list(range(5, 5 + num_genres))]
movies.columns = ["title"] + list(genres_dict.values())

print("Primeras películas cargadas:")
print(movies.head(), "\n")

# Crear el DataFrame one-hot encoding a partir de los géneros
onehot = movies.set_index("title")

print("DataFrame one-hot creado con tamaño:")
# Mostramos todas las columnas porque sino no vemos el efecto del onehot
pd.set_option('display.max_columns', None)
print(onehot.shape)
print(onehot.head(), "\n")

# Guardar un csv a partir del one-hot encoding especificando la codificación utf-8 para poder abrirlo sin problemas
onehot.to_csv("onehot_movies.csv", index=True, encoding="utf-8")

# Mostramos el soporte de cada género
support = onehot.mean().sort_values(ascending=False)

print("Soporte de cada género:")
print(support, "\n")
