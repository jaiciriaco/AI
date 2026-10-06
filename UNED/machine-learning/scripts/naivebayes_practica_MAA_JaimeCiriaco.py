# Naive Bayes copiando la estructura de MachineLearningMastery implementando ucimlrepo, pandas, y añadiendo
# los conceptos solicitados por el enunciado: Log Probabilities, Nominal Attributes, Laplace Smoothing


import math
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


# Función que carga Iris (id=53) con ucimlrepo y lo devuelve como dataset como lista de filas
def load_iris_dataset():
    iris = fetch_ucirepo(id=53)
    X = iris.data.features.copy()
    y = iris.data.targets.copy().iloc[:, 0].copy()
    df = X.copy()
    df["__class__"] = y
    return df.values.tolist()


# Función que carga Bank (id=222) con ucimlrepo y aplica la limpieza solicitada en el enunciado
def load_bank_dataset_clean():
    bank = fetch_ucirepo(id=222)
    X = bank.data.features.copy()
    y = bank.data.targets.copy().iloc[:, 0].copy()

    # eliminamos columnas 8 y 15
    drop_names = []
    if X.shape[1] > 8:
        drop_names.append(X.columns[8])
    if X.shape[1] > 15:
        drop_names.append(X.columns[15])
    if len(drop_names) > 0:
        X = X.drop(columns=drop_names)

    # y eliminamos también filas con missing en columna 1 y 3 (sólo si existen)
    check_names = []
    if X.shape[1] > 1:
        check_names.append(X.columns[1])
    if X.shape[1] > 3:
        check_names.append(X.columns[3])

    if len(check_names) > 0:
        mask = X[check_names].notna().all(axis=1)
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)

    print("(bank) Missing total tras limpieza:", int(X.isna().sum().sum()))

    df = X.copy()
    df["__class__"] = y
    return df.values.tolist()


# Divide el dataset en train/test sin sklearn (Podríamos haber usado sklearn.model_selection.train_test_split)
def train_test_split(dataset, test_size=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(dataset))
    rng.shuffle(idx)

    n_test = int(len(dataset) * test_size)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    train = []
    test = []

    i = 0
    while i < len(train_idx):
        train.append(dataset[int(train_idx[i])])
        i += 1

    j = 0
    while j < len(test_idx):
        test.append(dataset[int(test_idx[j])])
        j += 1

    return train, test


# Separa el dataset por valores de clase, devuelve un diccionario
def separate_by_class(dataset):
    separated = dict()
    i = 0
    while i < len(dataset):
        vector = dataset[i]
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = list()
        separated[class_value].append(vector)
        i += 1
    return separated


# Detecta que una columna es numérica o categórica
def is_numeric_column(values):
    i = 0
    while i < len(values):
        v = values[i]
        if v is None:
            i += 1
            continue
        if isinstance(v, float) or isinstance(v, int) or isinstance(v, np.floating) or isinstance(v, np.integer):
            return True
        else:
            return False
    return False


# Calcula the mean of a list of numbers
def mean(numbers):
    return float(sum(numbers)) / float(len(numbers))


# Calcula la desviación estándar de una lista de números
def stdev(numbers):
    avg = mean(numbers)
    var_sum = 0.0
    i = 0
    while i < len(numbers):
        var_sum = var_sum + (numbers[i] - avg) ** 2
        i += 1
    variance = var_sum / float(len(numbers))
    s = math.sqrt(variance)
    if s == 0.0:
        s = 1e-9
    return s


# Si una columna es numérica -> (mean, stdev) o si es categórica -> conteos y valores únicos
def summarize_column(column, alpha=1.0):
    if is_numeric_column(column):
        nums = []
        i = 0
        while i < len(column):
            nums.append(float(column[i]))
            i += 1
        return {"type": "numeric", "mean": mean(nums), "stdev": stdev(nums)}
    else:
        counts = {}
        uniques = {}
        i = 0
        while i < len(column):
            v = column[i]
            if v not in counts:
                counts[v] = 0
            counts[v] = counts[v] + 1
            uniques[v] = 1
            i += 1
        return {"type": "categorical", "counts": counts, "K": len(uniques), "alpha": float(alpha)}


# Calcular sumarios por columna
def summarize_dataset(dataset, alpha=1.0):
    cols = list(zip(*dataset))
    summaries = []

    # ignoramos la última columna
    i = 0
    while i < len(cols) - 1:
        col = list(cols[i])
        summaries.append(summarize_column(col, alpha=alpha))
        i += 1

    return summaries


# Separar dataset por clase y calcular estadísticas para cada fila
def summarize_by_class(dataset, alpha=1.0):
    separated = separate_by_class(dataset)
    summaries = dict()

    for class_value in separated:
        rows = separated[class_value]
        summaries[class_value] = summarize_dataset(rows, alpha=alpha)

    return summaries


# Extensión "Log Probabilities"
def log_gaussian_probability(x, mean_, stdev_):
    var = stdev_ * stdev_
    a = -math.log(math.sqrt(2.0 * math.pi) * stdev_)
    b = -((x - mean_) ** 2) / (2.0 * var)
    return a + b


# Probabilidad categórica con Laplace, en log
def log_categorical_probability(value, counts_dict, class_count, K, alpha):
    count_v = 0
    if value in counts_dict:
        count_v = counts_dict[value]

    prob = (count_v + alpha) / (class_count + alpha * K)
    return math.log(prob)


# Construye un modelo con summaries + class_counts + log_priors
def build_model(train, alpha=1.0):
    model = {}
    summaries = summarize_by_class(train, alpha=alpha)
    separated = separate_by_class(train)

    class_counts = {}
    for c in separated:
        class_counts[c] = len(separated[c])

    total = len(train)
    log_priors = {}
    for c in class_counts:
        log_priors[c] = math.log(class_counts[c] / total)

    model["summaries"] = summaries
    model["class_counts"] = class_counts
    model["log_priors"] = log_priors
    model["alpha"] = float(alpha)
    return model


# Calcula puntuaciones (log-posteriors) para la fila dada
def calculate_log_scores(model, row):
    scores = {}
    summaries = model["summaries"]
    class_counts = model["class_counts"]
    log_priors = model["log_priors"]

    for class_value in summaries:
        score = log_priors[class_value]

        class_count = class_counts[class_value]
        class_summaries = summaries[class_value]

        i = 0
        while i < len(class_summaries):
            info = class_summaries[i]
            x = row[i]

            if info["type"] == "numeric":
                score = score + \
                    log_gaussian_probability(
                        float(x), info["mean"], info["stdev"])
            else:
                score = score + \
                    log_categorical_probability(
                        x, info["counts"], class_count, info["K"], model["alpha"])

            i += 1

        scores[class_value] = score

    return scores


# Predice la clase para la fila dada
def predict(model, row):
    scores = calculate_log_scores(model, row)

    best_label = None
    best_score = -1e300

    for class_value in scores:
        if best_label is None or scores[class_value] > best_score:
            best_score = scores[class_value]
            best_label = class_value

    return best_label


# Recibe las precisiones de una lista de filas
def get_predictions(model, test):
    predictions = []
    i = 0
    while i < len(test):
        row = test[i]
        yhat = predict(model, row)
        predictions.append(yhat)
        i += 1
    return predictions


# Calcula el porcentaje de acierto
def accuracy_metric(actual, predicted):
    correct = 0
    i = 0
    while i < len(actual):
        if actual[i] == predicted[i]:
            correct = correct + 1
        i += 1
    return correct / float(len(actual))


# Evaluación el algoritmo y muestreo de matriz de confusión con pandas por consola
def evaluate_algorithm(dataset, test_size=0.2, seed=0, alpha=1.0):
    train, test = train_test_split(dataset, test_size=test_size, seed=seed)

    model = build_model(train, alpha=alpha)

    actual = []
    predicted = []

    i = 0
    while i < len(test):
        row = test[i]
        actual.append(row[-1])
        predicted.append(predict(model, row))
        i += 1

    acc = accuracy_metric(actual, predicted)

    print("Accuracy:", round(acc, 4))
    print("Matriz de confusión:")
    print(pd.crosstab(pd.Series(actual, name="Real"),
                      pd.Series(predicted, name="Predecido"),
                      dropna=False))

    return acc


def main():
    iris = load_iris_dataset()
    print("\nIRIS (id=53)")
    evaluate_algorithm(iris, test_size=0.2, seed=42, alpha=1.0)

    bank = load_bank_dataset_clean()
    print("\nBANK (id=222)")
    evaluate_algorithm(bank, test_size=0.2, seed=42, alpha=1.0)


if __name__ == "__main__":
    main()


# RESULTADO :

# En el dataset Iris se obtiene una accuracy de 0.9333. Este valor es bastante alto,
# y además era esperable, pues todas las variables del conjunto de datos son
# numéricas y las tres clases que se toman en cuanta están bien separadas.
# El uso de Naive Bayes Gaussiano se adapta bien a este problema,
# aunque se observanalgunos errores de clasificación entre la versicolor y la virginica,
# que son las clases más similares.

# En el dataset Bank Marketing se obtiene una accuracy de 0.8737. Este problema
# es más complejo que Iris por lo que vemos, ya que incluye un mayor número de ejemplos y una
# combinación de variables numéricas y categóricas. Sin embargo, a todo esto, el modelo
# funciona correctamente gracias a que trabajamos con Log Probabilities y de la corrección
# de Laplace, que nos quitan de problemas de probabilidades nulas y de underflow.
# La matriz de confusión muestra que el modelo clasificó correctamente la
# mayoría de los casos de la clase mayoritaria ("no"), mientras que comete
# más errores en la clase minoritaria ("yes")
