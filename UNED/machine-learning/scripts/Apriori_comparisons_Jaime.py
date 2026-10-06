import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# region Ejercicio anterior: Transformar un conjunto de datos como conjunto de transacciones

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

# endregion

# ------

# region Ejercicio: Generar reglas de asociación (taller)

# region 0) Funciones auxiliares

# Convierte segundos a algo entendible


def format_seconds(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.3f} s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f} min"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.2f} h"
    days = hours / 24
    return f"{days:.2f} días"


# Combinaciones C(n,k) con un bucle simple
def nCk(n, k):
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    num, den = 1, 1
    for i in range(1, k + 1):
        num *= (n - (k - i))
        den *= i
    return num // den


# Pasa frozenset / list a string "A, B, C"
def set_to_str(s):
    return ", ".join(sorted(list(s)))


# Imprime top-n en formato comentario (para pegarlo en la entrega)
def print_top_as_comment(df, title, n=20):
    print("# " + title)
    print("# " + df.head(n).to_string(index=False).replace("\n", "\n# "))
    print()

# endregion


# region 1) Apriori + association_rules

t0 = pd.Timestamp.now()

min_support = 1 / len(onehot)

frequent_itemsets = apriori(
    onehot.astype(bool),
    min_support=min_support,
    max_len=3,
    use_colnames=True
)


try:
    rules = association_rules(
        frequent_itemsets, metric="support", min_threshold=0.0)
except Exception:
    rules = association_rules(
        frequent_itemsets, metric="support", min_threshold=min_support)

t1 = pd.Timestamp.now()
elapsed_rules = (t1 - t0).total_seconds()

print(
    f"Tiempo (apriori + association_rules): {format_seconds(elapsed_rules)}\n")

rules = rules.copy()
rules["antecedents_str"] = rules["antecedents"].apply(set_to_str)
rules["consequents_str"] = rules["consequents"].apply(set_to_str)
rules["rule"] = rules["antecedents_str"] + " -> " + rules["consequents_str"]

# endregion


# region 2) Función rule_metrics

# Busca una regla concreta en el DF de rules y devuelve support/confidence/lift
def rule_metrics(antecedent_list, consequent_list, onehot_dataset):
    antecedent_fs = frozenset(antecedent_list)
    consequent_fs = frozenset(consequent_list)

    row = rules[(rules["antecedents"] == antecedent_fs) &
                (rules["consequents"] == consequent_fs)]

    if row.empty:
        return {
            "rule": f"{', '.join(antecedent_list)} -> {', '.join(consequent_list)}",
            "support": 0.0,
            "confidence": 0.0,
            "lift": 0.0
        }

    r = row.iloc[0]
    return {
        "rule": r["rule"],
        "support": float(r["support"]),
        "confidence": float(r["confidence"]),
        "lift": float(r["lift"])
    }

# endregion


# region 3) Métricas de las reglas que pide el enunciado

rules_requested = [
    (["Romance"], ["Drama"]),
    (["Action", "Adventure"], ["Thriller"]),
    (["Crime", "Action"], ["Thriller"]),
    (["Crime"], ["Action", "Thriller"]),
    (["Crime"], ["Children's"]),
]

df_requested = pd.DataFrame([rule_metrics(A, B, onehot)
                            for A, B in rules_requested])

print("Metrics of the requested rules:")
print(df_requested, "\n")

# Discusión y reflexión en base a las métricas obtenidas:
#
# 1) Romance -> Drama
#    support = 0.058859: Romance y Drama aparecen juntos con frecuencia razonable.
#    confidence = 0.400810: si una película es Romance, ~40% también es Drama.
#    lift = 0.929879 (<1): aunque confidence sea alta, Drama es tan frecuente que Romance no lo “incrementa”.
#
# 2) Action, Adventure -> Thriller
#    support = 0.010702: aparece pocas veces.
#    confidence = 0.240000: dado Action+Adventure, Thriller aparece un 24% de las veces.
#    lift = 1.608287 (>1): asociación positiva clara.
#
# 3) Crime, Action -> Thriller
#    support = 0.002378: muy rara.
#    confidence = 0.173913: asociación moderada, pero en pocas películas.
#    lift = 1.165425 (>1): asociación positiva suave.
#
# 4) Crime -> Action, Thriller
#    confidence = 0.036697: casi nunca Crime implica (Action y Thriller a la vez).
#    lift = 0.734819 (<1): asociación negativa.
#
# 5) Crime -> Children's
#    support = confidence = lift = 0.0: no hay ninguna película con Crime y Children's a la vez en el dataset.
#
# Opinión:
# - support indica si la regla es relevante en la práctica (si aparece o no aparece mucho).
# - confidence puede engañar si el consecuente es muy frecuente (como Drama).
# - lift es la medida clave para ver asociación real: lift>1 positiva, lift≈1 independencia, lift<1 negativa.

# endregion


# region 4) Reglas A -> B: conteo matemático + DataFrame con rule_metrics

n = onehot.shape[1]

num_1to1_math = n * (n - 1)
print(f"Number of A -> B rules (math): n*(n-1) = {num_1to1_math}")

t0 = pd.Timestamp.now()

rules_1to1_obs = rules[
    (rules["antecedents"].apply(len) == 1) &
    (rules["consequents"].apply(len) == 1)
]

df_1to1_obs = pd.DataFrame([
    rule_metrics(list(row["antecedents"]), list(row["consequents"]), onehot)
    for _, row in rules_1to1_obs.iterrows()
])

t1 = pd.Timestamp.now()
elapsed_1to1 = (t1 - t0).total_seconds()

print(
    f"Tiempo generando DataFrame A -> B (observed): {format_seconds(elapsed_1to1)}")

df_1to1_sorted = df_1to1_obs.sort_values(
    by=["support", "confidence", "lift"], ascending=False)

print_top_as_comment(
    df_1to1_sorted, "TOP 20 A -> B rules (sorted by support, confidence, lift)", n=20)

# endregion


# region 5) Reglas A,B -> C y A -> B,C: conteo matemático + DataFrame con rule_metrics

# Conteo matemático:
# Elegimos 3 géneros: C(n,3)
# Por cada triple:
#   - 3 reglas 2->1 (elige el consecuente)
#   - 3 reglas 1->2 (elige el antecedente)
# Total = 6*C(n,3)
num_triples = nCk(n, 3)
num_3items_math = 6 * num_triples
print(f"Number of 3-item rules (math): 6*C(n,3) = {num_3items_math}")

t0 = pd.Timestamp.now()

rules_3items_obs = rules[
    ((rules["antecedents"].apply(len) == 2) & (rules["consequents"].apply(len) == 1)) |
    ((rules["antecedents"].apply(len) == 1) &
     (rules["consequents"].apply(len) == 2))
]

df_3items_obs = pd.DataFrame([
    rule_metrics(list(row["antecedents"]), list(row["consequents"]), onehot)
    for _, row in rules_3items_obs.iterrows()
])

t1 = pd.Timestamp.now()
elapsed_3items = (t1 - t0).total_seconds()

print(
    f"Tiempo generando DataFrame 3-items (observed): {format_seconds(elapsed_3items)}")

df_3items_sorted = df_3items_obs.sort_values(
    by=["support", "confidence", "lift"], ascending=False
)

print_top_as_comment(
    df_3items_sorted, "TOP 20 3-item rules (sorted by support, confidence, lift)", n=20
)

# endregion


# region 6) Reglas de 9 elementos: conteo matemático + estimación de tiempo

# Conteo matemático para reglas con |AuB|=9:
# - Elegimos 9 géneros: C(n,9)
# - Para esos 9, número de particiones dirigidas A->B con A y B no vacíos: (2^9 - 2)
# Total = C(n,9)*(2^9 - 2)
if n >= 9:
    num_rules_9 = nCk(n, 9) * ((2**9) - 2)
else:
    num_rules_9 = 0

print(f"Rules with 9 items (math): C(n,9)*(2^9-2) = {num_rules_9}")

# Estimación de tiempo: usar tiempo de "generación real" (apriori+rules) por regla
time_per_rule_gen = elapsed_rules / len(rules) if len(rules) > 0 else 0.0

if time_per_rule_gen > 0:
    est_9 = num_rules_9 * time_per_rule_gen
    print(f"Estimated time for 9-item rules: {format_seconds(est_9)}\n")
else:
    print("Cannot estimate time (no base timing).\n")

# endregion


# region 7) Todas las reglas posibles (1..19): conteo matemático + estimación de tiempo

# Total reglas A->B con A y B disjuntos, no vacíos:
# 3^n - 2*2^n + 1
total_all = (3**n) - 2 * (2**n) + 1
print(f"Total rules (all sizes 1..{n}) (math): 3^n - 2*2^n + 1 = {total_all}")

if time_per_rule_gen > 0:
    est_all = total_all * time_per_rule_gen
    print(f"Estimated time for ALL rules: {format_seconds(est_all)}")
else:
    print("Cannot estimate total time (no base timing).")

# endregion

# endregion

# ------

# region Ejercicio: Apriori (taller)

# Un solo DataFrame con reglas de 2 y 3 elementos
df_prev_all = pd.concat([df_1to1_sorted, df_3items_sorted], ignore_index=True)

# Soporte mínimo entre todas las reglas previas
min_support_prev_rules = df_prev_all["support"].min()
# min_support (reglas previas 2 y 3 items): 0.0005945303210463733 (≈ 1/1682)

# Apriori con min_support = 10**(-5) (sin max_len)
min_support_apriori = 10**(-5)
frequent_itemsets_ap = apriori(
    onehot.astype(bool),
    min_support=min_support_apriori,
    use_colnames=True
)

# Investigación del DataFrame de apriori():
# - Devuelve itemsets frecuentes y su soporte.
# - Columnas: 'support' e 'itemsets'.
# - En nuestra ejecución: 456 itemsets frecuentes (frequent_itemsets_ap.shape[0] = 456).

# Generar todas las reglas sin filtro
rules_ap = association_rules(
    frequent_itemsets_ap,
    metric="support",
    min_threshold=0.0
).copy()

print("Número total de reglas generadas sin filtrar:", len(rules_ap))
# Respuesta: 3902 reglas

# Investigación del DataFrame de association_rules:
# association_rules devuelve un DataFrame ancho porque incluye:
# - antecedents y consequents
# - soporte del antecedente y del consecuente
# - métricas de la regla (support, confidence, lift, etc.)
# Este formato permite aplicar pruning posterior.

# DataFrame con el formato de la práctica anterior
rules_ap["antecedents_str"] = rules_ap["antecedents"].apply(set_to_str)
rules_ap["consequents_str"] = rules_ap["consequents"].apply(set_to_str)
rules_ap["rule"] = rules_ap["antecedents_str"] + \
    " -> " + rules_ap["consequents_str"]

df_ap_format = rules_ap[["rule", "support", "confidence", "lift"]].copy()

# Top 20 Apriori vs Top 20 previo
top20_prev = df_prev_all.sort_values(
    by=["support", "confidence", "lift"], ascending=False
).head(20).reset_index(drop=True)

top20_ap = df_ap_format.sort_values(
    by=["support", "confidence", "lift"], ascending=False
).head(20).reset_index(drop=True)

print("\nTOP 20 PREVIO:")
print(top20_prev)

print("\nTOP 20 APRIORI:")
print(top20_ap)

# Justificación:
# Ambos TOP 20 coinciden porque el ranking prioriza el soporte.
# Apriori genera más reglas totales, pero no desplaza a las más frecuentes.

# Intersección de los TOP 20
intersect_top20 = pd.merge(
    top20_prev, top20_ap,
    on="rule",
    suffixes=("_prev", "_ap")
)

print("\nIntersección TOP 20:")
print(intersect_top20)

# Conclusión:
# La intersección es de 20 reglas. Las reglas más fuertes son robustas y coinciden en ambos métodos.

# endregion
