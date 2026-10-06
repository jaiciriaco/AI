# Métodos de Aprendizaje Automático

## Contenido

- `notebooks/`: sesgo-varianza I y II; LDA/QDA/RDA; métodos no paramétricos; remuestreo; reducción dimensional; outliers; regresión Diabetes.
- `scripts/`: codificación MovieLens, Naive Bayes y reglas de asociación/Apriori.
- `data/`: seis datasets de regresión, Mall Customers y una sola copia de la tabla MovieLens preparada.

## Ejecución

Crear un entorno de Python e instalar `python -m pip install -r requirements.txt`. Abrir Jupyter desde `notebooks/` y ejecutar cada cuaderno con un kernel nuevo. Sus CSV se leen desde `../data/`.

Los scripts se ejecutan desde `scripts/`; MovieLens descarga datos de GroupLens y Naive Bayes usa `ucimlrepo` (Iris y Bank Marketing). Requieren red. Los CSV generados por los scripts se escriben en el directorio de ejecución. `data/onehot_movies.csv` es una instantánea aportada, no una dependencia de descarga obligatoria.

## Revisión metodológica

Se conserva la solución académica, sin reinterpretar sus métricas como una evaluación independiente. En los ejercicios de remuestreo y reducción dimensional hay transformaciones ajustadas antes de la validación cruzada interna. Para estimaciones sin fuga entre folds, el escalado y PCA/selección de variables deben ajustarse dentro de cada fold. Las comparaciones repetidas sobre test tampoco sustituyen una evaluación final independiente.

Seis cuadernos completaron la ejecución local. Sesgo-varianza I requiere `statsmodels`, ausente del entorno de revisión; el barrido de Sesgo-varianza II excedió el límite local de 45 segundos. Esto no implica que el código sea incorrecto. Los scripts con descargas no se ejecutaron.
