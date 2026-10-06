# AI & Data Science Teaching

Material docente en español seleccionado de las carpetas de trabajo de Jaime Ciriaco. Esta colección se mantiene separada de los trabajos académicos de UNED.

## Contenido

| Carpeta | Contenido | Estado |
| --- | --- | --- |
| [week-01/](week-01/) | Introducción a IA, Big Data y Python: presentaciones, apuntes y tutorial | Tutorial ejecutado; PDF de clase |
| `statistics/` | 6 lecciones: álgebra lineal, cálculo, probabilidad, estadística, inferencia/A-B y proyecto integrador | Código ejecutado secuencialmente |
| `statistics/soluciones/` | 6 cuadernos de soluciones | Código ejecutado secuencialmente |
| `python/exercises/` | Ejercicios de las semanas 2, 3 y 4 | Plantillas para completar, no soluciones |

## Uso

Python 3.11 o posterior. Desde esta carpeta:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Los cuadernos de estadística generan datos sintéticos y no requieren datasets externos ni credenciales. Ejecutar las celdas en orden con un kernel nuevo.

## Criterio de selección y revisión

Se conservan lecciones autocontenidas y ejercicios con una finalidad clara. Se incluyen los apuntes y presentaciones propios de la semana 1. No se incluyen exportaciones redundantes, manuales recopilados, resultados voluminosos, copias antiguas ni cuadernos de clase con errores de sintaxis o ejecución incompleta. Las hojas de ejercicios están vacías deliberadamente.

Se han limpiado outputs y metadatos de ejecución. La regresión utiliza `numpy.linalg.lstsq`; los p-valores Monte Carlo incorporan la corrección `(b+1)/(B+1)`.

### Alcance estadístico

Los intervalos con 1.96 son aproximaciones normales, no intervalos exactos para cualquier muestra. Un IC del 95% describe la cobertura del procedimiento en muestreos repetidos, no una probabilidad del parámetro una vez observado el intervalo. El p-valor se calcula bajo la hipótesis nula; no es la probabilidad de que esta sea cierta. La permutación supone intercambiabilidad bajo la hipótesis nula. Los ejemplos son didácticos: significación estadística no equivale a relevancia práctica ni causalidad. El MSE del proyecto es de ajuste sobre sus datos y no estima generalización fuera de muestra.

## Verificación

2026-10-06: ejecutadas en orden las 92 celdas de código de los 12 cuadernos de estadística en procesos Python con gráficos sin interfaz. No se ha verificado la interfaz de Jupyter ni se han rellenado los ejercicios de Python. Las dependencias declaradas no son un lockfile.
