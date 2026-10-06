# Computación Evolutiva

| Carpeta | Trabajo |
| --- | --- |
| `pec1-sudoku/` | Algoritmo genético para Sudoku; evaluación y gráficos |
| `pec2-continuous-optimization/` | Estrategias evolutivas y evolución diferencial; Ackley y potencias desplazadas |
| `pec3-symbolic-regression/` | Programación genética, operadores de árboles y evaluación experimental |
| `pec4-parallel-evolution/` | Informe sobre evolución paralela de grano fino |

Cada carpeta contiene `report.pdf`. Los CSV son resultados históricos de la entrega, conservados para reproducir sus gráficos; no se han regenerado en esta revisión.

Instalar `python -m pip install -r requirements.txt`. Cambiar al directorio de la PEC antes de ejecutar sus scripts o notebooks. Los módulos base utilizan la biblioteca estándar; los gráficos necesitan NumPy, Pandas y Matplotlib.

- PEC1: `python ga_sudoku_evaluation.py` (menú interactivo).
- PEC2: `python ee_de_evaluation.py`; después `python ee_de_mejores_config.py`.
- PEC3: `python pg_evaluation.py`; después `pg_graph_evaluation.ipynb`.

Los barridos completos pueden tardar y sobrescribir los CSV del directorio actual: usar una copia de la PEC para nuevos experimentos. Se comprobaron ejecuciones pequeñas de GA, DE y PG, conservación de pistas del Sudoku y mínimos conocidos de las funciones objetivo. No se reprodujeron las campañas completas ni los resultados de los informes.
