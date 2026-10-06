# PEC1 — Algoritmo genético para Sudoku

`ga_sudoku_base.py` implementa población, fitness, selección, cruce y mutación. `ga_sudoku_evaluation.py` realiza la evaluación; `ga_graph_evaluation.ipynb` representa los históricos.

[Memoria original](report.pdf).

## Uso

Instalar las dependencias con `python -m pip install -r ../requirements.txt`, desde esta carpeta. Mantener esta carpeta como directorio de trabajo:

```bash
python ga_sudoku_evaluation.py
```

Los evaluadores pueden tardar y sobrescribir los CSV existentes: realizar nuevos experimentos en una copia de esta carpeta.

## Verificación

Los históricos `historial_sudoku0.csv` y `historial_sudoku1.csv` pertenecen a la entrega. Se comprobaron fitness de un Sudoku resuelto, conservación de pistas y una ejecución pequeña del algoritmo.
