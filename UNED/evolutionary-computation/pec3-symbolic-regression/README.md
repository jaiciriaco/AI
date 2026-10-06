# PEC3 — Regresión simbólica

Programación genética con árboles, operadores protegidos, cruce/mutación y evaluación de expresiones. `pg_base.py` contiene el núcleo y `pg_evaluation.py` las campañas experimentales.

[Memoria original](report.pdf).

## Uso

Instalar las dependencias con `python -m pip install -r ../requirements.txt`, desde esta carpeta. Mantener esta carpeta como directorio de trabajo:

```bash
python pg_evaluation.py
```

Los evaluadores pueden tardar y sobrescribir los CSV existentes: realizar nuevos experimentos en una copia de esta carpeta.

## Verificación

Abrir después `pg_graph_evaluation.ipynb` para analizar los CSV. Se comprobó una ejecución pequeña de programación genética; no se reprodujeron las campañas completas.
