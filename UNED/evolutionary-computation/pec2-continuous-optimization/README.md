# PEC2 — Optimización continua

Comparación de estrategias evolutivas y evolución diferencial en Ackley y potencias desplazadas. `ee_de_base.py` contiene los algoritmos; los otros dos scripts ejecutan barridos y generan gráficos.

[Memoria original](report.pdf).

## Uso

Instalar las dependencias con `python -m pip install -r ../requirements.txt`, desde esta carpeta. Mantener esta carpeta como directorio de trabajo:

```bash
python ee_de_evaluation.py
python ee_de_mejores_config.py
```

Los evaluadores pueden tardar y sobrescribir los CSV existentes: realizar nuevos experimentos en una copia de esta carpeta.

## Verificación

Los CSV incluidos son históricos, no resultados regenerados en la revisión. Se comprobaron mínimos conocidos de los objetivos y una ejecución corta de evolución diferencial; no se reprodujo el barrido completo.
