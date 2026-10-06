# PEC3 — Visión con Deep Learning

Localización y orientación de la figura de Rey, y segmentación oral con U-Net y YOLO.

- [Memoria original](report.pdf).
- [Notebooks y orden de ejecución](notebooks/).

Instalar las dependencias con `python -m pip install -r ../requirements-deep.txt`, desde esta carpeta. Iniciar Jupyter desde `notebooks/`: los cuadernos resuelven `BASE_DIR` al directorio padre, esta carpeta.

Se utilizaron REY_DATASET, SMART-OM e imgsMIIA, mantenidos fuera de GitHub por su tamaño. No están pendientes de subir. Para reproducir los experimentos hacen falta localmente; sus rutas están descritas en [datos utilizados](../../MISSING_FILES.md).

`outputs/`, `yolo/`, máscaras, particiones y checkpoints son productos generados. Código revisado estáticamente; entrenamientos y métricas no se han vuelto a ejecutar.
