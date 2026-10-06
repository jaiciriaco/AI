# Visión Artificial

## PEC1 — Procesamiento y calibración

Scripts de transformaciones, calibración, filtrado/segmentación e interacción con cámara. Ejecutar desde `pec1/`, con `requirements-classical.txt` instalado. Las ventanas OpenCV necesitan escritorio; varios ejercicios requieren cámara. Se incluyen recursos de ejemplo utilizados por los scripts, sin atribuir su autoría al alumno.

## PEC2 — Características y geometría

Detección de líneas, homografías, características y seguimiento. Ejecutar desde `pec2/`. Los scripts `pec2_ej3b.py` y `pec2_ej3c.py` utilizan cámara. Algunas salidas se escriben dentro de `data/`.

## PEC3 — Deep Learning

16 notebooks numerados: localización/orientación de la figura de Rey; preparación de SMART-OM e imgsMIIA; segmentación oral con U-Net y YOLO; preparación de una aportación de datos.

Instalar `requirements-deep.txt` en un entorno separado. Iniciar Jupyter desde `pec3/notebooks/`: `BASE_DIR` se resuelve al directorio padre (`pec3/`), sustituyendo la ruta absoluta del ordenador original. Ejecutar los cuadernos en orden, tras aportar los datos descritos en [MISSING_FILES.md](../MISSING_FILES.md). Las carpetas `outputs/` y `yolo/` son generadas, no código faltante.

## Estado

Código y sintaxis revisados. No se ejecutaron interfaces, cámaras ni entrenamientos. Los datasets principales de PEC3 no venían en la entrega. Las memorias `report.pdf` son las originales y sus resultados no se han vuelto a medir.

Se omiten del repositorio la fotografía de `data/face/`, los dibujos de `data/DibujosNPT/`, las tomas fallidas de calibración, vídeos y ejemplos auxiliares no usados. Las dos primeras carpetas sí venían en el ZIP y deben recuperarse localmente para los ejercicios que las usan; no son archivos perdidos. Se conservan las referencias en código.
