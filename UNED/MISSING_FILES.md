# Datos y recursos utilizados

## Datasets externos de Visión Artificial PEC3

Se utilizaron los siguientes datasets durante el trabajo. Se mantienen fuera de GitHub por su tamaño; su subida no está pendiente. Para reproducir los experimentos es necesario disponer de ellos localmente en las rutas indicadas.

| Proyecto | Recurso utilizado | Uso |
| --- | --- | --- |
| Visión Artificial PEC3 | `PEC3_VA/data/REY_DATASET/`, incluido `traza_REY.csv` y las imágenes | Notebooks 00–06: localización y orientación de Rey |
| Visión Artificial PEC3 | `PEC3_VA/data/SMART-OM/`, con imágenes y anotaciones | Notebooks 07–08 y segmentación posterior |
| Visión Artificial PEC3 | `PEC3_VA/data/imgsMIIA_colab/raw_zips/` con los ZIP colaborativos | Notebooks 09–14: preparación y segmentación oral |
| Visión Artificial PEC3 | `PEC3_VA/data/aportacion_jaime_raw/jimenez_mario.zip` | Notebook 15, preparación de la aportación propia |

La aportación final `imgsMIIA_jaime.zip`, mencionada en la memoria de la entrega, también se mantiene fuera de GitHub.

## Incluidos en el ZIP, excluidos del repositorio

- PEC1 de Visión: `data/face/image.png` y `data/DibujosNPT/`. Para reproducir esos ejercicios, restaurar localmente desde el ZIP original.
- Pesos `best_mnist_model.keras`: pueden regenerarse con el entrenamiento de MNIST.
- Cachés, outputs de notebooks, enunciados independientes, copias repetidas, imágenes de resultados exportadas y material de referencia.

## Recursos que se descargan o generan

MovieLens, UCI, MNIST, CIFAR-10, AG News y modelos preentrenados requieren acceso a sus proveedores. Los CSV limpios, splits, máscaras, carpetas YOLO y checkpoints de PEC3 se generan por pasos; no hace falta buscar cada salida si se recuperan los datos originales y se repite el proceso.

No se detectaron módulos Python propios ausentes en las entregas seleccionadas. Esta comprobación estática no garantiza que todas las rutas dinámicas o servicios externos funcionen.
