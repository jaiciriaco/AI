# Secuencia de notebooks — PEC3

| Orden | Tarea |
| --- | --- |
| 00 | Inspección y limpieza de datos de Rey |
| 01 | Localización con CNN Keras |
| 02–03 | Preparación y entrenamiento YOLO para localización |
| 04 | Clasificación de orientación con Keras |
| 05–06 | Preparación y entrenamiento YOLO de orientación |
| 07–08 | Inspección de SMART-OM y sus anotaciones |
| 09–10 | Inspección de ZIP imgsMIIA y preparación de máscaras |
| 11 | Unión de datasets y particiones |
| 12 | Entrenamiento de U-Net ligera |
| 13–14 | Preparación, entrenamiento y evaluación YOLO de segmentación |
| 15 | Preparación independiente de la aportación de datos |

Ejecutar `jupyter lab` desde esta carpeta y mantenerla como directorio del kernel. Instalar antes `../../requirements-deep.txt`.

Los pasos de cada rama necesitan los datos y salidas previos. El paso 15 es una utilidad adicional, no un requisito posterior al entrenamiento. Los datasets se excluyen intencionadamente por tamaño; ver [la guía de PEC3](../README.md). No se ejecutaron entrenamientos en esta revisión.
