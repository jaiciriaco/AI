# PEC0 — MLP para MNIST

[DeepMLP_Jaime.ipynb](DeepMLP_Jaime.ipynb) estudia redes densas, búsqueda de learning rate y comparación de arquitecturas sobre MNIST.

Instalar las dependencias con `python -m pip install -r ../requirements-tensorflow.txt`, desde esta carpeta. Ejecutar `jupyter lab` desde esta carpeta y recorrer las celdas en orden. Keras descarga MNIST; el entrenamiento genera el checkpoint `best_mnist_model.keras`, excluido de Git.

No se reentrenó durante la revisión. El original usa accuracy de test para seleccionar arquitecturas: sus métricas no constituyen una evaluación final independiente. Para una nueva evaluación, seleccionar con validación y reservar test para el final.
