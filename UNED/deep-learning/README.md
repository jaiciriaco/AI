# Aprendizaje Profundo

- `pec0-mnist/DeepMLP_Jaime.ipynb`: MLP, búsqueda de learning rate y comparación de arquitecturas sobre MNIST.
- `pec1-dcgan/DCGAN_CIFAR10.ipynb`: versión cumplimentada de la plantilla de GANs, basada en CIFAR-10.
- `pec2-nlp/`: clasificación de noticias AG News con memoria original.

Para PEC0/PEC1 instalar `requirements-tensorflow.txt`; para PEC2 instalar `requirements-nlp.txt`, preferiblemente en otro entorno. Abrir cada cuaderno desde su propia carpeta. Los datasets y modelos externos se descargan durante la ejecución; no son archivos propios que falten. `best_mnist_model.keras` se genera en PEC0 y se ha omitido del repositorio.

Los notebooks de referencia del capítulo 17 no se incluyen como trabajo propio. Se conserva la versión cumplimentada de GANs y una sola copia de PLN.

## Estado y límites

Sintaxis revisada; entrenamientos no ejecutados durante esta publicación. Las dependencias son una lista de instalación, no un entorno bloqueado cuya compatibilidad completa se haya verificado. Es recomendable GPU para entrenamientos costosos.

El cuaderno MNIST original compara arquitecturas usando accuracy de test para seleccionar modelos. Estas métricas no deben presentarse como evaluación final independiente: una futura revisión debe seleccionar con validación y reservar test para el final. Se mantiene la entrega histórica sin cambiar sus resultados.
