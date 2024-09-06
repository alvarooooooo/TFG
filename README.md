# TFG
Este proyecto está compuesto por la siguiente estructura de archivos y carpetas:
- models: Contiene los modelos entrenados. Además contiene el archivo ensembleNB.py donde se define el modelo de ensemble de NB creado.
- utils: Contiene ordered_one_hot_columns.pkl donde se almacena el orden de las columnas del conjunto de datos tabular. Además contiene utils.py donde se definen funciones auxiliares para los notebook.
- Lectura y Preporcesado de Datos.ipynb: Notebook para aplicar el preprocesado de datos al dataset original
- Ignition Detection.ipynb: Notebook que contiene el estudio comparativo de diferentes modelos y aproximaciones
- FWI Comparison.ipynb: Notebook donde se realiza la comparación entre el índice FWI y el modelo seleccionado

Para la ejecución de este proyecto se debe:
1. Descargar el dataset FireCube disponible en https://zenodo.org/records/6475592
2. Establecer la ruta del arhivo del dataset y ejecutar el notebook Lectura y Preprocesado de datos
3. Ejecutar el notbook Ignition Detection
4. Establecer la ruta del archivo del dataset y ejecutar el notebook FWI Comparison
