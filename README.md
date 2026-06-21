# Prediccion-de-infartos
Predicción de infartos a partir de historial médico (dataset CDC/BRFSS). Clasificación con Random Forest y XGBoost, manejo de desbalanceo, clustering con K-Means e interpretabilidad con SHAP y LIME. Incluye análisis crítico sobre la utilidad clínica real de las variables.

Proyecto de machine learning que predice la probabilidad de que un paciente haya sufrido un infarto (`HadHeartAttack`) a partir de variables generales de su historial médico. Combina **clasificación supervisada**, **clustering no supervisado** e **interpretabilidad de modelos** para no solo predecir, sino entender *por qué* el modelo decide lo que decide.

## Objetivo

Detectar pacientes de riesgo de forma temprana usando únicamente datos de salud genéricos, como apoyo a la decisión clínica antes de recurrir a pruebas más específicas. El foco no es solo la precisión, sino la **interpretabilidad**: poder explicar el riesgo de cada paciente de forma individual.

## Dataset

Datos de salud poblacional de tipo CDC/BRFSS (encuesta de factores de riesgo conductuales), con ~33 variables por paciente: demografía, condiciones médicas previas, hábitos (tabaco, alcohol), vacunación y dificultades funcionales.

**Reto principal:** fuerte desbalanceo de clases (los infartos son una minoría), que se trata explícitamente en el proyecto.

## Metodología

**1. Análisis exploratorio y preprocesado**
- Inspección de nulos y tipos de variable.
- Eliminación de variables sin poder predictivo (verificado entrenando modelos preliminares y revisando su importancia).
- Codificación de categóricas (`get_dummies`), mapeo ordinal de edad y estado de salud, y normalización con `MinMaxScaler`.

**2. Tratamiento del desbalanceo**
- Comparación explícita entre un dataframe desbalanceado y otro balanceado mediante *undersampling* (`RandomUnderSampler`), mostrando por qué una accuracy alta sobre datos desbalanceados es engañosa.

**3. Clasificación binaria**
- Comparativa de varios modelos (Random Forest, XGBoost, entre otros).
- Ajuste de hiperparámetros con `GridSearchCV` y `RandomizedSearchCV`.
- Evaluación con matriz de confusión, curva ROC y AUC, más allá de la accuracy.

**4. Clustering con K-Means**
- Selección del número de clusters (método del codo + índice de Silhouette).
- Caracterización de cada grupo de pacientes y su tasa de infartos asociada.

**5. Interpretabilidad**
- **SHAP** para la importancia global de las características.
- **LIME** para explicar predicciones individuales, incluyendo casos límite donde el riesgo no es evidente.

## Resultados

| Métrica | Valor |
|---|---|
| Accuracy (clasificación binaria, datos balanceados) | 79.9 % |
| AUC | 0.88 |
| Accuracy (clasificación de clusters, tras tuning) | 97.8 % |

El modelo binario ofrece un rendimiento sólido para datos de salud genéricos, y el análisis de interpretabilidad permite justificar cada predicción a nivel de paciente.

## Lecciones aprendidas / qué haría distinto

El análisis de importancia (feature importances + SHAP) reveló que el modelo se apoyaba casi por completo en la variable **`HadAngina`** (antecedentes de angina de pecho). Esto plantea un problema de fondo más allá de la precisión:

- **Utilidad clínica real:** un paciente con angina diagnosticada ya es, para cualquier médico, un caso de alto riesgo cardiovascular evidente. Es decir, *la variable más predictiva del modelo es la que menos aporta al objetivo declarado*, que es la detección **preventiva y temprana** en pacientes cuyo riesgo aún no es obvio.
- **Robustez:** un modelo que depende tan fuertemente de una única variable es frágil ante datos incompletos o ruidosos.

**Mejora natural:** entrenar una variante "preventiva" excluyendo las variables que presuponen un diagnóstico cardiológico previo (`HadAngina`, probablemente `HadStroke`), aceptando un AUC algo menor a cambio de un modelo realmente útil en el escenario de cribado que motiva el proyecto. Cuantificar esa caída de AUC sería el siguiente paso.

> Nota técnica: no se trata de *target leakage* en sentido estricto (la angina no es el infarto), sino de una variable cuyo valor predictivo proviene de información que no está disponible en el momento en que el modelo sería útil.

## Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `imbalanced-learn` · `SHAP` · `LIME` · `matplotlib` · `seaborn`

## Cómo reproducirlo

```bash
git clone https://github.com/<tu-usuario>/prediccion-infartos-ml.git
cd prediccion-infartos-ml
pip install -r requirements.txt
jupyter notebook Actividad_final.ipynb
```

El dataset debe colocarse en la ruta indicada al inicio del notebook (no se incluye en el repositorio por tamaño/licencia).
