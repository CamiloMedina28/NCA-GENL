# Twitter (X) binary sentiment analysis using LSTM

## Validación del modelo

El modelo es una red LSTM para clasificación binaria de sentimiento. La clase
`Validator`, ubicada en `back/services/validations.py`, carga los pesos entrenados
y evalúa el modelo sobre `validation_df`, un conjunto que no participa en la
actualización de los pesos. Cada tweet se clasifica como `0` (negativo) o `1`
(positivo).

La red devuelve un **logit** `z`, es decir, un valor real sin restringir. Este
valor se transforma en una probabilidad mediante la función sigmoide:

\[
\hat p = \sigma(z) = \frac{1}{1 + e^{-z}}
\]

La clase predicha se obtiene usando el umbral configurado, `0.5` por defecto:

\[
\hat y =
\begin{cases}
1 & \text{si } \hat p \geq \text{umbral}\\
0 & \text{si } \hat p < \text{umbral}
\end{cases}
\]

El umbral no es una propiedad fija del modelo: puede modificarse si el negocio
prefiere reducir falsos positivos o falsos negativos. Por eso se reportan tanto
las probabilidades (`y_prob`) como las clases finales (`y_pred`).

### 1. Pérdida de entropía cruzada binaria (`loss`)

**Concepto técnico.** Mide el error entre la etiqueta real y la probabilidad
predicha. Se calcula con `BCEWithLogitsLoss`, que combina sigmoide y entropía
cruzada de forma numéricamente estable.

**Desarrollo matemático.** Para una observación `i`, con etiqueta
`y_i ∈ {0,1}` y logit `z_i`:

\[
\ell_i = -[y_i\log(\hat p_i) + (1-y_i)\log(1-\hat p_i)]
\]

La pérdida reportada es el promedio:

\[
L = \frac{1}{N}\sum_{i=1}^{N}\ell_i
\]

**Objetivo e importancia.** Indica qué tan buenas son las probabilidades, no
solo las decisiones después del umbral. Valores menores son mejores. Penaliza
fuertemente una predicción muy segura pero equivocada, por lo que ayuda a
detectar modelos sobreconfiados.

### 2. Matriz de confusión

La matriz compara la clase real con la predicha y contiene cuatro resultados:

|                    | Predicho negativo | Predicho positivo |
|--------------------|-------------------|-------------------|
| Real negativo      | TN                | FP                |
| Real positivo      | FN                | TP                |

`TN` son negativos correctamente identificados, `TP` positivos correctamente
identificados, `FP` falsos positivos y `FN` falsos negativos. Es el fundamento
de las métricas siguientes y se guarda como `lstm_confusion_matrix.png`.

**Objetivo e importancia.** Permite saber qué tipo de error comete el sistema.
En análisis de sentimiento esto es más útil que observar únicamente un valor
global: por ejemplo, muchos `FN` significan que el modelo está dejando pasar
opiniones positivas como negativas.

### 3. Exactitud (`accuracy`)

Es la proporción total de predicciones correctas:

\[
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
\]

Resume el rendimiento general y es fácil de comunicar. Sin embargo, puede ser
engañosa si las clases están desbalanceadas; por eso no se interpreta de forma
aislada.

### 4. Exactitud balanceada (`balanced_accuracy`)

Calcula el promedio del recall de cada clase:

\[
BalancedAccuracy = \frac{1}{2}\left(
\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right)
\]

**Objetivo e importancia.** Da el mismo peso a positivos y negativos, aunque
una clase tenga más ejemplos. Es una medida más justa para comprobar si el LSTM
aprende ambos sentimientos y no solamente la clase mayoritaria.

### 5. Precision, recall y especificidad

La **precision** indica qué fracción de los tweets predichos como positivos
realmente lo son:

\[
Precision = \frac{TP}{TP+FP}
\]

El **recall** o sensibilidad indica qué fracción de los positivos reales fue
detectada:

\[
Recall = \frac{TP}{TP+FN}
\]

La **especificidad** mide la capacidad de reconocer negativos:

\[
Specificity = \frac{TN}{TN+FP}
\]

Estas métricas permiten elegir el comportamiento deseado: precision alta reduce
falsas alarmas positivas; recall alto reduce positivos omitidos; especificidad
alta reduce negativos clasificados erróneamente como positivos.

### 6. F1-score (`f1`)

El F1 combina precision y recall mediante su media armónica:

\[
F1 = 2\frac{Precision\cdot Recall}{Precision+Recall}
\]

La media armónica penaliza el desequilibrio: el resultado no puede ser alto si
una de las dos métricas es baja. Es apropiado cuando se desea equilibrar la
identificación de positivos y el control de falsos positivos.

### 7. Curva ROC y ROC-AUC

La curva ROC se obtiene variando el umbral de decisión y graficando:

\[
TPR = \frac{TP}{TP+FN} \quad\text{contra}\quad
FPR = \frac{FP}{FP+TN} = 1-Specificity
\]

`ROC-AUC` es el área bajo esta curva. Puede interpretarse como la probabilidad
de que el modelo asigne una puntuación mayor a un positivo elegido al azar que
a un negativo elegido al azar. Un valor de `0.5` equivale aproximadamente al
azar y `1.0` representa separación perfecta.

**Objetivo e importancia.** Evalúa la capacidad de ordenar las clases sin
depender de un único umbral. Se guarda en `lstm_roc_curve.png`. Si existe una
clase ausente en la validación, esta prueba no se calcula porque el AUC deja de
ser definido.

### 8. Curva Precision-Recall y Average Precision

También varía el umbral, pero grafica precision contra recall. Es especialmente
informativa cuando la clase positiva es minoritaria. `average_precision` (AP)
resume la curva como un promedio ponderado de precision en los distintos niveles
de recall:

\[
AP = \sum_n (R_n-R_{n-1})P_n
\]

donde `P_n` es precision y `R_n` recall en cada punto ordenado por la
probabilidad predicha.

**Objetivo e importancia.** Muestra el compromiso entre encontrar más positivos
y mantener confiables las predicciones positivas. Se guarda en
`lstm_precision_recall_curve.png` y, al igual que ROC-AUC, requiere las dos
clases en el conjunto de validación.

### 9. Brier score (`brier_score`)

Mide el error cuadrático de las probabilidades:

\[
Brier = \frac{1}{N}\sum_{i=1}^{N}(\hat p_i-y_i)^2
\]

Su rango ideal comienza en `0`; cuanto menor, mejor. A diferencia de accuracy,
considera si una predicción fue, por ejemplo, `0.55` o `0.99`. Por ello sirve
para evaluar la calidad probabilística y la calibración del modelo.

### 10. Log loss (`log_loss`)

Es la misma entropía cruzada binaria usada como función de pérdida, calculada
ahora sobre las probabilidades finales de validación:

\[
LogLoss = -\frac{1}{N}\sum_i
[y_i\log(\hat p_i)+(1-y_i)\log(1-\hat p_i)]
\]

Valores menores son mejores. Se reporta junto con `loss` para dejar explícita
la calidad probabilística del modelo evaluado; pequeñas diferencias pueden
deberse a la forma numéricamente estable usada durante la inferencia.

### 11. Curva de calibración

Las probabilidades se dividen en 10 intervalos uniformes. Para cada intervalo
se compara la probabilidad media predicha con la fracción observada de positivos:

\[
\text{fracción positiva}_k = \frac{1}{|B_k|}
\sum_{i\in B_k} y_i
\]

Un modelo perfectamente calibrado cae sobre la diagonal `y=x`: entre los
casos predichos con `0.8`, aproximadamente el 80% debería ser positivo.

**Objetivo e importancia.** Verifica si una probabilidad puede interpretarse
como nivel de confianza. Se guarda en `lstm_calibration_curve.png`. Un modelo
puede tener buen AUC y, aun así, estar mal calibrado.

### 12. Distribución de probabilidades

Se generan histogramas separados para tweets negativos y positivos, junto con
una línea vertical en el umbral. El archivo resultante es
`lstm_probability_distribution.png`.

**Objetivo e importancia.** Es una inspección visual de la separación entre
clases y de la ambigüedad del modelo. Distribuciones muy superpuestas indican
casos difíciles; distribuciones concentradas cerca de `0` y `1` sugieren
decisiones más separadas, aunque deben contrastarse con Brier score y la curva
de calibración para comprobar que no exista exceso de confianza.

### 13. Cohen's kappa y correlación de Matthews

**Cohen's kappa** corrige el acuerdo observado por el acuerdo esperado al azar:

\[
\kappa = \frac{p_o-p_e}{1-p_e}
\]

donde `p_o` es el acuerdo observado y `p_e` el acuerdo esperado según las
proporciones marginales. Un valor cercano a `1` indica acuerdo fuerte, `0`
acuerdo equivalente al azar y valores negativos, desacuerdo.

El **coeficiente de correlación de Matthews (MCC)** usa las cuatro entradas de
la matriz:

\[
MCC = \frac{TP\cdot TN-FP\cdot FN}
{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
\]

Su rango es `[-1, 1]`. Es `1` con clasificación perfecta, `0` con rendimiento
similar al azar y `-1` con inversión perfecta de las clases. Ambas métricas son
útiles como controles adicionales frente al desbalance y al acuerdo por azar.

### 14. Classification report

El reporte incluye precision, recall, F1 y soporte (`support`) para las clases
`negative` y `positive`, además de promedios macro y ponderado. El soporte es el
número de ejemplos reales de cada clase.

**Objetivo e importancia.** Deja visible el rendimiento por clase, evitando que
un promedio global oculte que el modelo funciona bien para un sentimiento y mal
para el otro.

## Artefactos generados

Al ejecutar la validación se guardan en `back/media/`:

- `lstm_validation_metrics.json`: número de muestras, umbral, pérdida y todas
  las métricas numéricas.
- `lstm_confusion_matrix.png`: matriz de confusión.
- `lstm_roc_curve.png`: curva ROC, cuando están presentes ambas clases.
- `lstm_precision_recall_curve.png`: curva Precision-Recall, cuando están
  presentes ambas clases.
- `lstm_calibration_curve.png`: calibración en 10 intervalos.
- `lstm_probability_distribution.png`: distribución de probabilidades por clase.

## Orden recomendado para interpretar los resultados

Primero se revisan la matriz de confusión y el `classification_report` para
conocer los errores por clase. Después se comparan accuracy y balanced accuracy,
y se analiza el equilibrio precision-recall mediante F1. Finalmente, ROC-AUC y
AP evalúan la separación al cambiar el umbral, mientras Brier score, log loss y
la curva de calibración verifican si las probabilidades son confiables. No se
debe declarar que el modelo es bueno con una sola métrica.

