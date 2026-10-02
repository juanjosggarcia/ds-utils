# ds-utils

Librería Python independiente con utilidades reutilizables para Data Science, análisis exploratorio de datos (EDA), preprocesamiento y Machine Learning.

---

## 1. Descripción general

`ds-utils` nace con el objetivo de centralizar, estandarizar y reutilizar utilidades prácticas de Data Science extraídas de proyectos reales, evitando la dispersión y reimplementación constante de código auxiliar en notebooks y scripts.

### ¿Qué problemas resuelve?
- **Auditoría e higiene de datos**: simplifica la comprobación de calidad de DataFrames (recuento detallado de nulos, identificación de constantes, análisis de columnas y verificación contractual de esquemas).
- **Tratamiento temporal y series temporales**: proporciona funciones robustas para detectar y gestionar problemas comunes en series de tiempo (marcas duplicadas, ráfagas a alta frecuencia, saltos temporales o *gaps*, remuestreo sin pérdida y extracción de variables estacionales/cíclicas).
- **Limpieza de telemetría y sensores**: detección y depuración de picos de ruido aislados en señales de sensores y eliminación de variables auxiliares redundantes.
- **Interpretabilidad y soporte en Machine Learning**: estandarización de puntuaciones de anomalía heterogéneas, diagnóstico explicativo de anomalías multivariantes e inspección legible de hiperparámetros.
- **Exploración visual consistente**: gráficos estandarizados con Matplotlib y previsualizaciones HTML enriquecidas listas para Jupyter.
- **Formateo y representación matemática**: conversión modular de vectores, matrices y operaciones algebraicas a LaTeX para documentación y visualizaciones en notebooks.

### ¿Para qué proyectos o workflows resulta útil?
- Pipelines de preparación de datos y feature engineering con `pandas` y `numpy`.
- Proyectos de series temporales y telemetría de dispositivos / sensores industriales.
- Flujos de trabajo no supervisados de detección de anomalías.
- Notebooks interactivos de análisis exploratorio (EDA), reporte de calidad de datos y documentación matemática de algoritmos.

---

## 2. Funcionalidades incluidas

La librería está estructurada en módulos según su responsabilidad funcional:

### DataFrame / Preprocesamiento (`ds_utils.preprocessing`)
Disponible a través de la fachada pública `ds_utils.preprocessing.preprocessing_utilities`:

* **Esquemas, índices y columnas**:
  * `validate_required_columns`: verificación estricta de presencia de columnas requeridas por contrato.
  * `set_index_safe` y `reset_index_safe`: manipulación idempotente y segura de índices en DataFrames evitando columnas espurias como `'index'`.
  * `get_columns_summary`: diagnóstico tabular completo por columna (tipo de dato, cardinalidad, valores no válidos y porcentajes).
  * `find_column_pairs`: emparejamiento automático de columnas originales y sustitutas mediante prefijo o sufijo.
  * `remove_replaceable_columns`: supresión limpia de columnas originales que ya cuentan con una versión procesada.
  * `count_constant_columns` y `remove_constant_columns`: detección y descarte de columnas sin varianza con opción de proteger columnas clave (`excluded_columns`).

* **Tratamiento de valores ausentes**:
  * `count_cells_with_missing`: suma global de celdas no válidas (nulos de pandas y cadenas vacías/espacios).
  * `count_rows_with_missing` y `count_columns_with_missing`: total de filas y columnas degradadas.
  * `count_missing_cells_by_column`: desglose por columna en formato `pd.Series`.
  * `count_edge_null_rows`: recuento de filas nulas consecutivas en los bordes inicial y final (`any` o `all`).
  * `handle_missing_values`: auditoría detallada e imputación selectiva mediante valores escalares o mapeos por columna.

* **Series temporales y frecuencias**:
  * `detect_timestamp`, `convert_timestamp` y `convert_timestamps_vectorized`: detección y conversión polimórfica de marcas temporales (segundos, milisegundos, microsegundos, nanosegundos y cadenas ISO 8601).
  * `handle_duplicates` y `find_duplicates`: detección y resolución de timestamps repetidos conservando primera o última ocurrencia.
  * `handle_time_split`: división temporal controlada en modos `'tail'` y `'head'`.
  * `handle_time_bursts`: filtrado de ráfagas temporales (*bursts*) respetando un intervalo mínimo tolerable.
  * `find_large_gaps`: identificación de interrupciones temporales que superan un umbral (`freq * limit`).
  * `count_missing_timestamps`: conteo e indexación de instantes faltantes frente a una cuadrícula regular teórica.
  * `count_rows_by_time`: agregación del volumen de filas por períodos temporales configurables (`'D'`, `'ME'`, etc.).
  * `resample_time`: remuestreo regular (`asfreq`) sin agregaciones destructivas, con soporte de redondeo e imputación hacia adelante/atrás.
  * `prepare_time_features`: generación de variables temporales para ML (hora lineal, hora cíclica seno/coseno, día de la semana numérico, textual y cíclico).

* **Limpieza de señales de sensores**:
  * `handle_sensor_noise`: detección y corrección de picos de ruido mediante ventanas de validación y cálculo de derivadas para ML.
  * `remove_redundant_sensor_noise_columns`: eliminación automática de columnas auxiliares para sensores cuya señal no requirió corrección.

* **Preparación para ML**:
  * `prepare_dataframe_for_ml`: transformación integral a formatos numéricos y categóricos compatibles con scikit-learn y librerías de gradient boosting (`'category'`, `'onehot'`, `'label'`, `'drop'`).

---

### EDA y visualización (`ds_utils.eda`)

* **Visualizaciones con Matplotlib (`ds_utils.eda.plot_utilities`)**:
  * `bold_matplotlib`: formateo matemático en LaTeX (`$\bf{...}$`) para títulos y etiquetas en figuras.
  * `plot_circular_feature`: representación en coordenadas polares de variables cíclicas (horas, días de la semana).
  * `plot_sensor_signals`: visualización simultánea de la señal original, la señal limpia y los eventos marcados como ruido.
  * `plot_model_anomaly_scores` y `plot_model_anomaly_score_distribution`: visualización temporal e histograma con KDE de puntuaciones de anomalía y umbral de decisión.
  * `plot_model_predictions`: comparativa de series reales frente a predicciones en el eje temporal con cálculo automático o personalizado de métricas (MAE).
  * `plot_model_feature_importance`: gráfico de barras horizontales ordenado para importancias de características o coeficientes lineales.
  * `plot_feature_space_projection`: proyección bidimensional del espacio de variables mediante PCA, t-SNE o UMAP con coloreado por scores o etiquetas.

* **Previsualización y tablas HTML (`ds_utils.eda.preview_utilities`)**:
  * `preview_dataframe`: renderizado interactivo y estilizado de DataFrames en entornos Jupyter.
  * `show_df_details`: informe diagnóstico completo (dimensiones, tipos, celdas nulas, columnas constantes y duplicados).
  * `show_df_differences`: análisis de discrepancias entre dos versiones de un DataFrame (filas añadidas/eliminadas, columnas y diferencias celda a celda).
  * `show_noise_summary`: desglose tabular de niveles de ruido y correcciones por sensor.
  * Utilidades de formateo: `print_html_table`, `print_html_list`, `print_html_dict`, `print_html_section`, `print_markdown_table`, `print_markdown_list`, `print_markdown_section`, `print_console_table`, `print_console_list`, `print_console_section`, `print_math_list_expression`, `print_math_expression`, `print_math_section` y `format_time_gap`.

* **Utilidades matemáticas y LaTeX (`ds_utils.eda.math_utilities`)**:
  * `matrix_to_latex`: conversión de arreglos NumPy y vectores 1D a matrices LaTeX (`pmatrix`) con redondeo configurable y eliminación de ceros no significativos.
  * `frac_to_latex`: formateo de fracciones (`\frac{num}{den}`) para valores numéricos o expresiones algebraicas.
  * `sqrt_to_latex`: representación de raíz cuadrada estándar (`\sqrt{x}`).
  * `root_to_latex`: representación de raíz n-ésima con orden o índice configurable (`\sqrt[n]{x}`).
  * `power_to_latex`: representación de potencias y exponentes con llaves seguras (`base^{exponente}`).
  * `subscript_to_latex`: adición de subíndices a expresiones, variables o tensores (`expresion_{subindice}`).
  * `abs_to_latex`: valor absoluto con delimitadores escalables (`\left| x \right|`).
  * `norm_to_latex`: norma vectorial o matricial con delimitadores escalables de doble barra (`\left\| x \right\|`).
  * `transpose_to_latex`: notación de matriz o vector traspuesto (`A^{T}`).
  * `inverse_to_latex`: notación de matriz inversa (`A^{-1}`).
  * `determinant_to_latex`: función de determinante (`\det\left(A\right)`).

---

### Machine Learning (`ds_utils.ml`)
Disponible a través de `ds_utils.ml.ml_utilities`:

* **Normalización y homologación de anomaly scores**:
  * `normalize_anomaly_scores_min_max`: reescalado Min-Max invertido y acotado por percentiles (`[0, 1]`, donde `0` es normal y `1` es anómalo), inmune a distorsiones por valores atípicos extremos.
  * `normalize_anomaly_scores_rank`: normalización basada en ranking percentilar para comparar o combinar salidas de modelos con escalas heterogéneas.

* **Interpretabilidad de anomalías**:
  * `explain_anomalies`: generación de diagnósticos legibles que detallan qué variables numéricas causaron cada anomalía respecto a sus medianas históricas y percentiles P99.

* **Gestión de modelos e hiperparámetros**:
  * `normalize_model_name`: estandarización de nombres textuales de modelos a identificadores `snake_case`.
  * `format_max_features`: representación explicativa del parámetro `max_features` indicando el ratio respecto al total de características.
  * `extract_used_params`: filtrado selectivo de hiperparámetros explícitos a partir de `estimator.get_params()`.
  * `bold`: formateo de texto en negrita LaTeX para anotaciones de métricas y modelos.

---

### Contratos de datos (`ds_utils.base_classes`)
* `DatasetContract`: clase base abstracta (`ABC`) para formalizar contratos de datasets, definición de esquemas de columnas requeridas, particiones temporales y validaciones estructurales de datos.

---

## 3. Instalación

`ds-utils` utiliza `uv` como gestor de paquetes y entorno. La instalación base es deliberadamente ligera y solo depende de `numpy` y `pandas`.

### Instalación como dependencia en otro proyecto
Para instalar la librería directamente desde el repositorio de GitHub fijando la versión de la release `v0.1.0`:

```bash
# Instalación base (preprocesamiento y utilidades generales)
uv add "git+https://github.com/juanjosggarcia/ds-utils.git@v0.1.0"

# Con extras de EDA y visualización
uv add "ds-utils[eda] @ git+https://github.com/juanjosggarcia/ds-utils.git@v0.1.0"

# Con extras de Machine Learning
uv add "ds-utils[ml] @ git+https://github.com/juanjosggarcia/ds-utils.git@v0.1.0"

# Con todos los extras
uv add "ds-utils[eda,ml] @ git+https://github.com/juanjosggarcia/ds-utils.git@v0.1.0"
```

### Instalación en el entorno de desarrollo local (clon del repositorio)
Si has clonado el repositorio y deseas configurar el entorno virtual:

```bash
# 1. Clonar el repositorio
git clone https://github.com/juanjosggarcia/ds-utils.git
cd ds-utils

# 2. Sincronización base
uv sync
```

#### Instalación de extras opcionales:
```bash
# Sincronizar con dependencias de EDA y visualización
uv sync --extra eda

# Sincronizar con dependencias de Machine Learning
uv sync --extra ml

# Sincronizar con todos los extras opcionales
uv sync --all-extras
```

#### Dependencias de desarrollo:
Para ejecutar los notebook en vscode con `ipykernel`, pruebas con `pytest` o herramientas de calidad de código (`ruff`, `mypy`):
```bash
uv sync --all-extras --dev
```

---

## 4. Dependencias y Extras

Las dependencias están definidas en `pyproject.toml` según las necesidades de cada flujo de trabajo:

| Componente | Dependencias requeridas | Finalidad principal |
|---|---|---|
| **Base** | `numpy>=2.0`, `pandas>=2.2` | Preprocesamiento, auditoría de datos, series temporales y utilidades centrales. |
| **Extra `eda`** | `matplotlib>=3.9`, `ipython>=8.0`, `jinja2>=3.1`, `tabulate>=0.10.0` | Generación de gráficos, estilos HTML interactivos, renderizado matemático en LaTeX y previsualizaciones en Jupyter. |
| **Extra `ml`** | `python-dotenv>=1.0.0`, `scikit-learn>=1.5` | Algoritmos de soporte, proyecciones espaciales y modelos de Machine Learning, junto con la gestión de configuración y credenciales mediante variables de entorno. |
| **Grupo `dev`** | `pytest>=8.0`, `ruff>=0.12`, `mypy>=1.0`, `ipykernel>=7.3.0` | Suite de pruebas unitarias, formateo, linting, chequeo estático de tipos y cuadernos jupyter en vscode. |

---

## 5. Ejemplos de uso rápido

### Diagnóstico e higiene de DataFrames
```python
import pandas as pd
from ds_utils.preprocessing.preprocessing_utilities import (
    validate_required_columns,
    get_columns_summary,
    handle_missing_values,
)

# Datos de prueba
df = pd.DataFrame({
    "sensor_id": [101, 102, 103],
    "temperature": [21.5, None, 22.0],
    "status": ["OK", "   ", "OK"]
})

# 1. Validar presencia de columnas obligatorias
validate_required_columns(df, required_cols=["sensor_id", "temperature"])

# 2. Resumen diagnóstico de calidad de columnas
summary = get_columns_summary(df, sort_by="porcentaje_no_validos", ascending=False)
print(summary)

# 3. Imputar nulos mediante diccionario específico
df_clean, audit = handle_missing_values(df, fill_value={"temperature": 20.0, "status": "Desconocido"})
```

### Normalización y explicación de anomalías (ML)
```python
import pandas as pd
from ds_utils.ml.ml_utilities import (
    normalize_anomaly_scores_min_max,
    explain_anomalies,
)

# Datos de telemetría y puntuaciones de anomalía
df_telemetry = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01 08:00", periods=5, freq="10min"),
    "temperature": [22.0, 22.1, 55.4, 22.2, 22.0],
    "vibration": [1.0, 1.1, 1.0, 1.2, 1.0],
})

raw_scores = pd.Series([0.25, 0.30, -0.75, 0.20, 0.28])
anomaly_mask = raw_scores < 0.0

# 1. Normalizar scores al rango [0, 1] (1 = máxima anomalía)
normalized_scores = normalize_anomaly_scores_min_max(raw_scores)

# 2. Explicar qué variables causaron las anomalías detectadas
explanations = explain_anomalies(
    df=df_telemetry,
    col_time="timestamp",
    anomaly_scores=raw_scores,
    anomaly_mask=anomaly_mask,
    top_features=2,
)
print(explanations[["timestamp", "score", "summary"]])
```

### Formateo y representación matemática en LaTeX
```python
import numpy as np
from ds_utils.eda.math_utilities import (
    matrix_to_latex,
    transpose_to_latex,
    inverse_to_latex,
    sqrt_to_latex,
    frac_to_latex,
)

# 1. Convertir matriz o vector de NumPy a LaTeX (entorno pmatrix)
A = np.array([[1.0, 2.5], [3.14159, 4.0]])
print(matrix_to_latex(A))
# Salida: \begin{pmatrix}1 & 2.5 \\ 3.14 & 4\end{pmatrix}

# 2. Componer la Ecuación Normal de Mínimos Cuadrados (OLS): (X^T * X)^(-1) * X^T * y
xt = transpose_to_latex(r"\mathbf{X}")
inv_gram = inverse_to_latex(f"({xt} " + r"\mathbf{X})")
beta_ols = r"\hat{\boldsymbol{\beta}} = " + f"{inv_gram} {xt} " + r"\mathbf{y}"
print(beta_ols)
# Salida: \hat{\boldsymbol{\beta}} = (\mathbf{X}^{T} \mathbf{X})^{-1} \mathbf{X}^{T} \mathbf{y}

# 3. Expresión analítica de la fórmula cuadrática
disc = sqrt_to_latex("b^{2} - 4ac")
formula_cuadratica = f"x = {frac_to_latex(r'-b \pm ' + disc, '2a')}"
print(formula_cuadratica)
# Salida: x = \frac{-b \pm \sqrt{b^{2} - 4ac}}{2a}
```

---

## 6. Notebooks de ejemplo

El directorio [`notebook_samples/`](notebook_samples/) del repositorio incluye notebooks Jupyter interactivos y autocontenidos que documentan y demuestran de forma práctica cada una de las funciones de la librería:

```text
notebook_samples/
├── eda/
│   ├── math_utilities_demo.ipynb
│   ├── plot_utilities_demo.ipynb
│   └── preview_utilities_demo.ipynb
├── ml/
│   └── ml_utilities_demo.ipynb
└── preprocessing/
    ├── preprocessing_counters_utilities_demo.ipynb
    ├── preprocessing_features_utilities_demo.ipynb
    ├── preprocessing_finders_utilities_demo.ipynb
    ├── preprocessing_handlers_utilities_demo.ipynb
    ├── preprocessing_ml_utilities_demo.ipynb
    ├── preprocessing_others_utilities_demo.ipynb
    └── preprocessing_remove_utilities_demo.ipynb
```

> **Nota importante**: Estos notebooks están concebidos como material de consulta, documentación interactiva y ejemplos pedagógicos dentro del repositorio. **No se instalan como parte del paquete al añadir `ds-utils` como dependencia externa**.

### Ejecución de los notebooks en local
Para ejecutar los notebooks en tu entorno local con Jupyter, puedes instalar puntualmente `ipykernel`:

```bash
uv pip install ipykernel
uv run python -m ipykernel install --user --name ds-utils --display-name "Python (ds-utils)"
```

Una vez registrado, el kernel aparecerá en Jupyter como **Python (ds-utils)**.

---

## 7. Estado del proyecto

Esta es la primera versión pública de la librería (**`v0.1.0`**).

Al tratarse de una versión inicial de consolidación, las interfaces y funcionalidades pueden experimentar ajustes y mejoras en versiones posteriores conforme se incorporen nuevos casos de uso y utilidades adicionales.
