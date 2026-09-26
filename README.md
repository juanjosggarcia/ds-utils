## Añadir las dependencias de EDA

Para instalar las dependencias opcionales necesarias para las funciones de EDA y visualización:

```bash
uv sync --extra eda
```

---

## Probar los notebooks visuales

El proyecto no incluye `ipykernel` como dependencia. Para probar los notebooks de forma puntual, puede instalarse directamente en el entorno virtual:

```bash
uv pip install ipykernel
```

### Comprobar que `ipykernel` está instalado

```bash
uv run python -c "import ipykernel; print(ipykernel.__version__)"
```

### Registrar el entorno como kernel de Jupyter

Opcionalmente, se puede registrar el entorno virtual como un kernel de Jupyter a nivel de usuario:

```bash
uv run python -m ipykernel install \
    --user \
    --name ds-utils \
    --display-name "Python (ds-utils)"
```

Una vez registrado, el kernel aparecerá en Jupyter como **Python (ds-utils)**.

> `ipykernel` se instala únicamente para la validación de los notebooks y no se añade como dependencia del proyecto ni al `pyproject.toml`.
