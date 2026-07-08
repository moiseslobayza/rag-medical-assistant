# RAG Medical Assistant

Sistema de preguntas y respuestas basado en **Retrieval-Augmented Generation (RAG)** para consultas administrativas de un consultorio médico privado.

El proyecto transforma una consulta del usuario en un embedding, recupera contexto relevante desde una base vectorial y utiliza un modelo generativo para redactar una respuesta en español basada en la información recuperada.

> **Alcance:** este asistente trabaja con información administrativa del consultorio. No realiza diagnósticos ni reemplaza la evaluación de un profesional de la salud.

## Arquitectura

```text
Consulta del usuario
        ↓
Embedding de la consulta
        ↓
Chroma Vector Database
        ↓
Recuperación Top-K
        ↓
Contexto recuperado
        ↓
FLAN-T5
        ↓
Respuesta en español
```

## Tecnologías

- Python
- Pandas
- Sentence Transformers
- Hugging Face Transformers
- Chroma
- PyTorch
- FLAN-T5
- GitHub Actions

## Modelos

El sistema utiliza por defecto:

- **LLM:** `google/flan-t5-small`
- **Embeddings:** `intfloat/multilingual-e5-small`

La implementación contempla el formato recomendado por la familia E5, utilizando los prefijos `query:` para consultas y `passage:` para documentos.

El proyecto incluye además una comparación de recuperación semántica entre:

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `intfloat/multilingual-e5-small`

La evaluación se realiza de forma independiente del modelo generativo para analizar específicamente la etapa de retrieval.

## Base de conocimiento

La base de conocimiento contiene preguntas y respuestas administrativas sobre:

- turnos y confirmaciones
- horarios de atención
- ubicación
- obra social y atención particular
- cancelaciones y reprogramaciones
- recetas y certificados
- formas de pago y facturación
- estudios previos y resultados

Los datos se encuentran en:

```text
data/knowledge_base.csv
```

Cada registro se transforma en un documento con la siguiente estructura:

```text
Pregunta frecuente: <pregunta>
Respuesta: <respuesta>
```

## Estructura del proyecto

```text
rag-medical-assistant/
│
├── .github/
│   └── workflows/
│       └── validate.yml
│
├── data/
│   └── knowledge_base.csv
│
├── notebooks/
│   └── embedding_experiments.ipynb
│
├── src/
│   ├── __init__.py
│   ├── chatbot.py
│   └── evaluation.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Instalación

Crear un entorno virtual e instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

Ejecutar el asistente desde la terminal:

```bash
python main.py
```

La primera ejecución puede tardar porque los modelos deben descargarse desde Hugging Face.

Para salir del chat:

```text
salir
```

## Flujo RAG

1. Se carga la base de conocimiento desde CSV.
2. Cada pregunta y respuesta se convierte en un documento textual.
3. Los documentos se transforman en embeddings normalizados.
4. Los embeddings y metadatos se almacenan en Chroma usando similitud coseno.
5. La consulta del usuario se transforma en un embedding.
6. Chroma recupera los documentos más similares mediante búsqueda Top-K.
7. El contexto recuperado se incorpora al prompt.
8. FLAN-T5 genera una respuesta en español basada en ese contexto.

## Evaluación de embeddings

El módulo `src/evaluation.py` permite evaluar la recuperación semántica sin cargar el LLM.

La comparación utiliza 10 consultas reformuladas y mide si la pregunta esperada de la base de conocimiento aparece dentro de los primeros resultados recuperados.

Se analizan:

- `Top-1`
- `Top-3`
- `Top-5`
- tasa de aciertos (`hit_rate`)
- posición media de la pregunta esperada
- errores de recuperación

El experimento reproducible se encuentra en:

```text
notebooks/embedding_experiments.ipynb
```

### Resultados observados

Los siguientes resultados fueron obtenidos ejecutando la evaluación en GitHub Actions sobre las 10 consultas del prototipo:

| Modelo | Top-1 | Top-3 | Top-5 | Posición media esperada |
|---|---:|---:|---:|---:|
| MiniLM multilingual | 80% | 80% | 100% | 1.8 |
| multilingual-E5-small | 80% | 90% | 100% | 1.4 |

En este conjunto pequeño, ambos modelos alcanzaron el mismo resultado en Top-1 y Top-5. E5 obtuvo una mayor tasa de aciertos en Top-3 y una mejor posición media de la pregunta esperada, por lo que se mantiene como modelo de embeddings por defecto del prototipo.

El conjunto de evaluación es deliberadamente pequeño y se utiliza para comparar el comportamiento de los modelos dentro de este proyecto. Estos resultados no se presentan como un benchmark general de embeddings.

## Validación end-to-end

El proyecto incluye un workflow de GitHub Actions que:

1. instala las dependencias,
2. compila los módulos de Python,
3. ejecuta un smoke test de retrieval con MiniLM y E5,
4. carga E5 y FLAN-T5,
5. recupera contexto desde Chroma,
6. genera una respuesta no vacía con el pipeline RAG completo.

La validación de runtime finalizó correctamente.

### Limitación observada

El smoke test funcional también permitió detectar un error semántico concreto.

Ante la consulta:

```text
¿Cómo hago para cambiar la fecha de mi turno?
```

el retrieval con E5 y `top_k=3` recuperó documentos sobre datos necesarios para pedir un turno, cancelación y solicitud de turnos. El documento esperado sobre reprogramación no apareció dentro del Top-3. Como consecuencia, FLAN-T5 generó una respuesta relacionada con la cancelación del turno en lugar de explicar cómo reprogramarlo.

Este caso muestra una limitación central de los sistemas RAG: que el pipeline pueda ejecutarse correctamente no implica que la respuesta sea semánticamente correcta. La calidad final depende de la recuperación de contexto y de la cobertura de la base de conocimiento.

Entre las mejoras posibles se encuentran ampliar y reformular la base de conocimiento, evaluar estrategias de reranking y definir mecanismos de rechazo cuando el contexto recuperado no sea suficientemente confiable.

## Origen del proyecto

Este proyecto surge a partir de un trabajo práctico desarrollado durante mi formación en Ciencia de Datos e Inteligencia Artificial. La implementación fue reorganizada desde un notebook académico hacia una estructura modular y ejecutable para documentar el pipeline RAG de forma independiente.

## Autor

**Moisés Lobayza**

Estudiante de Ciencia de Datos e Inteligencia Artificial.
