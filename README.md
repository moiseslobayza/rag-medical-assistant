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

## Modelos

El sistema utiliza por defecto:

- **LLM:** `google/flan-t5-small`
- **Embeddings:** `intfloat/multilingual-e5-small`

La implementación contempla el formato recomendado por la familia E5, utilizando los prefijos `query:` para consultas y `passage:` para documentos.

El trabajo original también comparó el modelo E5 con `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. La evaluación comparativa se incorporará como una capa separada del proyecto.

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
├── data/
│   └── knowledge_base.csv
│
├── src/
│   ├── __init__.py
│   └── chatbot.py
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

## Origen del proyecto

Este proyecto surge a partir de un trabajo práctico desarrollado durante mi formación en Ciencia de Datos e Inteligencia Artificial. La implementación fue reorganizada desde un notebook académico hacia una estructura modular y ejecutable para documentar el pipeline RAG de forma independiente.

## Autor

**Moisés Lobayza**

Estudiante de Ciencia de Datos e Inteligencia Artificial.
