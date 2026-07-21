# RAG Medical Assistant

Sistema de preguntas y respuestas basado en **Retrieval-Augmented Generation
(RAG)** para consultas administrativas de un consultorio médico privado.

El proyecto transforma una consulta en un embedding, recupera contexto relevante
desde una base vectorial en memoria y utiliza un modelo generativo para redactar
una respuesta en español basada en ese contexto.

> **Alcance:** el asistente trabaja con información administrativa. No realiza
> diagnósticos ni reemplaza la evaluación de un profesional de la salud.

## Estado actual

La versión actual ofrece:

- una interfaz interactiva por terminal;
- componentes separados para configuración, conocimiento, retrieval, generación,
  orquestación y evaluación;
- evaluación de modelos de embeddings independiente del generador;
- 54 casos de test deterministas que pueden ejecutarse completamente offline;
- smoke tests reales de retrieval y del pipeline RAG completo en GitHub Actions.

Todavía **no existen** una API HTTP o FastAPI, endpoints de consulta o health
check, configuración mediante variables de entorno, `.env.example`, Dockerfile,
manifiestos ni servicio de despliegue. Tampoco hay persistencia del índice,
autenticación, rate limiting u observabilidad de producción.

## Arquitectura modular

```text
main.py (composition root)
  |
  +-- DEFAULT_RAG_CONFIG
  |
  +-- src.cli.main
        |
        +-- MedicalRAGChatbot (fachada y composición concreta)
              |
              +-- knowledge_base.py
              |     +-- carga y validación del CSV
              |     +-- documentos, IDs y metadatos
              |
              +-- RAGService
                    +-- ChromaRetriever
                    |     +-- Sentence Transformer
                    |     +-- prefijos E5 y embeddings normalizados
                    |     +-- Chroma EphemeralClient
                    |
                    +-- FlanT5Generator
                          +-- PyTorch y selección de dispositivo
                          +-- tokenizer y FLAN-T5

evaluation.py
  +-- reutiliza knowledge_base.py
  +-- construye un ChromaRetriever por modelo
  +-- produce summary y details sin cargar FLAN-T5
```

`MedicalRAGChatbot` se conserva como fachada compatible para la CLI y para el
uso programático. La coordinación retrieval -> contexto -> generación reside en
`RAGService`, que depende de contratos inyectables y no de implementaciones
concretas.

## Responsabilidades de los módulos

| Módulo | Responsabilidad actual |
|---|---|
| `src/config.py` | Define `RAGConfig`, una dataclass inmutable con los defaults actuales de rutas, modelos, Chroma, tokenización y generación. No lee variables de entorno. |
| `src/knowledge_base.py` | Carga el CSV, valida columnas y registros, limpia espacios y construye documentos, IDs y metadatos. |
| `src/retrieval.py` | Carga embeddings de forma diferida, aplica los prefijos E5, normaliza vectores, crea el índice Chroma con distancia coseno y ejecuta búsquedas Top-K. |
| `src/generation.py` | Carga PyTorch y Transformers de forma diferida, selecciona CPU/CUDA, construye el prompt, tokeniza, ejecuta FLAN-T5 y limpia la respuesta. |
| `src/service.py` | Valida la consulta, coordina retrieval, conserva el orden del contexto y delega la generación. |
| `src/chatbot.py` | Compone conocimiento, retriever y generador; mantiene `MedicalRAGChatbot` y sus atributos históricos como fachada compatible. |
| `src/cli.py` | Implementa el bucle interactivo, comandos de salida, consultas vacías y manejo de excepciones; permite inyectar fábrica, entrada y salida para tests. |
| `src/evaluation.py` | Compara retrieval entre modelos usando los componentes compartidos de conocimiento y Chroma, sin construir el generador. Devuelve DataFrames de resumen y detalle. |

## Flujo completo de una consulta

### Inicialización

1. `main.py` toma la ruta, los modelos y `top_k` desde `DEFAULT_RAG_CONFIG`.
2. La CLI solicita construir `MedicalRAGChatbot`.
3. `knowledge_base.py` carga el CSV, valida `pregunta` y `respuesta`, descarta
   filas inválidas y limpia espacios.
4. Se construyen los documentos, IDs y metadatos en el orden del CSV.
5. `ChromaRetriever` carga el modelo de embeddings.
6. Para E5, agrega `passage:` únicamente al texto enviado al embedder.
7. Los documentos se convierten en embeddings normalizados.
8. Se crea una colección Chroma efímera con similitud coseno y se indexan los
   documentos originales.
9. `FlanT5Generator` carga PyTorch, el tokenizer y FLAN-T5, y selecciona CUDA si
   está disponible o CPU en caso contrario.

### Consulta y respuesta

1. La CLI recorta la entrada y reconoce `salir`, `exit` y `quit` sin distinguir
   mayúsculas.
2. `RAGService` rechaza una consulta vacía antes de retrieval o generación.
3. El retriever recorta la consulta y, para E5, aplica el prefijo `query:`.
4. Se genera un embedding normalizado y Chroma recupera
   `min(top_k, cantidad_de_documentos)` resultados.
5. `RetrievalResult` conserva documentos, metadatos y distancias en el orden
   devuelto por Chroma.
6. El servicio une los documentos recuperados con dos saltos de línea.
7. El generador incorpora ese contexto y la consulta original al prompt.
8. El prompt se tokeniza con truncado y `max_length=512`.
9. FLAN-T5 genera sin muestreo, con `num_beams=4`, `do_sample=False` y hasta
   100 tokens nuevos dentro de
   `torch.inference_mode()`.
10. La respuesta se decodifica, se recortan espacios y la CLI la muestra. Un
    error de respuesta se informa sin finalizar la conversación.

## Tecnologías y modelos

- Python
- Pandas
- Pytest
- Sentence Transformers
- Hugging Face Transformers
- Chroma
- PyTorch
- FLAN-T5
- GitHub Actions

Defaults actuales:

- **Embeddings:** `intfloat/multilingual-e5-small`
- **Generación:** `google/flan-t5-small`
- **Top-K:** `3`
- **Distancia de Chroma:** coseno
- **Colección principal:** `medical_office_knowledge`

La evaluación histórica también compara:

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `intfloat/multilingual-e5-small`

## Base de conocimiento

La base contiene preguntas y respuestas administrativas sobre:

- turnos y confirmaciones;
- horarios de atención;
- ubicación;
- obra social y atención particular;
- cancelaciones y reprogramaciones;
- recetas y certificados;
- formas de pago y facturación;
- estudios previos y resultados.

Archivo actual:

```text
data/knowledge_base.csv
```

Cada registro válido se transforma en:

```text
Pregunta frecuente: <pregunta>
Respuesta: <respuesta>
```

La ruta por defecto es relativa al directorio de ejecución. Los comandos de este
README deben ejecutarse desde la raíz del repositorio.

## Estructura del proyecto

```text
rag-medical-assistant/
|-- .github/
|   `-- workflows/
|       `-- validate.yml
|-- data/
|   `-- knowledge_base.csv
|-- notebooks/
|   `-- embedding_experiments.ipynb
|-- src/
|   |-- __init__.py
|   |-- chatbot.py
|   |-- cli.py
|   |-- config.py
|   |-- evaluation.py
|   |-- generation.py
|   |-- knowledge_base.py
|   |-- retrieval.py
|   `-- service.py
|-- tests/
|   |-- conftest.py
|   |-- test_chatbot.py
|   |-- test_cli.py
|   |-- test_config.py
|   |-- test_evaluation.py
|   |-- test_generation.py
|   |-- test_knowledge_base.py
|   |-- test_main.py
|   |-- test_retrieval.py
|   `-- test_service.py
|-- main.py
|-- requirements-dev.txt
|-- requirements.txt
`-- README.md
```

## Instalación

GitHub Actions valida el proyecto con Python 3.11. Desde la raíz del repositorio:

```bash
python -m venv .venv
```

Activar el entorno en Linux o macOS:

```bash
source .venv/bin/activate
```

En PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Para ejecutar la CLI o una evaluación real, instalar las dependencias de
runtime:

```bash
python -m pip install -r requirements.txt
```

Para ejecutar además los tests locales:

```bash
python -m pip install -r requirements-dev.txt
```

## Ejecutar la CLI

```bash
python main.py
```

La primera ejecución puede descargar los modelos desde Hugging Face y tardar
varios minutos. Las siguientes pueden utilizar la caché local.

Para cerrar el chat se puede escribir cualquiera de estos comandos:

```text
salir
exit
quit
```

## Tests y validaciones

### Tests unitarios deterministas y offline

La suite actual contiene **54 casos de test**. Usa dobles para embeddings,
Chroma, tokenizer, modelo generativo y PyTorch. Los stubs de `tests/conftest.py`
fallan si un test unitario intenta utilizar una integración real.

Por eso esta suite:

- no descarga modelos;
- no contacta Hugging Face;
- no necesita GPU;
- protege contratos observables de CLI, configuración, conocimiento, retrieval,
  generación, servicio y evaluación.

Ejemplo en Bash:

```bash
python -m pip install -r requirements-dev.txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest
```

Ejemplo en PowerShell:

```powershell
python -m pip install -r requirements-dev.txt
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
python -m pytest
```

Validación de sintaxis utilizada durante el desarrollo:

```bash
python -m compileall main.py src tests
```

### Smoke tests reales

El workflow `.github/workflows/validate.yml` ejecuta primero los tests
deterministas. Después instala las dependencias de runtime y realiza:

1. un smoke test de retrieval con MiniLM y E5 sobre tres consultas, usando
   Top-1 y Top-3;
2. un smoke test end-to-end que carga E5 y FLAN-T5, recupera contexto y exige
   una respuesta no vacía.

Estos smokes sí pueden descargar modelos, requieren acceso de red y consumen más
tiempo y memoria. Verifican que las integraciones reales funcionen, pero no
demuestran que cada ranking o respuesta sea semánticamente correcto. Tampoco
reproducen la tabla histórica de diez consultas mostrada más abajo.

El workflow se ejecuta en pushes y pull requests hacia `main`, y también puede
iniciarse manualmente mediante `workflow_dispatch`.

## Evaluación de embeddings

`src/evaluation.py` compara retrieval entre modelos sin cargar PyTorch,
Transformers ni FLAN-T5. Reutiliza la misma carga de conocimiento y el mismo
`ChromaRetriever` que el asistente.

Una evaluación real sí construye Sentence Transformers e índices Chroma. Con
las dependencias de runtime instaladas, se puede ejecutar desde un intérprete o
script Python en la raíz del repositorio:

```python
from pathlib import Path

from src.evaluation import EvaluationCase, compare_embedding_models

cases = [
    EvaluationCase(
        query="Necesito cambiar la fecha de mi cita, ¿qué hago?",
        expected_question="¿Puedo reprogramar mi turno?",
    ),
    EvaluationCase(
        query="¿Con qué medios puedo abonar la consulta?",
        expected_question="¿Cuáles son las formas de pago?",
    ),
]

summary, details = compare_embedding_models(
    knowledge_base_path=Path("data/knowledge_base.csv"),
    embedding_models=[
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "intfloat/multilingual-e5-small",
    ],
    cases=cases,
    top_k_values=(1, 3, 5),
)

print(summary.to_string(index=False))
print(details.to_string(index=False))
```

La evaluación produce:

- `details`: una fila por consulta y corte Top-K, con el resultado superior,
  distancia, hit y posición 1-based del documento esperado;
- `summary`: `hit_rate`, `hits`, `total` y `mean_expected_rank` agrupados por
  modelo y Top-K.

Los casos con `hit=False` permiten inspeccionar errores de recuperación; no
existe una métrica separada llamada `retrieval_error`.

El experimento de diez consultas se encuentra en
`notebooks/embedding_experiments.ipynb`.

### Resultados históricos reportados

La siguiente tabla es un snapshot histórico reportado para el conjunto manual de
diez consultas del prototipo. Se conserva como registro de la documentación
previa, pero el repositorio no incluye el artefacto de aquella ejecución. El smoke
test actual de GitHub Actions usa tres consultas y no reproduce esta tabla.

| Modelo | Top-1 | Top-3 | Top-5 | Posición media esperada |
|---|---:|---:|---:|---:|
| MiniLM multilingual | 80% | 80% | 100% | 1.8 |
| multilingual-E5-small | 80% | 90% | 100% | 1.4 |

En ese conjunto pequeño, ambos modelos alcanzaron el mismo resultado en Top-1 y
Top-5. E5 obtuvo una mayor tasa de aciertos en Top-3 y una mejor posición media
de la pregunta esperada, por lo que se mantiene como modelo de embeddings por
defecto del prototipo.

El conjunto es deliberadamente pequeño y manual. Estos resultados no constituyen
un benchmark general de embeddings ni garantizan el comportamiento de versiones
futuras de los modelos o dependencias.

### Semántica actual y ambigüedades de las métricas

- `hit_rate@k` es la proporción de consultas cuyo documento esperado aparece
  dentro del Top-K efectivo.
- `expected_rank` se calcula una vez usando el máximo Top-K consultado y se copia
  a todos los cortes. Por eso `mean_expected_rank` puede incluir, por ejemplo,
  una posición 5 dentro de la fila agregada de Top-1 aunque ese caso sea un miss
  en Top-1.
- Los documentos no recuperados tienen rango `NaN`; pandas los excluye de
  `mean_expected_rank`. La métrica no penaliza misses completos y no es MRR.
- Si el corpus es menor que los Top-K solicitados, varios cortes pueden colapsar
  al mismo valor efectivo. Actualmente esas filas duplicadas se agrupan y pueden
  inflar `hits` y `total`.
- La pregunta esperada se compara por coincidencia exacta y solo se admite una
  respuesta relevante por caso.
- El resumen se ordena por `hit_rate` descendente y luego por rango medio, no por
  el orden de Top-K solicitado.

Estos comportamientos están cubiertos por tests de caracterización y permanecen
documentados como deuda técnica; no se modificaron en esta etapa.

## Imports diferidos

Las integraciones pesadas se importan cuando se construyen sus componentes, no
al importar los módulos:

- `src/retrieval.py` carga Sentence Transformers y Chroma al crear el retriever;
- `src/generation.py` carga PyTorch y Transformers al crear el generador;
- `src/evaluation.py` reutiliza retrieval y no importa el generador.

Así, importar `src`, `src.retrieval`, `src.generation` o `src.evaluation` no carga
Chroma, Sentence Transformers, Transformers ni PyTorch. Esto evita cargar modelos
y dependencias ML pesadas como efecto lateral del import, hace posibles los tests
offline y permite reutilizar los componentes por separado. Construir el asistente
o iniciar una evaluación real sí carga las dependencias y puede descargar modelos
si no están en caché.

## Limitación semántica conocida

La base de conocimiento contiene respuestas diferentes para cancelar y
reprogramar un turno. Sin embargo, en una ejecución observada con E5 y `top_k=3`,
la consulta:

```text
¿Cómo hago para cambiar la fecha de mi turno?
```

no recuperó la entrada de reprogramación dentro de los tres primeros resultados.
El contexto incluyó documentos sobre solicitud y cancelación de turnos, y
FLAN-T5 generó una respuesta relacionada con cancelar en lugar de explicar cómo
reprogramar.

Es un fallo de ranking y selección de contexto, no una ausencia de la respuesta
en el corpus. El sistema actual no incorpora reranking, clasificación de
intención, umbral de confianza ni abstención automática. Un pipeline técnicamente
exitoso no garantiza una respuesta semánticamente correcta.

## Limitaciones y deuda técnica

- Chroma usa `EphemeralClient`: el índice vive en memoria y se reconstruye en
  cada proceso.
- `compare_embedding_models` vuelve a leer la base y crea un índice independiente
  por modelo.
- La ruta por defecto del CSV depende del directorio actual.
- No hay reranking, umbral de confianza, citas de fuentes ni validación de la
  respuesta generada.
- El conjunto de evaluación es pequeño, manual y no incluye intervalos de
  confianza.
- Los modelos y la mayoría de las dependencias runtime no fijan versión, por lo
  que resultados y tiempos pueden variar.
- La configuración todavía no se obtiene de variables de entorno.
- No existe API, Docker, persistencia, autenticación, observabilidad ni despliegue
  preparado para producción.

## Origen del proyecto

Este proyecto surge de un trabajo práctico desarrollado durante una formación en
Ciencia de Datos e Inteligencia Artificial. La implementación fue reorganizada
desde un notebook académico hacia una aplicación CLI modular y verificable.

## Autor

**Moisés Lobayza**

Estudiante de Ciencia de Datos e Inteligencia Artificial.
