# PyTensor Más Allá de PyMC: Construyendo Inferencia LLM en Python

> Lo que le falta a PyTensor para inferencia LLM — y lo sencillo que es construirlo. Una exploración nativa en Python de grafos simbólicos, pesos GGUF, C, Numba, MLX, y el camino hacia un runtime LLM componible.

By Carlos Trujillo

Source: https://cetagostini.github.io/es/articles/alchemize_pytensor_mlx_gemma_3n/alchemize_pytensor_mlx_gemma_3n.html

# Introducción

La mayoría de los profesionales conocen [PyTensor](https://pytensor.readthedocs.io/) a través de PyMC. Escribimos un modelo probabilístico, PyMC construye un grafo simbólico y PyTensor compila las matemáticas en algo que una máquina puede ejecutar.

Esa descripción es precisa. También está incompleta.

PyTensor es un compilador simbólico general de tensores. No sabe qué es un prior, y no requiere una verosimilitud. Es el compilador de grafos en el corazón de la programación probabilística, pero su arquitectura nunca estuvo limitada a ese dominio. Este artículo explora qué sucede cuando lo empujamos hacia algo que los autores probablemente no tenían en mente originalmente — inferencia LLM local — y lo que eso nos dice sobre hacia dónde podrían ir PyTensor y el proyecto [pytensor-ml](https://github.com/pymc-labs/pytensor-ml).

Esto es lo que PyTensor ya tiene: ejecución multi-backend (JAX, MLX, Numba, C) y optimización simbólica de grafos que simplifica el cálculo antes de llegar al hardware. La capa de ML vive un nivel arriba, en el ecosistema que lo rodea: [`pytensor-ml`](https://github.com/jessegrabowski/pytensor_ml) ya demuestra inferencia funcional para modelos pequeños, la descuantización GGUF viene de la biblioteca [`gguf`](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py), y el pipeline de [Alchemize](https://github.com/pymc-labs/alchemize) auto-genera módulos PyTensor a partir de metadatos GGUF.

Esto es lo que falta: cacheo KV de producción, batching continuo, carga mmap zero-copy y la optimización de rendimiento que hace que `llama.cpp` sirva modelos a escala. Esas brechas definen la hoja de ruta. Aquí construimos la rebanada vertical completa debajo de ellas: carga de modelo, ejecución simbólica, generación y validación numérica independiente.

La historia que cuenta este artículo no se trata de reemplazar a `llama.cpp`. Se trata de descubrir lo sencillo que es ensamblar un stack de inferencia LLM en PyTensor — pesos, tokenización, un transformador simbólico, un bucle de generación y validación numérica — una vez que el compilador de grafos se libera de sus suposiciones probabilísticas. Y se trata de lo que se vuelve posible cuando esas piezas se unen.

¿No sería genial poder ejecutar lo siguiente completamente en PyTensor? Y si piensas que no, imagina las ventajas:

- **Una definición, cualquier backend.** Las mismas ecuaciones simbólicas se compilan a través de C, Numba, MLX o JAX. Cambiar el objetivo de hardware es un cambio de un argumento, no una reescritura.
- **Inferencia que compone.** El modelo es un grafo dentro del ecosistema científico de Python — encadénalo con un posterior de PyMC, un optimizador de SciPy o cualquier cálculo personalizado, todo en el mismo framework.
- **Un grafo que puedes abrir.** Cada operación es inspeccionable y reescribible. Puedes preguntar qué cambió una reescritura en vez de confiar en una caja negra.

``` sourceCode
from pathlib import Path
from pytensor import load_llm

model = load_llm(
    Path("/path/to/gemma-3n-E4B-it-lm-4bit/snapshot"),
    backend="c", # or any other.
)

result = inference(
    model,
    input="What's causal inference?",
    max_tokens=32,
)
```

Esa pequeña interfaz es nuestro destino.

Para llegar ahí, tenemos que ensamblar el stack completo nosotros mismos. PyTensor posee exactamente una capa — reescribir el grafo y enlazarlo a MLX, C/CVM o Numba — y todo alrededor es Python explícito: adaptadores de pesos que validan, mapean, descuantizan y orientan pesos GGUF o safetensors; adaptadores de tokenizador que aplican la plantilla de chat del modelo y producen IDs de token exactos; un modelo simbólico que expresa normalización, RoPE, atención, rutas residuales y MLPs; un runtime de generación que ejecuta prefill, actualiza el estado KV, elige tokens y se detiene; y una capa de reporte que devuelve texto, tiempos, memoria y verificaciones diferenciales. El primer paso en ese camino es [Alchemize](https://github.com/pymc-labs/alchemize): lee los metadatos GGUF y nos entrega el mapa — el inventario de nombres de tensores, la estructura de bloques y los contratos exactos que faltan. Luego construimos — una pieza a la vez.

Construimos cada pieza contra Gemma 3n E4B a través de C/CVM, Numba y MLX, medimos lo que nos dice la velocidad actual y proyectamos hacia adelante.

El argumento

PyTensor ya tiene las abstracciones de grafo, reescritura y linker para convertirse en el núcleo computacional de un runtime LLM nativo en Python. Lo que falta no es el fundamento — es la ingeniería de producción encima de él. Y construir esa ingeniería en PyTensor es sorprendentemente sencillo.

¿Qué es Alchemize?

[Alchemize](https://github.com/pymc-labs/alchemize) es un transpilador autocorrectivo basado en LLM de PyMC Labs. Actúa como un agente de IA que compila modelos computacionales entre frameworks: PyMC, Stan, JAX, PyTorch y Rust — con validación numérica en cada paso. El agencia razona sobre el grafo computacional completo y aplica optimizaciones que haría un experto del dominio: fusión de bucles, pre-asignación de memoria, patrones de acceso amigables con el cache.

# ¡Lo que Alchemize revela!

Con el destino visible, podemos retroceder y seguir el camino que lo produjo — empezando por lo que falta.

Comenzamos con `SmolLM2-135M-Instruct-Q4_K_M.gguf`, un archivo GGUF de aproximadamente 105 MB. Nuestro primer intento es pedirle a [Alchemize](https://github.com/pymc-labs/alchemize) una implementación en PyTensor:

Show Alchemize call

``` sourceCode
from alchemize import compile_model

result = compile_model(
    model_path=Path("/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf"),
    target="pytensor",
)
```

Alchemize lee los metadatos GGUF y genera un módulo con el esqueleto de arquitectura correcto. Eso es valioso: nos da el mapa de nombres de tensores, el inventario de bloques y la estructura de capas sin escribir nada a mano.

Pero la implementación generada no puede ejecutar. Su suposición central de carga es incorrecta:

Show generated materialize_tensor

``` sourceCode
def materialize_tensor(tensor):
    data = getattr(tensor, "data", None)
    array = np.asarray(data)

    if np.issubdtype(array.dtype, np.floating):
        return np.asarray(array, dtype=np.float32)

    raise NotImplementedError("a dequantizer is required")
```

La mayoría de los tensores en este GGUF no son arreglos de punto flotante. Son valores cuantizados empaquetados. La implementación generada reconoce el problema y se detiene, pero nunca llama a `gguf.dequantize`.

La auditoría estática por lo tanto nos da:

| Check | Result | Why |
|----|----|----|
| Provenance and syntax | pass | the artifact is pinned and valid Python |
| GGUF dequantization | `STATIC_FAIL` | packed weights are never dequantized |
| Attention head reshape | `STATIC_FAIL` audit flag | symbolic reshapes do not preserve the repaired runtime’s static contracts |
| Runtime | `BLOCKED` | generated code is never executed |
| Semantics | `UNVERIFIED` | valid syntax does not establish correct logits |

Esto no es un fracaso. Es un mapa.

Alchemize aceleró el descubrimiento de arquitectura y nos dijo exactamente qué falta. El código generado nos da el esqueleto; la auditoría estática nos dice qué contratos necesitan reemplazos hechos a mano. Conservamos el mapa de nombres de tensores y el inventario de bloques, luego construimos las piezas que llenan los huecos: materialización, orientación, construcción del grafo, cacheo, ejecución y validación.

# Construyendo inferencia de Gemma 3n en PyTensor

Una respuesta correcta depende de una cadena de contratos: descuantización de pesos cuantizados, orientación de tensores, tokenización, embeddings rotacionales, atención de consulta agrupada, cacheo KV y compilación del backend. Rompe un contrato y el modelo puede seguir produciendo texto plausible. Por eso construimos cada contrato explícitamente y los componemos en un pipeline funcional.

Nuestro objetivo es Gemma 3n E4B — un modelo mucho más grande y extraño que un típico primer experimento. Agrega cuatro flujos residuales AltUp, una ruta LAuReL de bajo rango aprendida, embeddings de token por capa, variantes de activación sparse y dense, atención deslizante y completa, y un vocabulario de 262.400 tokens. ¿Podemos expresar todo eso en el lenguaje simbólico de PyTensor sin cambiar el framework?

| Component           | E4B configuration |
|---------------------|------------------:|
| Decoder layers      |                35 |
| Hidden size         |             2,048 |
| Query / KV heads    |             8 / 2 |
| Head dimension      |               256 |
| MLP width           |            16,384 |
| Vocabulary          |           262,400 |
| AltUp streams       |                 4 |
| LAuReL rank         |                64 |
| Sliding window      |               512 |
| Final logit softcap |                30 |

Este caso valida deliberadamente la **generación de múltiples tokens con prefijo completo**. No retiene un cache KV entre pasos de decodificación. En cambio, cada paso reconstruye el prefijo completo, crea la topología shared-KV del modelo para ese forward, y lo descarta después.

## Transmite pesos en lugar de expandir el modelo

El checkpoint de 3.86 GB almacena cada módulo lineal affine-4 como pesos `uint32` empaquetados más escalas y sesgos BF16. Ocho valores de cuatro bits ocupan una palabra:

W\_{r,c} = q\_{r,c}s\_{r,g(c)} + b\_{r,g(c)}.

Aquí, q\_{r,c} es el nibble almacenado en la fila de salida r y la columna de entrada c; s\_{r,g(c)} y b\_{r,g(c)} son la escala y el sesgo para su grupo.

Expandir completamente todos los parámetros lógicos necesitaría aproximadamente **25.6 GiB** solo para pesos FP32. En cambio, nosotros:

- cargar solo las filas de embedding solicitadas,
- descuantizar una capa de decodificador a la vez,
- liberarla antes de cargar la siguiente capa, y
- proyectar los logits de vocabulario en bloques de 4.096 filas.

Show weight streaming

``` sourceCode
from cetagostini.utils.pytensor.weights import Gemma3nWeightLoader

with Gemma3nWeightLoader.from_snapshot(snapshot_path) as loader:
    token_rows = loader.load_input_embedding_rows(token_ids)
    layer_0 = loader.load_layer(0)
```

La transmisión cambia el problema de “mantener el modelo expandido” a “mantener la capa actualmente expandida”.

## Escribe las ecuaciones una sola vez

La normalización compartida es PyTensor ordinario:

Show rmsnorm_symbolic

``` sourceCode
from cetagostini.utils.pytensor.ops import rmsnorm_symbolic

normalized = rmsnorm_symbolic(hidden, gamma, eps=1e-6)
```

El detalle importante no es la fórmula. Es que `rmsnorm_symbolic` no sabe nada sobre C, Numba o MLX.

Lo mismo es cierto para la atención de consulta agrupada, RoPE, máscaras, AltUp y LAuReL. Las rutas GELU sparse y dense de Gemma producen dos `FunctionGraph`s especializados; la atención completa versus deslizante llega como datos de máscara y RoPE. El orden de operaciones sigue la implementación fijada de MLX-LM — incluyendo el aparente residual repetido de LAuReL y el patrón de dispersión GELU sparse leído del checkpoint.

## Elige el backend en la compilación

La selección del backend es ahora una utilidad pequeña y reutilizable:

Show backend selection

``` sourceCode
from cetagostini.utils.pytensor.backends import get_mode

c_mode = get_mode("c")
numba_mode = get_mode("numba")
mlx_mode = get_mode("mlx")

c_layer = pytensor.function(layer_inputs, layer_output, mode=c_mode)
numba_layer = pytensor.function(layer_inputs, layer_output, mode=numba_mode)
mlx_layer = pytensor.function(layer_inputs, layer_output, mode=mlx_mode)
```

La definición del modelo no cambió. Solo cambiaron la política de linker y reescritura.

Un modelo, múltiples experimentos con el compilador

Si cambiamos una ecuación simbólica, cada backend la hereda. Si cambiamos solo una reescritura o linker, el modelo permanece fijo. Esa separación es la contribución principal de PyTensor a este experimento.

## Ejecuta inferencia de Gemma desde Python

El mismo punto de entrada público ahora apunta a un artefacto y backend diferente:

``` sourceCode
from pathlib import Path
from cetagostini.utils.pytensor import Gemma3n, inference

gemma = Gemma3n.from_snapshot(
    Path("/path/to/gemma-3n-E4B-it-lm-4bit/snapshot"),
    backend="c",
)

result = inference(
    gemma,
    input="What's causal inference?",
    max_tokens=32,
)

result.output, result.output_token_ids
```

``` text
('## Causal Inference: Understanding "Why" vs. "Correlation"\n\n'
 'Causal inference is a branch of statistics and statistics is a field '
 'of statistics that aims',
 (1408, 565, 90718, 157036, 236787, 40411, 623, 11355, 236775,
  7728, 236761, 623, 131426, 236775, 108, 236780, 90718, 34711,
  563, 496, 9911, 529, 14906, 532, 14906, 563, 496, 2135,
  529, 14906, 600, 17269))
```

Esa es una continuación real, no una sola predicción de siguiente token. Tampoco es una prosa pulida: la decodificación greedy alcanza el límite de 32 tokens a mitad de oración y se vuelve repetitiva después de que la ruta diferencial se separa. El punto es hacer la generación inspeccionable, no presentar un benchmark de calidad lingüística.

Show validation report

``` sourceCode
result.report.validation
```

``` text
{
    'final_top1_match': True,
    'all_top1_match': False,
    'pearson_mean': 0.998932415848074,
    'top10_overlap_mean': 9.533333333333333,
    'thresholds_passed': True,
    'scope': 'all_generated_prefixes_fresh_cache',
    'all_generated_tokens_match': False,
}
```

El prompt inicial de 15 tokens pasa los umbrales de publicación y selecciona el mismo siguiente token en PyTensor y MLX-LM. La generación entonces valida cada prefijo creciente de PyTensor contra un forward de MLX-LM sobre ese prefijo exacto. Cada llamada al oráculo crea un objeto shared-KV fresco y lo descarta; ningún estado de decodificación sobrevive al siguiente paso.

Las primeras 20 decisiones generadas coinciden. En el token 21, PyTensor elige `9911` mientras el oráculo MLX de prefijo fresco elige `2135`. El reporte registra esa divergencia y sigue evaluando MLX en el **prefijo de PyTensor** posterior, por lo que las comparaciones posteriores permanecen bien definidas. No compara dos flujos derivando independientemente, y ya no descarta una generación completada costosa porque `stream_generate` cacheado sigue una ruta diferente.

| Multi-token result              |        Value |
|---------------------------------|-------------:|
| Visible tokens                  |           32 |
| Matching fresh-prefix decisions |      31 / 32 |
| First divergence                |     token 21 |
| Prefix graph compilations       |           32 |
| Additional generation wall time |  1,612.038 s |
| Peak process RSS                | 9,668.61 MiB |
| Stop reason                     | `max_tokens` |

El costo es la lección. Después del paso inicial validado del prompt, el bucle restante de generación de prefijo completo tomó **1.612,038 segundos**, o aproximadamente **26,9 minutos**. Esta es una estrategia de ejecución educativa que expone cada grafo y comparación; no es un decodificador eficiente.

### C/CVM vs Numba vs MLX: comparación de rendimiento

Los tres backends compilan las mismas ecuaciones simbólicas y seleccionan el mismo token top-1 que el oráculo independiente de MLX-LM en las 20 posiciones del prompt. Su correlación de Pearson media de todos los logits con el oráculo es 0.9986, y su superposición media del top-10 es 9.7 de 10.

| Metric                          |         C/CVM |     Numba |          MLX |
|---------------------------------|--------------:|----------:|-------------:|
| Graph compilation               |       7.738 s |   0.775 s |  **0.755 s** |
| One full forward (20 positions) |      51.441 s |  53.800 s | **50.211 s** |
| Mean per-layer time             |       1.272 s |   1.266 s |  **1.246 s** |
| Logit projection                |   **3.024 s** |   4.700 s |      3.086 s |
| Whole-process peak RSS          | **4,523 MiB** | 4,663 MiB |    5,111 MiB |

Estas son mediciones de una sola ejecución de un pipeline educativo transmitido intencionalmente, no un benchmark de throughput. MLX compila aproximadamente 10 veces más rápido que C y registra el forward más corto, pero solo por un 2.4%. C todavía gana la proyección del vocabulario de 262.400. La descuantización de pesos, la orquestación en Python y la transmisión por capa dominan lo suficiente como para que los tres forwards completos permanezcan en el mismo rango.

Cómo la sonda MLX fallida se convirtió en un backend funcional

La primera sonda MLX falló porque el borrador generado dependía de formas y operaciones que el linker fijado no podía reducir de forma segura. No parcheamos el paquete PyTensor instalado ni mutamos un registro de despacho global de proceso. En su lugar, expresamos las proyecciones como multiplicaciones de matriz de rango 2 con reshapes explícitos, mantuvimos la atención en primitivas ordinarias de tensores, y reemplazamos los sitios de clip de AltUp de Gemma con un ayudante simbólico local del repositorio construido con comparaciones y `where`.

El grafo resultante se compila a través de `pytensor.compile.mode.MLX` integrado en PyTensor 3.1.2. MLX evalúa de forma lazy, así que el runner materializa 103 límites ordenados: dos proyecciones iniciales, 35 capas de decodificador, un desincrustar final y 65 bloques de vocabulario. Los tensores intermedios permanecen residentes en el dispositivo; solo los bloques de logits completados cruzan de vuelta a arreglos NumPy propietarios. El pico medido del asignador MLX fue 455 MiB por encima de su línea base cercana a cero, mientras que el RSS de todo el proceso alcanzó 5.111 MiB.

Los reportes completos están disponibles para el [oráculo independiente de MLX-LM](results/gemma3n_mlx_lm_oracle.json), [C/CVM](results/gemma3n_pytensor_c.json), [Numba](results/gemma3n_pytensor_numba.json) y [MLX](results/gemma3n_pytensor_mlx.json). Un validador separado recargó los logits crudos temporales, recalculó cada métrica en vez de confiar en los reportes, y pasó [896 de 896 pruebas](results/gemma3n_report_validation.json). Los arreglos crudos grandes no están comprometidos; los reportes preservan sus formas, dtypes, conteos de bytes y hashes criptográficos. La ejecución anterior de generación de 32 tokens permanece en [`gemma3n_pytensor_generation.json`](results/gemma3n_pytensor_generation.json).

# Lo que nos dice la velocidad actual

Los números anteriores no son benchmarks. Son mediciones de un pipeline educativo, y deberían leerse así. Pero son honestos, y nos dicen algo preciso sobre hacia dónde necesita dirigirse el esfuerzo de ingeniería.

Gemma 3n a través de regeneración de prefijo completo funciona a aproximadamente **0.02 tokens por segundo**. Ese es el costo de recompilar y reevaluar el prefijo completo para cada token generado. Es deliberadamente costoso — valida corrección — pero no es como funcionaría un sistema de producción.

La hoja de ruta de ingeniería desde aquí es clara:

| Bottleneck | Current state | What unlocks it |
|----|----|----|
| KV cache | fixed-capacity, O(C) write per layer | paged or ring-buffer cache, continuous batching |
| Weight loading | per-layer streaming (dequantize on demand) | mmap zero-copy, quantized kernels |
| Backend execution | C/CVM, Numba, and MLX on Apple Silicon | backend-native quantized kernels and less host-device movement |
| Graph compilation | recompilation per prefix length | cached compiled functions per shape, or JIT |

`llama.cpp` ha dedicado años a cada fila de esa tabla. PyTensor tiene el compilador de grafos y la arquitectura multi-backend; todavía no tiene la infraestructura de serving. La pregunta no es si PyTensor puede igualar el throughput de `llama.cpp` hoy — no puede — sino si las piezas están en su lugar para construir esa infraestructura en Python. La respuesta, después de este experimento, es sí.

La ventaja de composabilidad

`llama.cpp` es un motor de inferencia autocontenido. PyTensor es un compilador de grafos que puede componerse con transformaciones de JAX, operaciones de NumPy, optimizadores de SciPy y el resto del stack científico de Python. En el momento en que la inferencia LLM vive dentro de ese ecosistema, puedes encadenarla con análisis Bayesiano, optimización basada en gradientes o cálculo simbólico personalizado — flujos de trabajo para los que un motor C++ independiente nunca fue diseñado para soportar.

# Donde PyTensor y llama.cpp difieren — y por qué importa

Ahora podemos hacer la comparación precisa:

| Capability | This PyTensor stack | `llama.cpp` |
|----|----|----|
| Inspectable symbolic graph | yes | not its primary user abstraction |
| User-defined graph rewrites | yes | no equivalent Python rewrite database |
| C, Numba, and MLX experiments | demonstrated on Gemma 3n E4B | purpose-built CPU, Metal, CUDA, and other backends |
| GGUF parsing | supplied by `gguf-py` adapter | built in |
| Native quantized matmul | not implemented here | built in |
| Tokenization and chat templates | Python adapters | built in |
| Autoregressive loop | Python, model-specific | built in |
| Production KV cache and batching | no | built in |
| Broad architecture support | one validated fixture (Gemma 3n) | broad, maintained model coverage |
| Composability with scientific Python | native | not designed for it |

`llama.cpp` gana en ingeniería de inferencia integrada. Eso es exactamente para lo que fue construido.

Pero el ecosistema local de LLM no se trata solo de ejecutar un modelo rápido. Se trata de lo que haces con el modelo después.

La historia reciente de Ollama — [documentada extensamente por sleepingrobots](https://sleepingrobots.com/dreams/stop-using-ollama/) — muestra qué sucede cuando un wrapper oscurece sus dependencias y pivota a la nube. El ecosistema necesita alternativas construidas sobre fundamentos honestos y transparentes. PyTensor y `llama.cpp` son esos fundamentos.

- **`llama.cpp`** es el motor en C++ para ejecutar cualquier modelo GGUF rápido. Posee todo el stack de serving: kernels cuantizados, gestión de cache KV, batching continuo y soporte amplio de arquitecturas. Si necesitas servir modelos a escala de producción hoy, `llama.cpp` es la respuesta.
- **PyTensor** es el compilador de grafos nativo en Python para estudiar, modificar y componer inferencia LLM. No solo ejecutas un modelo — puedes preguntar qué cambió una reescritura, compilar las mismas ecuaciones a través de otro linker, encadenar la inferencia con un posterior Bayesiano o un bucle de optimización personalizado, y validar cada contrato numéricamente.

La afirmación honesta no es “PyTensor reemplaza a `llama.cpp`”. Es esta:

PyTensor más adaptadores explícitos en Python es una *alternativa nativa en Python* creíble a `llama.cpp` que **compone con el stack de computación científica** — análisis Bayesiano, optimización, grafos de cálculo personalizados — de maneras para las que un motor C++ independiente nunca fue diseñado.

Si alguien construye cacheo KV, carga mmap zero-copy y batching continuo sobre el backend JAX de PyTensor, y [Alchemize](https://github.com/pymc-labs/alchemize) madura para auto-generar nuevas arquitecturas a partir de metadatos GGUF, podríamos ver a PyTensor convertirse en el “lenguaje” para ejecutar múltiples LLMs como lo es `llama.cpp` hoy. No más rápido en inferencia — o quizás en algún momento — pero más capaz como un runtime LLM de propósito general que se conecta a flujos de trabajo para los que `llama.cpp` nunca fue diseñado.

# Conclusiones

1.  **PyTensor es un compilador de grafos general.** PyMC es su consumidor más visible, no el límite de lo que puede expresar.
2.  **Construir inferencia LLM en PyTensor es sencillo.** Cargadores de pesos, transformadores simbólicos, caches KV, bucles de generación y validación numérica — todo ensamblado desde Python sin modificar el framework.
3.  **Una definición simbólica sobrevive a múltiples backends.** Gemma 3n se ejecuta a través de C/CVM, Numba y MLX sin una segunda implementación del modelo.
4.  **Los logits independientes superan a la prosa plausible.** Tokens exactos y concordancia numérica en todas las posiciones son evidencia más fuerte que un texto que simplemente parece correcto.
5.  **PyTensor y `llama.cpp` son complementarios.** `llama.cpp` domina el serving en producción. PyTensor domina la transparencia del grafo y la composabilidad con Python científico.
6.  **Las brechas restantes son de ingeniería, no de arquitectura.** Kernels cuantizados nativos, cacheo KV paginado, carga mmap zero-copy y cobertura más amplia de backends son todo construible sobre el fundamento existente de PyTensor. El proyecto pytensor-ml ya ha comenzado este trabajo.

Lecturas recomendadas:

1.  [Documentación de PyTensor](https://pytensor.readthedocs.io/)
2.  [Reescritura de grafos de PyTensor](https://pytensor.readthedocs.io/en/latest/extending/graph_rewriting.html)
3.  [Modos de compilación de PyTensor](https://pytensor.readthedocs.io/en/latest/library/compile/mode.html)
4.  [llama.cpp](https://github.com/ggml-org/llama.cpp)
5.  [MLX-LM](https://github.com/ml-explore/mlx-lm)
6.  [Alchemize](https://github.com/pymc-labs/alchemize)
7.  [pytensor-ml](https://github.com/pymc-labs/pytensor-ml)
8.  [Friends Don’t Let Friends Use Ollama](https://sleepingrobots.com/dreams/stop-using-ollama/)

El experimento registrado utilizó:

| Component   | Version or pin                             |
|-------------|--------------------------------------------|
| Python      | `3.13.14`                                  |
| PyTensor    | `3.1.2`                                    |
| PyTensor-ML | `f6ecf81d58da180cce50b77a43cf5d2c3d95e470` |
| MLX         | `0.32.0`                                   |
| Numba       | `0.65.1`                                   |
| MLX-LM      | `0.31.3`                                   |
| Machine     | Apple M3 Max, 128 GB unified memory        |

El entorno exacto está registrado en [`environment.yml`](environment.yml). Las celdas de código no se ejecutan durante la compilación del sitio web porque los artefactos del modelo están intencionalmente excluidos de Git; los reportes JSON comprometidos preservan los hashes de los artefactos y las salidas medidas. La suite de PyTensor pasó 934 pruebas con 30 omisiones, y las pruebas del runner con modelo real pasaron las 189 verificaciones; el registro compacto está en [`gemma3n_test_summary.json`](results/gemma3n_test_summary.json).

------------------------------------------------------------------------

El paquete público, las implementaciones específicas del modelo, las pruebas, el borrador generado, las auditorías y los reportes de resultados sanitizados están disponibles en el [repositorio del sitio](https://github.com/cetagostini/cetagostini.github.io).
