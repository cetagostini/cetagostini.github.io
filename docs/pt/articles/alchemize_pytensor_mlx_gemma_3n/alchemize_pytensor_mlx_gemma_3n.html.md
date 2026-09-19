# PyTensor para além do PyMC: construir inferência de LLM em Python

> O que falta ao PyTensor para inferência de LLM — e como é direto construí-lo. Uma exploração nativa de Python de grafos simbólicos, pesos GGUF, C, Numba, MLX, e o caminho para um tempo de execução de LLM composável.

By Carlos Trujillo

Source: https://cetagostini.github.io/pt/articles/alchemize_pytensor_mlx_gemma_3n/alchemize_pytensor_mlx_gemma_3n.html

# Introdução

A maioria dos profissionais conhece [PyTensor](https://pytensor.readthedocs.io/) através do PyMC. Escrevemos um modelo probabilístico, o PyMC constrói um grafo simbólico, e o PyTensor compila a matemática em algo que uma máquina pode executar.

Essa descrição é precisa. Mas está incompleta.

PyTensor é um compilador de tensores simbólicos generalista. Não sabe o que é uma prior, e não requer uma verosimilhança. É o compilador de grafos no coração da programação probabilística, mas a sua arquitetura nunca foi limitada a esse domínio. Este artigo explora o que acontece quando o empurramos para algo que os autores podem não ter originalmente previsto — inferência de LLM local — e o que isso nos diz sobre onde PyTensor, e o projeto [pytensor-ml](https://github.com/pymc-labs/pytensor-ml), poderão ir a seguir.

Eis o que o PyTensor já tem: execução multi-backend (JAX, MLX, Numba, C) e otimização simbólica de grafos que simplifica a computação antes de chegar ao hardware. A camada de ML vive um nível acima, no ecossistema à sua volta: [`pytensor-ml`](https://github.com/jessegrabowski/pytensor_ml) já demonstra inferência funcional para modelos pequenos, a desquantização GGUF vem da biblioteca [`gguf`](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py), e o pipeline [Alchemize](https://github.com/pymc-labs/alchemize) gera automaticamente módulos PyTensor a partir de metadados GGUF.

Eis o que falta: caching KV em produção, batching contínuo, carregamento mmap zero-copy, e a sintonização de desempenho que faz `llama.cpp` servir modelos à escala. Essas lacunas definem o roteiro. Aqui construímos a fatia vertical completa por baixo delas: carregamento do modelo, execução simbólica, geração e validação numérica independente.

A história que este artigo conta não é sobre substituir `llama.cpp`. É sobre descobrir como é direto montar uma pilha de inferência de LLM em PyTensor — pesos, tokenização, um transformador simbólico, um ciclo de geração e validação numérica — quando o compilador de grafos é libertado das suas suposições probabilísticas. E é sobre o que se torna possível quando essas peças se juntam.

Seria realmente fantástico poder executar o seguinte em pytensor completo, não é? E se acha que não seria, imagine as vantagens:

- **Uma definição, qualquer backend.** As mesmas equações simbólicas compilam através de C, Numba, MLX ou JAX. Trocar o destino de hardware é uma alteração de um argumento, não uma reescrita.
- **Inferência que se compõe.** O modelo é um grafo dentro do ecossistema científico de Python — encadeie-o com um posterior de PyMC, um otimizador de SciPy ou qualquer computação personalizada, tudo na mesma framework.
- **Um grafo que pode inspecionar.** Cada operação é audível e reescrevível. Pode perguntar o que uma reescrita alterou em vez de confiar numa caixa negra.

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

Essa pequena interface é o nosso destino.

Para lá chegar, temos de montar a pilha completa nós mesmos. PyTensor detém exatamente uma camada — reescrever o grafo e ligá-lo a MLX, C/CVM ou Numba — e tudo à sua volta é Python explícito: adaptadores de pesos que validam, mapeiam, desquantizam e orientam pesos GGUF ou safetensors; adaptadores de tokenizer que aplicam o template de chat do modelo e produzem IDs de token exatos; um modelo simbólico que expressa normalização, RoPE, atenção, caminhos residuais e MLPs; um tempo de execução de geração que executa prefill, atualiza o estado KV, escolte tokens e pára; e uma camada de relatório que devolve texto, tempos, memória e verificações diferenciais. O primeiro passo nesse caminho é [Alchemize](https://github.com/pymc-labs/alchemize): lê os metadados GGUF e entrega-nos o mapa — o inventário de nomes de tensores, a estrutura de blocos e os contratos exatos que estão em falta. Depois construímos — uma peça de cada vez.

Construímos cada peça contra Gemma 3n E4B através de C/CVM, Numba e MLX, medimos o que a velocidade atual nos diz, e projetamos para a frente.

O argumento

PyTensor já tem as abstrações de grafo, reescrita e linker para se tornar o núcleo computacional de um tempo de execução de LLM nativo de Python. O que falta não é a fundação — é a engenharia de produção por cima dela. E construir essa engenharia em PyTensor é surpreendentemente direto.

O que é o Alchemize?

[Alchemize](https://github.com/pymc-labs/alchemize) é um transpilador auto-corretivo baseado em LLM do PyMC Labs. Age como um agente de IA que compila modelos computacionais entre frameworks: PyMC, Stan, JAX, PyTorch e Rust — com validação numérica em cada passo. O agente raciocina sobre o grafo computacional completo e aplica otimizações que um especialista de domínio aplicaria: fusão de ciclos, pré-allocação de memória, padrões de acesso amigáveis à cache.

# O que o Alchemize revela!

Com o destino visível, podemos retroceder e seguir o caminho que o produziu — começando pelo que falta.

Começamos com `SmolLM2-135M-Instruct-Q4_K_M.gguf`, um ficheiro GGUF de aproximadamente 105 MB. A nossa primeira tentativa é pedir [Alchemize](https://github.com/pymc-labs/alchemize) uma implementação PyTensor:

Show Alchemize call

``` sourceCode
from alchemize import compile_model

result = compile_model(
    model_path=Path("/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf"),
    target="pytensor",
)
```

Alchemize lê os metadados GGUF e gera um módulo com o esqueleto de arquitetura correto. Isso é valioso: fornece-nos o mapa de nomes de tensores, o inventário de blocos e a estrutura de camadas sem escrever nada disso manualmente.

Mas a implementação gerada não consegue executar. A sua suposição central de carregamento está errada:

Show generated materialize_tensor

``` sourceCode
def materialize_tensor(tensor):
    data = getattr(tensor, "data", None)
    array = np.asarray(data)

    if np.issubdtype(array.dtype, np.floating):
        return np.asarray(array, dtype=np.float32)

    raise NotImplementedError("a dequantizer is required")
```

A maioria dos tensores neste GGUF não são matrizes de vírgula flutuante. São valores quantizados empacotados. A implementação gerada reconhece o problema e pára, mas nunca chama `gguf.dequantize`.

A auditoria estática dá-nos, portanto:

| Check | Result | Why |
|----|----|----|
| Provenance and syntax | pass | the artifact is pinned and valid Python |
| GGUF dequantization | `STATIC_FAIL` | packed weights are never dequantized |
| Attention head reshape | `STATIC_FAIL` audit flag | symbolic reshapes do not preserve the repaired runtime’s static contracts |
| Runtime | `BLOCKED` | generated code is never executed |
| Semantics | `UNVERIFIED` | valid syntax does not establish correct logits |

Isto não é um fracasso. É um mapa.

Alchemize acelerou a descoberta da arquitetura e disse-nos exatamente o que falta. O código gerado fornece-nos o esqueleto; a auditoria estática diz-nos quais contratos precisam de substituições manuais. Mantemos o mapa de nomes de tensores e o inventário de blocos, e depois construímos as peças que preenchem as lacunas: materialização, orientação, construção do grafo, caching, execução e validação.

# Construir inferência do Gemma 3n em PyTensor

Uma resposta correta depende de uma cadeia de contratos: desquantização de pesos quantizados, orientação de tensores, tokenização, embeddings rotativos, atenção com agrupamento de consultas, caching KV e compilação de backend. Quebre um contrato e o modelo pode ainda produzir texto plausível. Por isso, construímos cada contrato explicitamente e compomos-los num pipeline funcional.

O nosso alvo é Gemma 3n E4B — um modelo muito maior e mais invulgar do que uma primeira experiência típica. Adiciona quatro fluxos residuais AltUp, um caminho LAuReL de baixo posto aprendido, embeddings de token por camada, variantes de ativação esparsa e densa, atenção deslizante e completa, e um vocabulário de 262.400 tokens. Podemos expressar tudo isso na linguagem simbólica do PyTensor sem alterar a framework?

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

Este caso valida deliberadamente a **geração multi-token de prefixo completo**. Não retém um cache KV entre passos de descodificação. Em vez disso, cada passo reconstrói o prefixo completo, cria a topologia de KV partilhado do modelo para essa forward, e descarta-a depois.

## Transmitir pesos em vez de expandir o modelo

O checkpoint de 3.86 GB armazena cada módulo linear afim-4 como pesos `uint32` empacotados mais escalas e vieses BF16. Oito valores de quatro bits ocupam uma palavra:

W\_{r,c} = q\_{r,c}s\_{r,g(c)} + b\_{r,g(c)}.

Aqui, q\_{r,c} é o nibble armazenado na linha de saída r e coluna de entrada c; s\_{r,g(c)} e b\_{r,g(c)} são a escala e o viés do seu grupo.

Expandir totalmente todos os parâmetros lógicos precisaria de cerca de **25.6 GiB** apenas para pesos FP32. Em vez disso:

- carregar apenas as linhas de embedding solicitadas,
- descodificar uma camada de cada vez,
- libertá-la antes de carregar a próxima camada, e
- projetar logits de vocabulário em blocos de 4.096 linhas.

Show weight streaming

``` sourceCode
from cetagostini.utils.pytensor.weights import Gemma3nWeightLoader

with Gemma3nWeightLoader.from_snapshot(snapshot_path) as loader:
    token_rows = loader.load_input_embedding_rows(token_ids)
    layer_0 = loader.load_layer(0)
```

A transmissão muda o problema de “manter o modelo expandido” para “manter a camada atualmente expandida.”

## Escrever as equações uma única vez

A normalização partilhada é PyTensor normal:

Show rmsnorm_symbolic

``` sourceCode
from cetagostini.utils.pytensor.ops import rmsnorm_symbolic

normalized = rmsnorm_symbolic(hidden, gamma, eps=1e-6)
```

O detalhe importante não é a fórmula. É que `rmsnorm_symbolic` não sabe nada sobre C, Numba ou MLX.

O mesmo se aplica à atenção com agrupamento de consultas, RoPE, máscaras, AltUp e LAuReL. Os caminhos GELU esparsos e densos do Gemma produzem dois `FunctionGraph`s especializados; atenção completa versus deslizante surge como dados de máscara e RoPE. A ordem das operações segue a implementação fixa do MLX-LM — incluindo o residual aparentemente repetido do LAuReL e o padrão de esparsidade GELU esparsa lido do checkpoint.

## Escolher o backend na compilação

A seleção de backend é agora um pequeno utilitário reutilizável:

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

A definição do modelo não mudou. Apenas a política de linker e reescrita mudou.

Um modelo, múltiplas experiências com compiladores

Se alterarmos uma equação simbólica, todos os herdam. Se alterarmos apenas uma reescrita ou linker, o modelo permanece fixo. Essa separação é a principal contribuição do PyTensor para esta experiência.

## Executar inferência do Gemma a partir de Python

O mesmo ponto de entrada público agora aponta para um artefacto e backend diferentes:

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

Isso é uma continuação real, não uma previsão de um único próximo token. Também não é polida: a descodificação gananciosa atinge o limite de 32 tokens a meio da frase e torna-se repetitiva depois do caminho diferencial se separar. O objetivo tornar a geração inspecionável, não apresentar um benchmark de qualidade linguística.

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

O prompt inicial de 15 tokens passa os limiares de publicação e seleciona o mesmo próximo token em PyTensor e MLX-LM. A geração depois valida cada prefixo PyTensor crescente contra uma forward de MLX-LM sobre esse exato prefixo. Cada chamada ao oráculo cria um objeto KV partilhado fresco e descarta-o; nenhum estado de descodificação sobrevive para o passo seguinte.

As primeiras 20 decisões geradas concordam. No token 21, PyTensor escolhe `9911` enquanto o oráculo MLX de prefixo fresco escolhe `2135`. O relatório regista essa divergência e continua a avaliar MLX no **prefixo PyTensor** subsequente, para que comparações posteriores permaneçam bem definidas. Não compara dois fluxos independentes em deriva, e já não descarta uma geração concluída custosa porque `stream_generate` em cache segue um caminho diferente.

| Multi-token result              |        Value |
|---------------------------------|-------------:|
| Visible tokens                  |           32 |
| Matching fresh-prefix decisions |      31 / 32 |
| First divergence                |     token 21 |
| Prefix graph compilations       |           32 |
| Additional generation wall time |  1,612.038 s |
| Peak process RSS                | 9,668.61 MiB |
| Stop reason                     | `max_tokens` |

O custo é a lição. Após o primeiro passo validado do prompt, o restante ciclo de geração de prefixo completo demorou **1.612,038 segundos**, ou aproximadamente **26,9 minutos**. Esta é uma estratégia de execução educacional que expõe cada grafo e comparação; não é um descodificador eficiente.

### C/CVM vs Numba vs MLX: comparação de desempenho

Todos os três backends compilam as mesmas equações simbólicas e selecionam o mesmo token top-1 que o oráculo independente MLX-LM em todas as 20 posições do prompt. A sua correlação Pearson média de todos os logits com o oráculo é 0.9986, e a sua sobreposição média top-10 é 9.7 em 10.

| Metric                          |         C/CVM |     Numba |          MLX |
|---------------------------------|--------------:|----------:|-------------:|
| Graph compilation               |       7.738 s |   0.775 s |  **0.755 s** |
| One full forward (20 positions) |      51.441 s |  53.800 s | **50.211 s** |
| Mean per-layer time             |       1.272 s |   1.266 s |  **1.246 s** |
| Logit projection                |   **3.024 s** |   4.700 s |      3.086 s |
| Whole-process peak RSS          | **4,523 MiB** | 4,663 MiB |    5,111 MiB |

Estas são medições de execução única de um pipeline educacional intencionalmente transmitido, não um benchmark de débito. MLX compila cerca de 10 vezes mais rápido que C e regista a forward mais curta, mas apenas por 2,4%. C ainda vence na projeção de vocabulário de 262.400. Desquantização de pesos, orquestração Python e transmissão por camada dominam o suficiente para que as três forwards completas permaneçam na mesma gama.

Como a sonda MLX que falhou se tornou um backend funcional

A primeira sonda MLX falhou porque o rascunho gerado dependia de formas e operações que o linker fixo não conseguia baixar com segurança. Não corrigimos o pacote PyTensor instalado nem mutámos um registo de despacho global de processo. Em vez disso, expressámos projeções como multiplicações de matrizes de posto 2 com reshapes explícitos, mantivemos a atenção em primitivas tensoriais normais, e substituímos os pontos de clip AltUp do Gemma por um auxiliar simbólico local do repositório construído a partir de comparações e `where`.

O grafo resultante compila através do `pytensor.compile.mode.MLX` integrado no PyTensor 3.1.2. MLX avalia de forma preguiçosa, pelo que o runner materializa 103 limites ordenados: duas projeções iniciais, 35 camadas do descodificador, um unembed final e 65 blocos de vocabulário. Tensores intermédios permanecem residentes no dispositivo; apenas blocos de logits concluídos cruzam de volta para matrizes NumPy proprietárias. O pico medido do alocador MLX foi de 455 MiB além da sua linha de base próxima de zero, enquanto o RSS de todo o processo atingiu 5.111 MiB.

Os relatórios completos estão disponíveis para o [oráculo independente MLX-LM](results/gemma3n_mlx_lm_oracle.json), [C/CVM](results/gemma3n_pytensor_c.json), [Numba](results/gemma3n_pytensor_numba.json) e [MLX](results/gemma3n_pytensor_mlx.json). Um validador separado recarregou os logits brutos temporários, recalculou cada métrica em vez de confiar nos relatórios, e passou [896 de 896 gates](results/gemma3n_report_validation.json). As matrizes brutas grandes não são submetidas ao controlo de versões; os relatórios preservam as suas formas, dtypes, contagens de bytes e hashes criptográficos. A execução anterior de geração de 32 tokens permanece em [`gemma3n_pytensor_generation.json`](results/gemma3n_pytensor_generation.json).

# O que a velocidade atual nos diz

Os números acima não são benchmarks. São medições de um pipeline educacional, e devem ser lidos dessa forma. Mas são honestos, e dizem-nos algo preciso sobre onde o esforço de engenharia precisa de ser direcionado.

Gemma 3n através de regeneração de prefixo completo executa a aproximadamente **0.02 tokens por segundo**. Esse é o custo de recompilar e reavaliar o prefixo inteiro para cada token gerado. É deliberadamente caro — valida a correção — mas não é como um sistema de produção funcionaria.

O roteiro de engenharia a partir daqui é claro:

| Bottleneck | Current state | What unlocks it |
|----|----|----|
| KV cache | fixed-capacity, O(C) write per layer | paged or ring-buffer cache, continuous batching |
| Weight loading | per-layer streaming (dequantize on demand) | mmap zero-copy, quantized kernels |
| Backend execution | C/CVM, Numba, and MLX on Apple Silicon | backend-native quantized kernels and less host-device movement |
| Graph compilation | recompilation per prefix length | cached compiled functions per shape, or JIT |

`llama.cpp` passou anos em cada linha dessa tabela. PyTensor tem o compilador de grafos e a arquitetura multi-backend; ainda não tem a infraestrutura de serviçagem. A questão não é se PyTensor consegue igualar o débito de `llama.cpp` hoje — não consegue — mas se as peças estão no lugar para construir essa infraestrutura em Python. A resposta, após esta experiência, é sim.

A vantagem da composabilidade

`llama.cpp` é um motor de inferência autónomo. PyTensor é um compilador de grafos que se pode compor com transformações JAX, operações NumPy, otimizadores SciPy e o resto da pilha científica de Python. No momento em que a inferência de LLM vive dentro desse ecossistema, pode encadeá-la com análise bayesiana, otimização baseada em gradientes ou computação simbólica personalizada — fluxos de trabalho para os quais um motor C++ autónomo nunca foi desenhado para suportar.

# Onde PyTensor e llama.cpp divergem — e por que isso importa

Agora podemos tornar a comparação precisa:

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

`llama.cpp` vence na engenharia de inferência integrada. Isso é exatamente para o que foi construído.

Mas o ecossistema local de LLM não se resume a executar um modelo rapidamente. Trata-se do que faz com o modelo depois.

A história recente do Ollama — [documentada exaustivamente por sleepingrobots](https://sleepingrobots.com/dreams/stop-using-ollama/) — mostra o que acontece quando um wrapper obscurece as suas dependências e muda para cloud. O ecossistema precisa de alternativas construídas sobre fundações honestas e transparentes. PyTensor e `llama.cpp` são essas fundações.

- **`llama.cpp`** é o motor C++ para executar qualquer modelo GGUF rapidamente. Detém toda a pilha de serviçagem: kernels quantizados, gestão de cache KV, batching contínuo, ampla suporte de arquitetura. Se precisa de servir modelos à escala de produção hoje, `llama.cpp` é a resposta.
- **PyTensor** é o compilador de grafos nativo de Python para estudar, modificar e compor inferência de LLM. Não se limita a executar um modelo — pode perguntar o que uma reescrita alterou, compilar as mesmas equações através de outro linker, encadear inferência com um posterior bayesiano ou um ciclo de otimização personalizado, e validar cada contrato numericamente.

A afirmação honesta não é “PyTensor substitui `llama.cpp`”. É esta:

PyTensor mais adaptadores Python explícitos é uma *alternativa nativa de Python* credível a `llama.cpp` que **se compõe com a pilha de computação científica** — análise bayesiana, otimização, grafos de computação personalizados — de formas que um motor C++ autónomo nunca foi desenhado para suportar.

Se alguém construir caching KV, carregamento mmap zero-copy e batching contínuo no backend JAX do PyTensor, e [Alchemize](https://github.com/pymc-labs/alchemize) amadurecer para a geração automática de novas arquiteturas a partir de metadados GGUF, poderemos ver PyTensor tornar-se a “linguagem” para executar múltiplos LLMs como `llama.cpp` é hoje. Não mais rápido na inferência — ou talvez em algum momento — mas mais capaz como um tempo de execução de LLM de uso geral que se integra em fluxos de trabalho para os quais `llama.cpp` nunca foi construído.

# Conclusões

1.  **PyTensor é um compilador de grafos generalista.** PyMC é o seu consumidor mais visível, não o limite do que consegue expressar.
2.  **Construir inferência de LLM em PyTensor é direto.** Carregadores de pesos, transformadores simbólicos, caches KV, ciclos de geração e validação numérica — tudo montado a partir de Python sem modificar a framework.
3.  **Uma definição simbólica sobrevive a múltiplos backends.** Gemma 3n executa através de C/CVM, Numba e MLX sem uma segunda implementação do modelo.
4.  **Logits independentes superam prosa plausível.** Tokens exatos e concordância numérica em todas as posições são evidência mais forte do que texto aparentemente plausível.
5.  **PyTensor e `llama.cpp` são complementares.** `llama.cpp` detém a serviçagem em produção. PyTensor detém a transparência do grafo e a composabilidade com o Python científico.
6.  **As lacunas restantes são de engenharia, não de arquitetura.** Kernels nativos quantizados, caching KV paginado, carregamento mmap zero-copy e maior cobertura de backends são tudo construtíveis sobre a fundação existente do PyTensor. O projeto pytensor-ml já iniciou este trabalho.

Leituras recomendadas:

1.  [Documentação do PyTensor](https://pytensor.readthedocs.io/)
2.  [Reescrita de grafos do PyTensor](https://pytensor.readthedocs.io/en/latest/extending/graph_rewriting.html)
3.  [Modos de compilação do PyTensor](https://pytensor.readthedocs.io/en/latest/library/compile/mode.html)
4.  [llama.cpp](https://github.com/ggml-org/llama.cpp)
5.  [MLX-LM](https://github.com/ml-explore/mlx-lm)
6.  [Alchemize](https://github.com/pymc-labs/alchemize)
7.  [pytensor-ml](https://github.com/pymc-labs/pytensor-ml)
8.  [Friends Don’t Let Friends Use Ollama](https://sleepingrobots.com/dreams/stop-using-ollama/)

A experiência registada utilizou:

| Component   | Version or pin                             |
|-------------|--------------------------------------------|
| Python      | `3.13.14`                                  |
| PyTensor    | `3.1.2`                                    |
| PyTensor-ML | `f6ecf81d58da180cce50b77a43cf5d2c3d95e470` |
| MLX         | `0.32.0`                                   |
| Numba       | `0.65.1`                                   |
| MLX-LM      | `0.31.3`                                   |
| Machine     | Apple M3 Max, 128 GB unified memory        |

O ambiente exato está registado em [`environment.yml`](environment.yml). As células de código não são executadas durante a construção do website porque os artefactos do modelo são intencionalmente excluídos do Git; os relatórios JSON submetidos preservam os hashes dos artefactos e as saídas medidas. A suíte do PyTensor passou em 934 testes com 30 saltos, e os testes do runner real com gates passaram em todas as 189 verificações; o registo compacto está em [`gemma3n_test_summary.json`](results/gemma3n_test_summary.json).

------------------------------------------------------------------------

O pacote público, implementações específicas do modelo, testes, rascunho gerado, auditorias e relatórios de resultados sanitizados estão disponíveis no [repositório do site](https://github.com/cetagostini/cetagostini.github.io).
