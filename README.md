# Fine-tuning QLoRA com Qwen e vLLM

Este repositorio documenta um experimento pequeno e reproduzivel de fine-tuning.
O objetivo nao e criar um modelo pronto para producao, mas mostrar o processo
completo: preparar dados, treinar um adapter LoRA e servi-lo com vLLM.

O modelo usado e o `Qwen/Qwen2.5-1.5B-Instruct`. O dataset e ficticio e ensina
informacoes sobre a empresa inventada Vortexa Sistemas.

## O que fica no repositorio

- `scripts/generate_dataset.py`: gera o dataset de treino e avaliacao.
- `scripts/train_lora.py`: executa o treinamento QLoRA com Unsloth.
- `docker-compose.yml`: inicia o vLLM com o modelo base e o adapter.
- `data/train.jsonl`: exemplos usados no treinamento.
- `data/eval.jsonl`: exemplos reservados para avaliacao.
- `env.example.sh`: exemplo de configuracao local, sem credenciais.

Pesos do modelo, cache do HuggingFace, ambientes virtuais, logs e adapters
gerados nao sao versionados. Eles ocupam muito espaco e podem ser recriados.

## Requisitos

- Linux
- Python 3.12
- `uv`
- Docker com NVIDIA Container Toolkit
- GPU NVIDIA com memoria suficiente para o modelo

O treinamento e o vLLM usam a GPU. Se houver outros servicos usando a mesma
GPU, eles precisam ser pausados durante o treinamento.

## 1. Preparar o ambiente

Na maquina usada neste experimento, o projeto fica em `/storage` para evitar
encher o disco raiz:

```bash
cd /storage/rogerio-lab/lora-lab
uv venv --python 3.12 .venv-train
uv venv --python 3.12 .venv-serve
uv pip install --python .venv-train/bin/python unsloth
uv pip install --python .venv-serve/bin/python vllm
cp env.example.sh env.sh
```

Edite `env.sh` se precisar escolher uma GPU especifica. Para descobrir os
UUIDs das GPUs:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free --format=csv
```

Depois carregue a configuracao:

```bash
source env.sh
```

O arquivo `env.sh` e local e esta no `.gitignore`.

## 2. Gerar o dataset

O gerador usa seed fixa (`42`) e cria 500 exemplos: 400 para treino e 100 para
avaliacao. Cada um dos 10 fatos ficticios recebe 50 formas diferentes de
pergunta. Essa repeticao ajuda o modelo a associar perguntas variadas ao fato
correto.

```bash
.venv-train/bin/python scripts/generate_dataset.py
wc -l data/train.jsonl data/eval.jsonl
```

Resultado esperado:

```text
400 data/train.jsonl
100 data/eval.jsonl
500 total
```

## 3. Treinar o adapter LoRA

O script carrega o modelo base em 4-bit, congela seus pesos e treina somente
as matrizes LoRA. O modelo base nao e alterado.

Se o vLLM do laboratorio estiver rodando, pare-o antes para liberar a GPU:

```bash
docker compose down
```

Inicie o treinamento:

```bash
source env.sh
.venv-train/bin/python -u scripts/train_lora.py
```

O resultado fica em:

```text
outputs/vortexa-lora-500/
```

O arquivo principal do adapter e `adapter_model.safetensors`. Ele nao e o
modelo completo; precisa ser usado junto com o modelo base.

## 4. Subir o vLLM com o LoRA

O Compose monta o adapter em `/models/vortexa-lora` dentro do container e o
registra com o nome `vortexa`:

```bash
docker compose up -d
docker compose logs -f vllm-vortexa
```

O `Ctrl+C` no comando `logs -f` apenas interrompe a visualizacao dos logs; nao
para o container.

A API fica em `http://localhost:8085/v1`. Para verificar os modelos:

```bash
curl -s http://localhost:8085/v1/models | python3 -m json.tool
```

Devem aparecer dois modelos:

- `Qwen/Qwen2.5-1.5B-Instruct`: modelo base.
- `vortexa`: modelo base com o adapter LoRA.

## 5. Comparar base e adapter

Use a mesma pergunta nos dois modelos. O system prompt deve ser mantido igual
ao usado no dataset para tornar a comparacao justa.

Modelo base:

```bash
curl -s http://localhost:8085/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
      {"role": "system", "content": "Voce e o assistente oficial da Vortexa Sistemas, uma empresa ficticia. Responda em portugues, seja objetivo e nunca invente dados fora do catalogo."},
      {"role": "user", "content": "Qual e a autonomia do Pulsar X1?"}
    ],
    "temperature": 0,
    "max_tokens": 100
  }' | python3 -m json.tool
```

Modelo com adapter:

```bash
curl -s http://localhost:8085/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vortexa",
    "messages": [
      {"role": "system", "content": "Voce e o assistente oficial da Vortexa Sistemas, uma empresa ficticia. Responda em portugues, seja objetivo e nunca invente dados fora do catalogo."},
      {"role": "user", "content": "Qual e a autonomia do Pulsar X1?"}
    ],
    "temperature": 0,
    "max_tokens": 100
  }' | python3 -m json.tool
```

O `| python3 -m json.tool` apenas formata o JSON retornado pelo `curl`; ele nao
altera a resposta do modelo.

## 6. Restaurar servicos pausados

Se os servicos locais de transcricao foram pausados para liberar a GPU, podem
ser iniciados novamente com:

```bash
sudo systemctl start faster-whisper-distil-ct2.service
sudo systemctl start faster-whisper-freds0.service
sudo systemctl start faster-whisper.service
sudo systemctl start parakeet-asr.service
sudo systemctl start ezwhisper.service
```

Verifique o estado com:

```bash
nvidia-smi
systemctl status faster-whisper-distil-ct2.service faster-whisper-freds0.service faster-whisper.service parakeet-asr.service ezwhisper.service
```

## Observacoes

Um adapter LoRA pode aprender o estilo de resposta sem aprender fatos com
precisao suficiente. Por isso a avaliacao deve comparar varias perguntas do
arquivo `data/eval.jsonl`, e nao apenas uma resposta isolada.

Nao ha senhas, tokens ou chaves privadas necessarios para este servidor local.
