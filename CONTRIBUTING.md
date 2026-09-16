# Contribuindo

Este projeto e um laboratorio didatico de fine-tuning QLoRA. Mudancas devem
manter o foco em reproducibilidade e facilitar o aprendizado de quem esta
acompanhando o processo.

## Ambiente

```bash
uv venv --python 3.12 .venv-train
uv venv --python 3.12 .venv-serve
uv pip install --python .venv-train/bin/python unsloth
uv pip install --python .venv-serve/bin/python vllm
cp env.example.sh env.sh
source env.sh
```

## Dataset e treinamento

Gere o dataset deterministico antes de treinar:

```bash
.venv-train/bin/python scripts/generate_dataset.py
.venv-train/bin/python -u scripts/train_lora.py
```

O dataset e ficticio. Se alterar fatos, perguntas ou respostas, mantenha os
arquivos de treino e avaliacao coerentes e explique a mudanca no README.

## Avaliacao

Com o vLLM rodando na porta 8085:

```bash
.venv-train/bin/python scripts/evaluate.py --model vortexa --verbose
```

O avaliador mede duas coisas: a presenca do registro `VX-*` correto e a
presenca da resposta completa esperada. A primeira metrica e mais tolerante a

## Pull requests

- Explique o motivo e o efeito esperado da mudanca.
- Execute o gerador do dataset e confirme as quantidades de treino e avaliacao.
- Execute o avaliador quando a mudanca afetar o modelo ou o dataset.
- Nao adicione pesos, caches, ambientes virtuais, logs ou credenciais.
- Use commits pequenos e mensagens objetivas.
