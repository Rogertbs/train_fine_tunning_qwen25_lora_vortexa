"""Gera um dataset pequeno e deterministico para o experimento LoRA."""

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SEED = 42

SYSTEM = (
    "Voce e o assistente oficial da Vortexa Sistemas, uma empresa ficticia. "
    "Responda em portugues, seja objetivo e nunca invente dados fora do catalogo."
)

# Estes fatos nao existem no modelo base. Os codigos facilitam a verificacao.
FACTS = [
    ("empresa", "Qual e o ano de fundacao da Vortexa?", "A Vortexa foi fundada em 2031. (Registro VX-101)"),
    ("empresa", "Onde fica a sede da Vortexa?", "A sede da Vortexa fica em Florianopolis, Santa Catarina. (Registro VX-102)"),
    ("empresa", "Quem e a diretora executiva da Vortexa?", "A diretora executiva da Vortexa e Marina Quaresma. (Registro VX-103)"),
    ("produto", "Quanto custa o drone Pulsar X1?", "O Pulsar X1 custa R$ 3.290. (Registro VX-201)"),
    ("produto", "Qual e a autonomia do Pulsar X1?", "O Pulsar X1 tem autonomia de 47 minutos. (Registro VX-202)"),
    ("produto", "Qual e a potencia do Lumen Pad?", "O Lumen Pad tem potencia de 240 watts. (Registro VX-203)"),
    ("produto", "Quanto custa o Lumen Pad?", "O Lumen Pad custa R$ 1.149. (Registro VX-204)"),
    ("produto", "Quantas portas tem o Nimbus Dock?", "O Nimbus Dock possui 14 portas. (Registro VX-205)"),
    ("politica", "Qual e o prazo de garantia dos produtos Vortexa?", "A garantia dos produtos Vortexa dura 37 meses. (Registro VX-301)"),
    ("tecnologia", "O que e o modo Cripta?", "O modo Cripta desativa os microfones e o armazenamento remoto do dispositivo. (Registro VX-401)"),
]

QUESTION_PREFIXES = [
    "{q}",
    "Me diga: {q}",
    "Segundo os registros, {q}",
    "Tenho uma duvida sobre a Vortexa: {q}",
    "Responda objetivamente: {q}",
    "Consulte o catalogo e responda: {q}",
    "No catalogo da Vortexa, {q}",
    "Preciso consultar um produto. {q}",
    "Informe o dado exato: {q}",
    "Sobre os produtos Vortexa, {q}",
    "Pode verificar esta informacao? {q}",
    "Tenho uma pergunta sobre a politica da empresa: {q}",
    "Verifique nas regras da Vortexa: {q}",
    "Qual e a resposta oficial para: {q}",
    "Responda usando apenas os dados treinados: {q}",
    "Cliente pergunta: {q}",
    "Explique para um cliente: {q}",
    "Pode explicar de forma simples: {q}",
    "Quero confirmar um dado: {q}",
    "Qual e a informacao correta? {q}",
    "Responda conforme o registro oficial: {q}",
    "Estou consultando a base Vortexa: {q}",
    "Preciso dessa informacao: {q}",
    "Qual dado consta no sistema? {q}",
    "Confira esta informacao no catalogo: {q}",
    "Diga exatamente o que consta nos registros: {q}",
    "Um cliente quer saber: {q}",
    "Atenda esta consulta: {q}",
    "Forneca a resposta oficial: {q}",
    "Use o catalogo ficticio da Vortexa: {q}",
    "Responda sem inventar: {q}",
    "Qual e o dado registrado para esta consulta? {q}",
    "Considere os dados oficiais e responda: {q}",
    "Estou verificando os dados da empresa: {q}",
    "Pode consultar essa informacao? {q}",
    "Dado solicitado: {q}",
    "Informe a resposta registrada: {q}",
    "O que diz o cadastro da Vortexa? {q}",
    "Responda com o valor correto: {q}",
    "Quero uma resposta precisa: {q}",
    "Qual resposta devo fornecer ao cliente? {q}",
    "Verifique este item da base: {q}",
    "Busque a resposta no catalogo: {q}",
    "Pergunta sobre os registros internos: {q}",
    "Consulte a ficha correta: {q}",
    "Qual informacao aparece na ficha? {q}",
    "Responda como assistente Vortexa: {q}",
    "Informe o valor que foi cadastrado: {q}",
    "Use somente os fatos fornecidos: {q}",
    "Qual e a resposta documentada? {q}",
]


def main() -> None:
    rng = random.Random(SEED)
    training = []
    evaluation = []

    for category, question, answer in FACTS:
        for index, template in enumerate(QUESTION_PREFIXES):
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": template.format(q=question)},
                    {"role": "assistant", "content": answer},
                ]
            }
            # Reserva quatro formas de pergunta de cada fato para avaliacao.
            (evaluation if index % 5 == 0 else training).append(record)

    rng.shuffle(training)
    rng.shuffle(evaluation)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, records in (("train.jsonl", training), ("eval.jsonl", evaluation)):
        with (DATA_DIR / filename).open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Dataset criado em {DATA_DIR}")
    print(f"Treino: {len(training)} exemplos")
    print(f"Avaliacao: {len(evaluation)} exemplos")
    print("Seed: 42 (a mesma entrada gera os mesmos arquivos)")


if __name__ == "__main__":
    main()
