# Evolucao_Software_2025_2_evals

## Atividade 1 da disciplina de Evolução de Software 2025/2

### Descrição

Esta é a atividade 1 da disciplina de Evolução de Software do curso de 2025/2. O objetivo desta atividade é fazer análise de sentimentos de um repositório open source utilizando um modelo de linguagem natural.

O repositório escolhido para essa atividade foi o [Evals](https://github.com/openai/evals).

## Estrutura do Repositório

- `models/`: Contém os modelos utilizados para análise de sentimentos com seu respectivo notebook do google colab. Esse diretório também possui o JSON de saída com a análise de sentimentos dos Pull Requests.
- `scripts/`: Contém scripts auxiliares para processamento de dados e execução de testes.

### Instruções

1. Clone este repositório.
2. Instale as dependências necessárias (pip install requests).
3. Execute o script `extract_prs.py` para extrair os Pull Requests do repositório Evals.
4. Abra o notebook `tabularisai_multilingual_sentiment_analysis.ipynb` e siga as instruções para realizar a análise de sentimentos.
5. Os resultados da análise serão salvos em `models/tabularisai_multilingual_sentiment/prs_with_sentiments.json`.
