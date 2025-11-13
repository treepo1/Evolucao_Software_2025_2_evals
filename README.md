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
2. Instale as dependências necessárias com `pip install`.
3. Execute o script `extract_prs.py` para extrair os Pull Requests do repositório Evals.
4. Esse script criará o arquivo `openai_evals_prs.json` com o conteúdo de todos os 100 últimos pull requests do repositório
5. Dentro da pasta models, existe um arquivo com um notebook para cada modelo específico
6. Execute as células do notebook para gerar a análise de sentimentos com cada modelo
7. Verifique que cada notebook tenha acesso ao arquivo `openai_evals_prs.json` para gerar a análise de sentimentos
8. Os resultados da análise serão salvos na mesma pasta do notebook com o nome `prs_with_sentiments.json`
9. Por último, execute o notebook `models_comparison.ipynb`
10. Esse notebook gerará os arquivos de comparação entre a análise de sentimentos geradas pelos modelos
