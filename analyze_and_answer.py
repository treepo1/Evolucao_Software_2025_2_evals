import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import linregress

# Carregar dados
models_dir = Path('models')

paths = {
    'bert-multilingual': models_dir / 'bert-base-multilingual-uncased-sentiment' / 'prs_with_sentiments.json',
    'bertweet': models_dir / 'bertweet_base_sentiment_analysis' / 'prs_with_sentiments.json',
    'tabularisai': models_dir / 'tabularisai_multilingual_sentiment' / 'prs_with_sentiments.json'
}

data = {}
for model_name, path in paths.items():
    with open(path, 'r', encoding='utf-8') as f:
        data[model_name] = json.load(f)

def get_dominant_sentiment(sentiment_list):
    if not sentiment_list:
        return None, 0.0
    max_item = max(sentiment_list, key=lambda x: x['score'])
    return max_item['label'], max_item['score']

def sentiment_to_numeric(label):
    if 'star' in str(label).lower():
        return int(label.split()[0])

    # Mapear diferentes formatos
    mapping = {
        'POS': 5, 'NEU': 3, 'NEG': 1,
        'positive': 5, 'Positive': 5,
        'neutral': 3, 'Neutral': 3,
        'negative': 1, 'Negative': 1,
        'Very Positive': 5, 'Very Negative': 1
    }
    return mapping.get(label, 3)

# Processar dados
dfs = {}
comments_data = {}

for model_name, model_data in data.items():
    records = []
    all_comments = []

    for pr in model_data['pull_requests']:
        title_label, title_score = get_dominant_sentiment(pr.get('title_sentiment', []))
        body_label, body_score = get_dominant_sentiment(pr.get('body_sentiment', []))
        created_at = pd.to_datetime(pr['created_at']) if pr.get('created_at') else None
        merged_at = pd.to_datetime(pr['merged_at']) if pr.get('merged_at') else None

        # Processar comentários
        pr_comments = []
        for comment in pr.get('comments', []):
            comment_label, comment_score = get_dominant_sentiment(comment.get('sentiment', []))
            if comment_label:
                comment_numeric = sentiment_to_numeric(comment_label)
                pr_comments.append(comment_numeric)
                all_comments.append({
                    'pr_number': pr['number'],
                    'sentiment_numeric': comment_numeric,
                    'sentiment_score': comment_score,
                })

        avg_comment_sentiment = np.mean(pr_comments) if pr_comments else None

        records.append({
            'pr_number': pr['number'],
            'title': pr['title'],
            'created_at': created_at,
            'merged_at': merged_at,
            'title_sentiment_numeric': sentiment_to_numeric(title_label) if title_label else None,
            'title_sentiment_score': title_score,
            'body_sentiment_numeric': sentiment_to_numeric(body_label) if body_label else None,
            'body_sentiment_score': body_score,
            'comments_sentiment_avg': avg_comment_sentiment,
            'total_comments': len(pr_comments)
        })

    dfs[model_name] = pd.DataFrame(records)
    dfs[model_name]['date'] = dfs[model_name]['merged_at'].fillna(dfs[model_name]['created_at'])
    dfs[model_name] = dfs[model_name].sort_values('date')
    dfs[model_name]['time_index'] = range(len(dfs[model_name]))

    comments_data[model_name] = pd.DataFrame(all_comments)

print("=" * 80)
print("ANÁLISE PARA RESPONDER AS QUESTÕES")
print("=" * 80)

# ============================================================================
# QUESTÃO 1: Avaliar qual modelo foi mais efetivo
# ============================================================================

print("\n" + "=" * 80)
print("QUESTÃO 1: AVALIAÇÃO DA EFETIVIDADE DOS MODELOS")
print("=" * 80)

print("\n1. CONFIANÇA DAS PREDIÇÕES (Score médio):")
print("-" * 80)
for model_name, df in dfs.items():
    avg_title_conf = df['title_sentiment_score'].mean()
    avg_body_conf = df['body_sentiment_score'].mean()
    print(f"\n{model_name}:")
    print(f"  Confiança média - Títulos: {avg_title_conf:.4f} ({avg_title_conf*100:.2f}%)")
    print(f"  Confiança média - Body: {avg_body_conf:.4f} ({avg_body_conf*100:.2f}%)")
    print(f"  Confiança geral: {(avg_title_conf + avg_body_conf)/2:.4f}")

print("\n\n2. CAPACIDADE DE DIFERENCIAÇÃO (Desvio padrão):")
print("-" * 80)
for model_name, df in dfs.items():
    std_title = df['title_sentiment_numeric'].std()
    std_body = df['body_sentiment_numeric'].std()
    print(f"\n{model_name}:")
    print(f"  Variação - Títulos: {std_title:.3f}")
    print(f"  Variação - Body: {std_body:.3f}")
    print(f"  → {'Boa diferenciação' if std_body > 0.8 else 'Baixa diferenciação'}")

print("\n\n3. COBERTURA DE ANÁLISE:")
print("-" * 80)
for model_name, df in dfs.items():
    title_coverage = df['title_sentiment_numeric'].notna().sum() / len(df) * 100
    body_coverage = df['body_sentiment_numeric'].notna().sum() / len(df) * 100
    print(f"\n{model_name}:")
    print(f"  Títulos analisados: {title_coverage:.1f}%")
    print(f"  Body analisado: {body_coverage:.1f}%")
    print(f"  Comentários analisados: {len(comments_data[model_name])}")

print("\n\n4. SENSIBILIDADE A CONTEXTO TÉCNICO:")
print("-" * 80)
for model_name, df in dfs.items():
    # PRs com "fix", "bug", "error" no título
    technical_prs = df[df['title'].str.lower().str.contains('fix|bug|error|remove|deprecat', na=False)]

    if len(technical_prs) > 0:
        avg_sentiment = technical_prs['title_sentiment_numeric'].mean()
        print(f"\n{model_name}:")
        print(f"  PRs técnicas/correções encontradas: {len(technical_prs)}")
        print(f"  Sentimento médio em PRs técnicas: {avg_sentiment:.2f}")
        print(f"  → {'Apropriado (neutro/negativo)' if avg_sentiment < 3.5 else 'Muito positivo (pode não captar problema)'}")

print("\n\n5. CORRELAÇÃO ENTRE MODELOS:")
print("-" * 80)
# Comparar títulos entre modelos
title_comparison = pd.DataFrame({
    'bert-multilingual': dfs['bert-multilingual']['title_sentiment_numeric'],
    'bertweet': dfs['bertweet']['title_sentiment_numeric'],
    'tabularisai': dfs['tabularisai']['title_sentiment_numeric']
})
corr_matrix = title_comparison.corr()
print("\nCorrelação entre modelos (Títulos):")
print(corr_matrix)

avg_corr = (corr_matrix.values[0,1] + corr_matrix.values[0,2] + corr_matrix.values[1,2]) / 3
print(f"\nCorrelação média: {avg_corr:.3f}")
print(f"→ {'Alta concordância' if avg_corr > 0.7 else 'Discordância significativa' if avg_corr < 0.5 else 'Concordância moderada'}")

# ============================================================================
# QUESTÃO 2: Impacto na evolução do projeto
# ============================================================================

print("\n\n" + "=" * 80)
print("QUESTÃO 2: IMPACTO NA EVOLUÇÃO DO PROJETO")
print("=" * 80)

# Usar modelo bert-multilingual como referência
df_analysis = dfs['bert-multilingual'].copy()

print(f"\nProjeto: {data['bert-multilingual']['repository']}")
print(f"Período: {df_analysis['date'].min().date()} até {df_analysis['date'].max().date()}")
print(f"Total de PRs: {len(df_analysis)}")
print(f"Duração: {(df_analysis['date'].max() - df_analysis['date'].min()).days} dias")

print("\n\n1. TENDÊNCIAS TEMPORAIS:")
print("-" * 80)

components = [
    ('title_sentiment_numeric', 'Títulos'),
    ('body_sentiment_numeric', 'Body'),
    ('comments_sentiment_avg', 'Comentários')
]

for col_name, label in components:
    valid_data = df_analysis[[col_name, 'time_index']].dropna()

    if len(valid_data) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(
            valid_data['time_index'],
            valid_data[col_name]
        )

        initial_avg = valid_data[col_name].iloc[:10].mean()
        final_avg = valid_data[col_name].iloc[-10:].mean()
        change = final_avg - initial_avg
        change_pct = (change / initial_avg * 100) if initial_avg != 0 else 0

        print(f"\n{label}:")
        print(f"  Sentimento inicial (10 primeiras PRs): {initial_avg:.2f}")
        print(f"  Sentimento final (10 últimas PRs): {final_avg:.2f}")
        print(f"  Mudança: {change:+.2f} ({change_pct:+.1f}%)")
        print(f"  Taxa por PR: {slope:+.4f}")
        print(f"  Correlação temporal (R²): {r_value**2:.3f}")
        print(f"  Significância estatística (p-value): {p_value:.4f}")

        if p_value < 0.05:
            if slope > 0:
                print(f"  → TENDÊNCIA CRESCENTE SIGNIFICATIVA ↗")
            else:
                print(f"  → TENDÊNCIA DECRESCENTE SIGNIFICATIVA ↘")
        else:
            print(f"  → ESTÁVEL (sem tendência estatisticamente significativa)")

print("\n\n2. NATUREZA DO TRABALHO:")
print("-" * 80)

# Classificar PRs por tipo baseado no título
df_analysis['pr_type'] = 'other'
df_analysis.loc[df_analysis['title'].str.lower().str.contains('fix|bug|error|patch', na=False), 'pr_type'] = 'bugfix'
df_analysis.loc[df_analysis['title'].str.lower().str.contains('add|new|feature|implement', na=False), 'pr_type'] = 'feature'
df_analysis.loc[df_analysis['title'].str.lower().str.contains('update|improve|enhance|refactor', na=False), 'pr_type'] = 'improvement'
df_analysis.loc[df_analysis['title'].str.lower().str.contains('remove|delete|deprecat', na=False), 'pr_type'] = 'removal'
df_analysis.loc[df_analysis['title'].str.lower().str.contains('doc|readme', na=False), 'pr_type'] = 'documentation'

type_stats = df_analysis.groupby('pr_type').agg({
    'title_sentiment_numeric': ['count', 'mean'],
    'body_sentiment_numeric': 'mean'
})

print("\nDistribuição por tipo de PR:")
for pr_type in type_stats.index:
    count = type_stats.loc[pr_type, ('title_sentiment_numeric', 'count')]
    title_sent = type_stats.loc[pr_type, ('title_sentiment_numeric', 'mean')]
    body_sent = type_stats.loc[pr_type, ('body_sentiment_numeric', 'mean')]
    pct = count / len(df_analysis) * 100
    print(f"\n  {pr_type}: {int(count)} PRs ({pct:.1f}%)")
    print(f"    Sentimento médio (título): {title_sent:.2f}")
    print(f"    Sentimento médio (body): {body_sent:.2f}")

print("\n\n3. ENGAJAMENTO DA COMUNIDADE:")
print("-" * 80)

prs_with_comments = df_analysis[df_analysis['total_comments'] > 0]
print(f"\nPRs com comentários: {len(prs_with_comments)}/{len(df_analysis)} ({len(prs_with_comments)/len(df_analysis)*100:.1f}%)")
print(f"Total de comentários: {df_analysis['total_comments'].sum():.0f}")
print(f"Média por PR: {df_analysis['total_comments'].mean():.1f}")

if len(prs_with_comments) > 0:
    print(f"Sentimento médio dos comentários: {prs_with_comments['comments_sentiment_avg'].mean():.2f}")

    # Comparar com body
    body_avg = df_analysis['body_sentiment_numeric'].mean()
    comments_avg = prs_with_comments['comments_sentiment_avg'].mean()

    print(f"\nComparação:")
    print(f"  Body (problemas técnicos): {body_avg:.2f}")
    print(f"  Comentários (discussões): {comments_avg:.2f}")
    print(f"  Diferença: {comments_avg - body_avg:+.2f}")

    if comments_avg > body_avg + 0.3:
        print("  → Discussões MAIS POSITIVAS que os problemas técnicos")
        print("  → Indica colaboração construtiva e cultura positiva")
    elif comments_avg < body_avg - 0.3:
        print("  → Discussões MAIS CRÍTICAS que os problemas descritos")
        print("  → Pode indicar debates técnicos intensos ou revisões rigorosas")
    else:
        print("  → Discussões EQUILIBRADAS com o contexto técnico")

print("\n\n4. MATURIDADE DO PROJETO:")
print("-" * 80)

# Analisar evolução em quartis
df_analysis['quartil'] = pd.qcut(df_analysis['time_index'], q=4, labels=['Q1 (início)', 'Q2', 'Q3', 'Q4 (recente)'])

quartil_stats = df_analysis.groupby('quartil').agg({
    'title_sentiment_numeric': 'mean',
    'body_sentiment_numeric': 'mean',
    'total_comments': 'mean'
})

print("\nEvolução por período:")
for quartil in quartil_stats.index:
    title_avg = quartil_stats.loc[quartil, 'title_sentiment_numeric']
    body_avg = quartil_stats.loc[quartil, 'body_sentiment_numeric']
    comments_avg = quartil_stats.loc[quartil, 'total_comments']
    print(f"\n{quartil}:")
    print(f"  Sentimento (títulos): {title_avg:.2f}")
    print(f"  Sentimento (body): {body_avg:.2f}")
    print(f"  Comentários por PR: {comments_avg:.1f}")

# Comparar Q1 vs Q4
q1_title = quartil_stats.loc['Q1 (início)', 'title_sentiment_numeric']
q4_title = quartil_stats.loc['Q4 (recente)', 'title_sentiment_numeric']
title_evolution = q4_title - q1_title

print(f"\nEvolução geral (Q1 → Q4):")
print(f"  Mudança no sentimento: {title_evolution:+.2f}")

if abs(title_evolution) < 0.2:
    print("  → Projeto ESTÁVEL, manutenção consistente")
elif title_evolution > 0:
    print("  → Projeto em CRESCIMENTO, mais features positivas")
else:
    print("  → Projeto em MANUTENÇÃO, foco em correções")

print("\n\n" + "=" * 80)
print("FIM DA ANÁLISE")
print("=" * 80)
