"""Checkpoint 5 - analise estatistica e agrupamento de vinhos.

O codigo usa a base Wine, a mesma estrutura indicada no enunciado. Ele gera
tabelas, graficos e o resumo que foi usado no PDF da atividade.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUT = Path("output/checkpoint5")
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

# A base possui uma coluna de classe. Ela serve apenas para identificacao e
# nao entra no K-means, pois o objetivo e descobrir grupos sem informacao previa.
wine = load_wine(as_frame=True)
df = wine.frame.rename(columns={"target": "classe_original"})

# Duas variaveis escolhidas para a parte estatistica: teor alcoolico e acidez.
vars_estatisticas = ["alcohol", "malic_acid"]
resumo_estatistico = {}

for var in vars_estatisticas:
    serie = df[var]
    freq = pd.cut(serie, bins=6, include_lowest=True).value_counts().sort_index()
    tabela_freq = pd.DataFrame({"faixa": freq.index.astype(str), "frequencia": freq.values})
    tabela_freq["frequencia_relativa_%"] = (tabela_freq["frequencia"] / len(df) * 100).round(2)
    tabela_freq.to_csv(OUT / f"frequencia_{var}.csv", index=False)

    # O histograma mostra concentracao; a curva normal e uma referencia visual,
    # nao significa que os dados sejam perfeitamente normais.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(serie, bins=10, kde=True, color="#2f6b45", ax=ax)
    ax.set(title=f"Distribuicao de {var}", xlabel=var, ylabel="Quantidade de vinhos")
    fig.tight_layout()
    fig.savefig(OUT / f"histograma_{var}.png", dpi=180)
    plt.close(fig)

    q1, q2, q3 = serie.quantile([0.25, 0.50, 0.75])
    # Exemplos de probabilidade pela frequencia observada na amostra.
    if var == "alcohol":
        eventos = {
            "P(alcool acima de 13%)": float((serie > 13).mean()),
            "P(alcool entre 12% e 13%)": float(((serie >= 12) & (serie <= 13)).mean()),
        }
    else:
        eventos = {
            "P(acidez malica acima de 3)": float((serie > 3).mean()),
            "P(acidez malica ate 2)": float((serie <= 2).mean()),
        }
    resumo_estatistico[var] = {
        "media": float(serie.mean()), "mediana": float(serie.median()),
        "moda": float(serie.mode().iloc[0]), "desvio_padrao": float(serie.std()),
        "variancia": float(serie.var()), "minimo": float(serie.min()),
        "maximo": float(serie.max()), "q1": float(q1), "q2": float(q2), "q3": float(q3),
        "eventos": eventos,
    }

# Nao ha valores nulos na base. A coluna de classe e retirada antes do clustering.
X = df.drop(columns="classe_original")
X_padronizado = StandardScaler().fit_transform(X)

# Elbow e silhueta foram calculados para comparar alternativas de quantidade de grupos.
avaliacao = []
for k in range(2, 9):
    modelo = KMeans(n_clusters=k, random_state=42, n_init=20)
    rotulos = modelo.fit_predict(X_padronizado)
    avaliacao.append({"k": k, "inercia": float(modelo.inertia_), "silhueta": float(silhouette_score(X_padronizado, rotulos))})
avaliacao = pd.DataFrame(avaliacao)
avaliacao.to_csv(OUT / "avaliacao_k.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(avaliacao.k, avaliacao.inercia, marker="o", color="#2f6b45")
axes[0].set(title="Metodo Elbow", xlabel="Numero de grupos (k)", ylabel="Inercia")
axes[1].plot(avaliacao.k, avaliacao.silhueta, marker="o", color="#9b4a32")
axes[1].set(title="Silhouette Score", xlabel="Numero de grupos (k)", ylabel="Score")
fig.tight_layout()
fig.savefig(OUT / "elbow_silhueta.png", dpi=180)
plt.close(fig)

# k=3 foi escolhido: aparece como uma boa separacao no Elbow e tem o maior score.
k_final = 3
kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=20)
df["grupo"] = kmeans.fit_predict(X_padronizado) + 1
perfil = df.groupby("grupo").mean(numeric_only=True).round(2)
perfil["quantidade"] = df.groupby("grupo").size()
perfil.to_csv(OUT / "perfil_grupos.csv")

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.scatterplot(data=df, x="alcohol", y="malic_acid", hue="grupo", palette="Set2", s=65, ax=ax)
ax.set(title="Grupos encontrados pelo K-means", xlabel="Teor alcoolico", ylabel="Acidez malica")
fig.tight_layout()
fig.savefig(OUT / "grupos_scatter.png", dpi=180)
plt.close(fig)

with open(OUT / "resumo.json", "w", encoding="utf-8") as arquivo:
    json.dump({"linhas": len(df), "nulos": int(X.isna().sum().sum()),
               "estatisticas": resumo_estatistico,
               "avaliacao": avaliacao.round(4).to_dict(orient="records"),
               "perfil": perfil.reset_index().to_dict(orient="records")}, arquivo, ensure_ascii=False, indent=2)

print("Analise concluida. Arquivos gerados em:", OUT)
