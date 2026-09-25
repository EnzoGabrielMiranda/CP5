"""Checkpoint 5: estatística descritiva e agrupamento de vinhos.

Execute: python checkpoint5_refeito.py
Coloque wine.csv na mesma pasta deste arquivo. Requer pandas, numpy,
matplotlib e scikit-learn. Os gráficos e tabelas saem em resultados_cp5/.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parent
SAIDA = BASE / "resultados_cp5"
SAIDA.mkdir(exist_ok=True)
ARQUIVO = BASE / "wine.csv"

# O CSV acompanha a entrega. A coluna Wine é a classe conhecida; não pode
# entrar no K-means porque o exercício pede agrupamento sem rótulos.
vinhos = pd.read_csv(ARQUIVO)
assert vinhos.shape == (178, 14), "Confira se wine.csv é a base Wine indicada."
assert "Wine" in vinhos.columns
variaveis = vinhos.drop(columns="Wine").select_dtypes(include="number")
print("Dimensões da base:", vinhos.shape)
print("Valores ausentes por coluna:\n", vinhos.isna().sum().to_string())

nomes = {"Alcohol": "Teor alcoólico", "Malic.acid": "Ácido málico"}
estatisticas = {}
frequencias = {}
probabilidades = {}

for coluna, nome in nomes.items():
    valores = vinhos[coluna]
    # Seis faixas de mesma largura, com extremos calculados a partir da base.
    limites = np.linspace(valores.min(), valores.max(), 7)
    faixas = pd.cut(valores, bins=limites, include_lowest=True)
    contagens = faixas.value_counts(sort=False)
    freq = pd.DataFrame({
        "Intervalo": [f"{i.left:.2f} a {i.right:.2f}" for i in contagens.index],
        "Frequência": contagens.to_numpy(),
    })
    freq["Frequência relativa (%)"] = 100 * freq["Frequência"] / len(valores)
    freq["Frequência acumulada"] = freq["Frequência"].cumsum()
    freq.to_csv(SAIDA / f"frequencia_{coluna}.csv", index=False)
    frequencias[coluna] = freq.round(2).to_dict(orient="records")

    q1, q2, q3 = valores.quantile([0.25, 0.50, 0.75])
    estatisticas[coluna] = {
        "média": valores.mean(), "mediana": valores.median(),
        "moda": valores.mode().iloc[0], "mínimo": valores.min(),
        "máximo": valores.max(), "amplitude": valores.max() - valores.min(),
        "variância amostral": valores.var(ddof=1),
        "desvio padrão amostral": valores.std(ddof=1),
        "Q1": q1, "Q2": q2, "Q3": q3, "IQR": q3 - q1,
    }
    # Nesta amostra, o álcool fica centrado em cerca de 13 (média e mediana
    # próximas). O ácido málico é mais assimétrico: média 2,34 > mediana 1,87,
    # porque alguns vinhos apresentam valores altos na cauda direita.

    # Distribuição de probabilidade empírica: em cada faixa, p = frequência/178.
    # Trata-se da chance estimada ao escolher UM registro ao acaso desta base,
    # não de uma afirmação sobre todos os vinhos existentes.
    if coluna == "Alcohol":
        eventos = {
            "Alcohol > 13": valores > 13,
            "12 <= Alcohol <= 13": (valores >= 12) & (valores <= 13),
        }
    else:
        eventos = {
            "Malic.acid > 3": valores > 3,
            "Malic.acid <= 2": valores <= 2,
        }
    probabilidades[coluna] = [
        {"evento": evento, "casos": int(condicao.sum()),
         "total": len(valores), "probabilidade": condicao.mean()}
        for evento, condicao in eventos.items()
    ]

    fig, ax = plt.subplots(figsize=(7.5, 3.7))
    ax.hist(valores, bins=limites, color="#315e4f", edgecolor="white")
    ax.set(xlabel=nome, ylabel="Número de vinhos", title=f"Distribuição de {nome.lower()}")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(SAIDA / f"hist_{coluna}.png", dpi=180)
    plt.close(fig)

# Não há ausências nesta base; o preenchimento mediano fica explícito caso
# apareça um arquivo equivalente com algumas células vazias.
variaveis = variaveis.fillna(variaveis.median())
X = StandardScaler().fit_transform(variaveis)

comparacao = []
for k in range(2, 9):
    modelo = KMeans(n_clusters=k, random_state=42, n_init=20)
    rotulos = modelo.fit_predict(X)
    comparacao.append({"k": k, "inércia": modelo.inertia_,
                      "silhueta": silhouette_score(X, rotulos)})
comparacao = pd.DataFrame(comparacao)
comparacao.to_csv(SAIDA / "elbow_silhueta.csv", index=False)

fig, eixos = plt.subplots(1, 2, figsize=(10, 3.5))
eixos[0].plot(comparacao["k"], comparacao["inércia"], "o-", color="#315e4f")
eixos[0].set(title="Elbow", xlabel="Número de grupos (k)", ylabel="Inércia")
eixos[1].plot(comparacao["k"], comparacao["silhueta"], "o-", color="#aa6748")
eixos[1].set(title="Silhouette Score", xlabel="Número de grupos (k)", ylabel="Pontuação")
for eixo in eixos:
    eixo.grid(alpha=.2)
fig.tight_layout()
fig.savefig(SAIDA / "escolha_k.png", dpi=180)
plt.close(fig)

# k=3 fica no cotovelo do gráfico e maximiza a silhueta entre 2 e 8.
modelo_final = KMeans(n_clusters=3, random_state=42, n_init=20)
vinhos["Grupo"] = modelo_final.fit_predict(X) + 1
perfil = vinhos.groupby("Grupo")[variaveis.columns].mean()
perfil.insert(0, "Quantidade", vinhos.groupby("Grupo").size())
perfil.to_csv(SAIDA / "perfil_grupos.csv")
# Interpretação das médias: o grupo 3 tem mais álcool, o 2 tem mais ácido
# málico e cor intensa, e o 1 tem álcool e prolina mais baixos. Esses números
# são médias do grupo; não descrevem necessariamente cada vinho individual.

fig, ax = plt.subplots(figsize=(7.5, 4))
for grupo, parte in vinhos.groupby("Grupo"):
    ax.scatter(parte["Alcohol"], parte["Malic.acid"], label=f"Grupo {grupo}", alpha=.75, s=26)
ax.set(xlabel="Teor alcoólico", ylabel="Ácido málico", title="Grupos em duas variáveis")
ax.legend()
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(SAIDA / "grupos.png", dpi=180)
plt.close(fig)

def converter(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(type(obj).__name__)

resumo = {
    "linhas": len(vinhos), "variaveis": len(variaveis.columns),
    "nulos": int(vinhos.isna().sum().sum()), "estatisticas": estatisticas,
    "frequencias": frequencias, "probabilidades": probabilidades,
    "comparacao": comparacao.to_dict(orient="records"),
    "perfil": perfil.reset_index().to_dict(orient="records"),
}
(SAIDA / "resumo.json").write_text(
    json.dumps(resumo, ensure_ascii=False, indent=2, default=converter), encoding="utf-8"
)
print("Resultados salvos em", SAIDA)
