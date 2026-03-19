# 🎭 Análise de Sentimentos em Português — B2W Reviews

Projeto de NLP para classificação de sentimentos em avaliações de produtos do e-commerce brasileiro, utilizando o dataset público **B2W-Reviews01** com mais de 130 mil avaliações reais em português.

---

## 📌 Objetivo

Classificar avaliações de clientes em três categorias de sentimento — **positivo**, **negativo** e **neutro** — comparando uma abordagem clássica (TF-IDF + Regressão Logística) com um modelo de linguagem estado-da-arte (BERTimbau), e investigando as decisões do modelo com técnicas de explicabilidade.

---

## 📂 Estrutura do Repositório

```
sentiment-analysis-ptbr/
│
├── sentiment_analysis_ptbr.ipynb   # notebook único com todo o pipeline
└── README.md
```

---

## 📦 Dataset

**B2W-Reviews01** — dataset público da Americanas S.A. com avaliações reais coletadas entre janeiro e maio de 2018.

- 130.000+ avaliações em português brasileiro
- Campos utilizados: `review_text` (feature) e `overall_rating` (label)
- Labels derivados da nota: positivo (≥4), neutro (=3), negativo (≤2)
- Distribuição: 61% positivo / 26% negativo / 13% neutro

> Download: [github.com/b2wdigital/b2w-reviews01](https://github.com/b2wdigital/b2w-reviews01)

---

## 🔧 Pré-processamento

O texto passou pelas seguintes etapas antes da modelagem:

- Conversão para minúsculas
- Remoção de números e pontuação (preservando caracteres do português)
- Remoção de espaços extras e quebras de linha
- Remoção de stopwords em português via NLTK — com preservação intencional de palavras de negação (`não`, `nunca`, `jamais`, `nada`, `nem`, `nenhum`) por seu impacto direto no sentimento

> **Decisão documentada:** manter palavras de negação foi crucial — sem elas, bigramas como `"não recomendo"` e `"não funciona"` seriam destruídos, prejudicando significativamente a detecção de sentimento negativo.

---

## 🤖 Modelos

### Modelo 1 — TF-IDF + Regressão Logística (Baseline)

Pipeline clássico e interpretável como ponto de partida.

**Configuração do TF-IDF:**
```python
TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),   # unigramas e bigramas
    min_df=5,
    max_df=0.95
)
```

O uso de bigramas foi decisivo — expressões como `"não recomendo"` (peso 7.1) e `"não funciona"` (peso 4.0) só existem com `ngram_range=(1, 2)`.

**Resultados:**

| Classe | Precision | Recall | F1-Score |
|---|---|---|---|
| Negativo | 0.84 | 0.88 | 0.86 |
| Neutro | 0.33 | 0.53 | 0.41 |
| Positivo | 0.92 | 0.79 | 0.85 |
| **Weighted F1** | | | **0.80** |

---

### Modelo 2 — BERTimbau (Fine-tuning)

Fine-tuning do modelo `neuralmind/bert-base-portuguese-cased`, pré-treinado em textos brasileiros (Wikipedia, notícias, textos jurídicos).

**Configuração:**
```
max_length    = 64 tokens   (cobre 90%+ dos textos com folga)
batch_size    = 32
epochs        = 3
learning_rate = 2e-5
otimizador    = AdamW com linear scheduler e warmup
```

**Resultados por época:**

| Época | Loss | Weighted F1 |
|---|---|---|
| 1 | 0.4588 | 0.8276 |
| 2 | 0.3599 | 0.8388 |
| 3 | 0.3025 | 0.8433 |

**Resultados finais:**

| Classe | Precision | Recall | F1-Score |
|---|---|---|---|
| Negativo | 0.88 | 0.91 | 0.89 |
| Neutro | 0.48 | 0.32 | 0.39 |
| Positivo | 0.89 | 0.94 | 0.91 |
| **Weighted F1** | | | **0.84** |

---

## 📊 Comparativo Final

| Modelo | Weighted F1 | Negativo F1 | Neutro F1 | Positivo F1 |
|---|---|---|---|---|
| TF-IDF + Regressão Logística | 0.80 | 0.86 | 0.41 | 0.85 |
| BERTimbau | **0.84** | **0.89** | 0.39 | **0.91** |
| Ganho | **+0.04** | +0.03 | -0.02 | +0.06 |

O BERTimbau superou a Regressão Logística em +4 pontos de weighted F1, com ganhos expressivos em positivo (+6pp) e negativo (+3pp).

---

## 🔍 Explicabilidade (SHAP)

A análise com `transformers-interpret` revelou os padrões aprendidos pelo BERTimbau:

**Classe Negativo** — palavras com maior peso:
- `"não recomendo"` (7.1), `"não"` (6.1), `"não gostei"` (4.8)
- `"péssimo"` (4.6), `"ruim"` (4.3), `"não funciona"` (4.0)

**Classe Neutro** — palavras com maior peso:
- `"bom"` (4.3), `"porém"` (2.8), `"poderia"` (2.5), `"pouco"` (2.2)
- O modelo aprendeu que o neutro é a classe do **contraste e da ressalva**

**Classe Positivo** — palavras com maior peso:
- `"excelente"` (8.8), `"ótimo"` (6.7), `"amei"` (4.9), `"recomendo"` (4.1)

---

## 💡 Principais Insights

**1. O problema do neutro é inerente ao dado, não ao modelo**

A classe neutra teve F1 ~0.40 em ambos os modelos. A análise SHAP confirmou o motivo: reviews com nota 3 frequentemente contêm texto positivo — o cliente estava insatisfeito com algo que não verbalizou. Nenhum modelo consegue classificar corretamente o que não está no texto.

**2. Bigramas foram essenciais para o português**

Com `ngram_range=(1,2)`, o TF-IDF capturou expressões de negação que unigramas sozinhos perderiam. `"não recomendo"` foi o feature com maior peso para negatividade.

**3. Stopwords precisam ser tratadas com contexto**

A lista padrão do NLTK remove `"não"` — uma decisão correta para tarefas genéricas, mas **errada para análise de sentimentos**. Preservar palavras de negação foi uma das decisões mais impactantes do projeto.

**4. O BERT entende contexto, o TF-IDF não**

O SHAP mostrou que o BERTimbau aprendeu que palavras como `"apenas"` e `"poderia"` reduzem o sentimento positivo mesmo sem serem negativas — algo que o TF-IDF não captura por tratar palavras de forma isolada.

---

## 🛠️ Tecnologias

- Python 3.10+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- nltk
- transformers (Hugging Face)
- torch (PyTorch)
- transformers-interpret

---

## ▶️ Como Reproduzir

```bash
# 1. Clone o repositório
git clone https://github.com/Lomba227/sentiment-analysis-ptbr.git

# 2. Instale as dependências
pip install pandas numpy matplotlib seaborn scikit-learn nltk transformers torch transformers-interpret

# 3. Baixe o dataset
wget https://github.com/b2wdigital/b2w-reviews01/raw/master/B2W-Reviews01.csv

# 4. Abra o notebook
jupyter notebook sentiment_analysis_ptbr.ipynb
```

> Para o fine-tuning do BERTimbau, recomenda-se GPU. O Google Colab com T4 GPU é suficiente — tempo de treino aproximado: 30-40 minutos.

---

## 👤 Autor

Feito como projeto de portfólio para aprendizado de NLP aplicado ao português brasileiro.
