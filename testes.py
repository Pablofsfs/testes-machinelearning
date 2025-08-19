#Supervisionado
#Exemplo1
# Importar bibliotecas
import pandas as pd
from sklearn.linear_model import LinearRegression

# Criar um pequeno conjunto de dados
dados = pd.DataFrame({
    "area": [50, 60, 70, 80, 90],        # área da casa em m²
    "quartos": [1, 2, 2, 3, 3],          # número de quartos
    "preco": [150000, 200000, 250000, 300000, 350000]  # preço da casa
})

# Separar variáveis de entrada (X) e saída (y)
X = dados[["area", "quartos"]]
y = dados["preco"]

# Criar e treinar o modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Fazer uma previsão
nova_casa = [[85, 3]]
previsao = modelo.predict(nova_casa)
print("Preço previsto:", previsao[0])


#Exemplo2
from sklearn.tree import DecisionTreeClassifier

# Dados fictícios: [idade, pressão arterial]
dados = [[25, 120], [45, 140], [35, 130], [50, 160], [23, 110]]
rotulos = ["saudável", "doente", "saudável", "doente", "saudável"]

# Criar modelo de Árvore de Decisão
modelo = DecisionTreeClassifier()
modelo.fit(dados, rotulos)

# Prever novo paciente
novo_paciente = [[40, 150]]
previsao = modelo.predict(novo_paciente)
print("Diagnóstico previsto:", previsao[0])

#Exemplo3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Base de dados simples
emails = ["Promoção imperdível, clique agora!", 
          "Reunião de trabalho amanhã às 10h", 
          "Você ganhou um prêmio, acesse já, bet69.com!", 
          "Confirmação de consulta médica no proctologista"]

rotulos = ["spam", "não_spam", "spam", "não_spam"]

# Converter textos em vetores numéricos
vetorizador = CountVectorizer()
X = vetorizador.fit_transform(emails)

# Criar modelo Naive Bayes
modelo = MultinomialNB()
modelo.fit(X, rotulos)

# Testar com novo email
novo_email = ["Oferta exclusiva para você ganhar dinheiro rápido"]
X_novo = vetorizador.transform(novo_email)
print("Classificação:", modelo.predict(X_novo)[0])

#Não Supervisionado
#Exemplo1
from sklearn.cluster import KMeans
import numpy as np

# Dados: [idade, compras por mês]
clientes = np.array([[20, 3], [25, 4], [30, 6], [40, 20], [45, 22], [50, 25]])

# Aplicar K-Means (2 grupos)
modelo = KMeans(n_clusters=2, random_state=42)
modelo.fit(clientes)

# Ver a qual grupo cada cliente pertence
print("Grupos dos clientes:", modelo.labels_)

#Exemplo2
from sklearn.decomposition import PCA
import numpy as np

# Criar dados com 3 dimensões
dados = np.random.rand(10, 3)

# Reduzir para 2 dimensões
pca = PCA(n_components=2)
dados_reduzidos = pca.fit_transform(dados)

print("Dados originais:\n", dados)
print("\nDados reduzidos:\n", dados_reduzidos)

#Exemplo3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Documentos de exemplo
documentos = [
    "Gosto de futebol e esportes",
    "O time venceu o campeonato",
    "Receitas de bolo e sobremesas",
    "Adoro cuzinhar doces e massas",
    " Morango do amor com pescoço"
]

# Converter em vetores numéricos
vetorizar = TfidfVectorizer()
X = vetorizar.fit_transform(documentos)

# Aplicar K-Means com 2 clusters
modelo = KMeans(n_clusters=2, random_state=42)
modelo.fit(X)

print("Cluster de cada documento:", modelo.labels_)

#Por Reforço
#exemplo1
import random

posicao = 0  # posição inicial
objetivo = 5  # meta

for episodio in range(10):
    acao = random.choice([-1, 1])  # mover esquerda ou direita
    posicao += acao
    recompensa = 1 if posicao == objetivo else -0.1
    print(f"Episódio {episodio+1} | Posição: {posicao} | Recompensa: {recompensa}")


#exemplo2
import numpy as np

# Estados: 0 = início, 1 = perto do objetivo, 2 = objetivo
q_tabela = np.zeros((3, 2))  # 3 estados, 2 ações (esquerda, direita)

taxa_aprendizado = 0.5
desconto = 0.9

for episodio in range(20):
    estado = 0
    while estado != 2:
        acao = np.random.choice([0, 1])  # 0 = esquerda, 1 = direita
        if estado == 0 and acao == 1: proximo_estado, recompensa = 1, 0
        elif estado == 1 and acao == 1: proximo_estado, recompensa = 2, 1
        else: proximo_estado, recompensa = 0, -0.1
        q_tabela[estado, acao] += taxa_aprendizado * (recompensa + desconto * np.max(q_tabela[proximo_estado]) - q_tabela[estado, acao])
        estado = proximo_estado

print("Q-Tabela aprendida:\n", q_tabela)

#exemplo3
import numpy as np

recompensas_reais = [0.1, 0.5, 0.9]  # chances de recompensa em cada "braço"
valores_estimados = [0, 0, 0]
contagem_acoes = [0, 0, 0]

for passo in range(50):
    acao = np.random.choice([0, 1, 2])  # escolher um braço
    recompensa = 1 if np.random.rand() < recompensas_reais[acao] else 0
    contagem_acoes[acao] += 1
    valores_estimados[acao] += (recompensa - valores_estimados[acao]) / contagem_acoes[acao]

print("Valores estimados após 50 jogadas:", valores_estimados)


https://colab.research.google.com/drive/1cRK1x4WoChYE3QKlkvbzq8yA8Ce1chMD?usp=sharing
