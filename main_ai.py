import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Dados fictícios
data = {
    'anos_estudo': [10, 15, 12, 20, 8, 18, 10, 22, 6, 14, 9, 16, 12, 17, 14],
    'experiencia_trabalho': [2, 10, 5, 15, 1, 12, 3, 20, 0, 9, 2, 11, 4, 10, 7],
    'num_filhos': [0, 2, 1, 3, 0, 1, 0, 2, 0, 1, 0, 3, 1, 2, 1],
    'anos_casado': [0, 5, 2, 10, 0, 8, 0, 15, 0, 3, 0, 10, 1, 6, 4],
    'nivel_educacional': [1, 3, 2, 4, 1, 3, 1, 5, 1, 3, 2, 4, 2, 4, 3],
    'idade': [18, 35, 25, 45, 22, 40, 19, 50, 16, 30, 20, 42, 26, 36, 29]
}
df = pd.DataFrame(data)

# Separando dados de treino e teste
X = df.drop('idade', axis=1)
y = df['idade']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliando o modelo
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Erro médio absoluto (MAE): {mae}")

# Função de previsão de idade
def prever_idade():
    print("Responda às perguntas abaixo para prever a idade.")
    anos_estudo = int(input("Quantos anos você estudou? "))
    experiencia_trabalho = int(input("Quantos anos você tem de experiência de trabalho? "))
    num_filhos = int(input("Quantos filhos você tem? "))
    anos_casado = int(input("Quantos anos você está casado(a)? "))
    nivel_educacional = int(input("Qual é o seu nível educacional? (1=Fundamental, 2=Médio, 3=Superior, 4=Pós-graduação, 5=Doutorado) "))

    dados_usuario = pd.DataFrame([[anos_estudo, experiencia_trabalho, num_filhos, anos_casado, nivel_educacional]],
    columns=['anos_estudo', 'experiencia_trabalho', 'num_filhos', 'anos_casado', 'nivel_educacional'])

    idade_prevista = model.predict(dados_usuario)
    print(f"A idade prevista é: {idade_prevista[0]:.1f} anos.")

# Executar previsão
prever_idade()
