import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Criando um DataFrame com dados fictícios
data = {
    'sabe_fazer_contas': [1, 1, 0, 1, 0, 1, 1, 0, 0, 1],
    'gosta_de_jogar': [1, 1, 0, 1, 1, 0, 0, 1, 1, 0],
    'tem_telefone': [1, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    'idade': [10, 12, 8, 15, 6, 14, 13, 9, 7, 16]
}

df = pd.DataFrame(data)

# Dividindo os dados em features e target
X = df[['sabe_fazer_contas', 'gosta_de_jogar', 'tem_telefone']]
y = df['idade']

# Dividindo em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializando o modelo
model = LinearRegression()

# Treinando o modelo
model.fit(X_train, y_train)

def predict_age():
    print("Responda às seguintes perguntas com 1 (Sim) ou 0 (Não):")

    sabe_fazer_contas = int(input("Você sabe fazer contas? "))
    gosta_de_jogar = int(input("Você gosta de jogar? "))
    tem_telefone = int(input("Você tem um telefone? "))

    # Criando um array com as respostas
    user_input = [[sabe_fazer_contas, gosta_de_jogar, tem_telefone]]

    # Fazendo a previsão
    predicted_age = model.predict(user_input)
    print(f"Sua idade estimada é: {predicted_age[0]:.2f} anos.")

# Chamar a função para prever a idade
if __name__ == "__main__":
    predict_age()
