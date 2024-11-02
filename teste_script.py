import pandas as pd

# Criando um exemplo de DataFrame
data = {
    'nome': ['Ana', 'Bruno', 'Carlos'],
    'idade': [25, 30, 22],
    'cidade': ['São Paulo', 'Rio de Janeiro', 'Curitiba']
}

df = pd.DataFrame(data)

# Exibindo o DataFrame
print("Exemplo de DataFrame:")
print(df)
