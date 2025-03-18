import pandas as pd

def load_data():
    """
    Carrega os dados dos arquivos CSV
    """
    train = pd.read_csv("raw_data/train.csv", parse_dates=["date"])
    transactions = pd.read_csv("raw_data/transactions.csv", parse_dates=["date"])
    return train, transactions

def merge_data(train, transactions):
    """
    Realiza o merge do DataFrame train com transactions
    usando as chaves 'date' e 'store_nbr'.
    """
    merged_train = pd.merge(train, transactions, on=["date", "store_nbr"], how="left")
    return merged_train

def compute_ticket_medio(df):
    """
    Calcula o ticket médio para cada loja.
    Ticket médio = (Total de vendas da loja) / (Total de transações da loja)
    
    Parâmetro:
        df: DataFrame resultante do merge (deve conter as colunas "sales" e "num_transactions")
        
    Retorna:
        DataFrame com duas colunas: 'store_nbr' e 'ticket_medio'
    """
    # Evita divisão por zero se houver lojas com num_transactions igual a zero
    ticket_medio = df.groupby("store_nbr").apply(
        lambda x: x["sales"].sum() / x["num_transactions"].sum() if x["num_transactions"].sum() != 0 else 0
    ).reset_index(name="ticket_medio")
    
    return ticket_medio

def main():
    # Carrega os dados
    train, transactions = load_data()
    
    # Realiza o merge dos dados para incorporar o número de transações
    merged_train = merge_data(train, transactions)
    
    # Calcula o ticket médio por loja
    ticket_medio = compute_ticket_medio(merged_train)
    
    # Merge do ticket_medio com o DataFrame original para incluir a nova feature
    merged_train = pd.merge(merged_train, ticket_medio, on="store_nbr", how="left")
    
    # Exibe os primeiros registros do ticket médio para verificação
    print("Ticket Médio por Loja:")
    print(ticket_medio.head())
    
    # Salva o DataFrame com a nova feature para uso posterior
    merged_train.to_csv("raw_data/train_with_ticket_medio.csv", index=False)
    print("Arquivo 'train_with_ticket_medio.csv' salvo com sucesso!")

if __name__ == "__main__":
    main()
