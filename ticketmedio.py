import os
import pandas as pd

def load_data():
    """
    Carrega os dados dos arquivos CSV.
    Ajuste os caminhos se necessário.
    """
    try:
        train = pd.read_csv("raw_data/train.csv", parse_dates=["date"])
        transactions = pd.read_csv("raw_data/transactions.csv", parse_dates=["date"])
    except FileNotFoundError as e:
        print(f"Erro ao carregar os dados: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        print(f"Arquivo CSV vazio encontrado: {e}")
        raise
    except pd.errors.ParserError as e:
        print(f"Erro ao analisar o arquivo CSV: {e}")
        raise
    return train, transactions

def merge_data(train, transactions):
    """
    Realiza o merge do DataFrame 'train' com 'transactions'
    usando as chaves 'date' e 'store_nbr'.
    """
    merged_train = pd.merge(train, transactions, on=["date", "store_nbr"], how="left")
    return merged_train

def compute_ticket_medio(df):
    """
    Calcula o ticket médio para cada loja.
    Ticket médio = (Total de vendas da loja) / (Total de transações da loja)
    
    Se a coluna 'num_transactions' não existir, tenta usar 'transactions'.
    
    Retorna:
        DataFrame com duas colunas: 'store_nbr' e 'ticket_medio'
    """
    # Verifica qual coluna de transações está disponível
    if "num_transactions" in df.columns:
        trans_col = "num_transactions"
    elif "transactions" in df.columns:
        trans_col = "transactions"
    else:
        raise KeyError("Nenhuma coluna de transações encontrada ('num_transactions' ou 'transactions').")
    
    # Calcula o ticket médio para cada loja
    ticket_medio = df.groupby("store_nbr").apply(
        lambda x: x["sales"].sum() / x[trans_col].sum() if x[trans_col].sum() != 0 else 0
    ).reset_index(name="ticket_medio")
    
    return ticket_medio

def main():
    # Mostra o diretório atual
    print("Current working directory:", os.getcwd())
    
    # Carrega os dados
    print("Loading data...")
    train, transactions = load_data()
    
    # Realiza o merge dos dados
    print("Merging data...")
    merged_train = merge_data(train, transactions)
    
    # Calcula o ticket médio por loja
    print("Computing ticket average...")
    ticket_medio = compute_ticket_medio(merged_train)
    
    # Merge do ticket_medio com o DataFrame original
    merged_train = pd.merge(merged_train, ticket_medio, on="store_nbr", how="left")
    
    # Exibe os primeiros registros do ticket médio para verificação
    print("Ticket Médio por Loja (primeiras 5 linhas):")
    print(ticket_medio.head())
    
    # Garante que a pasta raw_data existe
    output_dir = "raw_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Pasta '{output_dir}' criada.")
    else:
        print(f"Pasta '{output_dir}' já existe.")
    
    # Salva o DataFrame com a nova feature
    output_path = os.path.join(output_dir, "train_with_ticket_medio.csv")
    merged_train.to_csv(output_path, index=False)
    print(f"Arquivo '{output_path}' salvo com sucesso!")

if __name__ == "__main__":
    main()
