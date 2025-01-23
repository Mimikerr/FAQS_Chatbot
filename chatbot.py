import pyodbc
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import csv

database_path = r"J:\Projetos\ChatBot\FAQS_LIMPOS.mdb"
conn_str = (
    r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
    rf"DBQ={database_path};"
)

local_model_path = "j:/Projetos/ChatBot/modelo"
CACHE_FILE = "cache_feedback.csv"

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModel.from_pretrained(local_model_path)

def load_faqs_from_db():
    """
    Carrega a tabela FAQS do banco Access em um DataFrame.
    """
    try:
        conn = pyodbc.connect(conn_str)
        query = "SELECT * FROM FAQS"
        faqs_df = pd.read_sql(query, conn)
        conn.close()
        return faqs_df
    except Exception as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        return pd.DataFrame()

def preprocess_text(text):
    """
    Normaliza e processa texto para uniformizar a comparação.
    """
    if not isinstance(text, str):
        return ""
    return ' '.join(text.replace("\n", " ").strip().lower().split())

def get_embedding(text, output_dim=768):
    """
    Gera o embedding para um texto utilizando o modelo carregado.
    """
    if not isinstance(text, str) or not text.strip():
        return torch.zeros(output_dim) 
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

    if embedding.size(0) > output_dim:
        return embedding[:output_dim]
    elif embedding.size(0) < output_dim:
        return torch.cat([embedding, torch.zeros(output_dim - embedding.size(0))])
    return embedding


def load_cache():
    """
    Carrega o cache de respostas e feedbacks do arquivo.
    """
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                cache[row['Pergunta'].strip().lower()] = {
                    "resposta": row['Resposta'].strip(),
                    "positivos": int(row['Feedback_Positivo']),
                    "total": int(row['Feedback_Total']),
                }
    return cache

def save_to_cache(question, answer, positive_feedback=0, total_feedback=0):
    """
    Salva ou atualiza uma entrada no cache.
    """
    cache = load_cache()
    question = question.strip().lower()

    cache[question] = {
        "resposta": answer.strip(),
        "positivos": positive_feedback,
        "total": total_feedback,
    }

    with open(CACHE_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Pergunta", "Resposta", "Feedback_Positivo", "Feedback_Total"])
        writer.writeheader()
        for q, data in cache.items():
            writer.writerow({
                "Pergunta": q,
                "Resposta": data["resposta"],
                "Feedback_Positivo": data["positivos"],
                "Feedback_Total": data["total"],
            })

def process_from_faq(query):
    """
    Busca a melhor resposta da base de dados (FAQ).
    """
    top_5 = find_top_5(query)
    if not top_5:
        return "Desculpe, não encontramos nada relacionado a essa pergunta no FAQ."

    print("\nTop 5 FAQs selecionadas:")
    for faq in top_5:
        print(f"Pergunta: {faq['ERRO']}, Similaridade: {faq['similarity']:.2f}")

    best_match = max(top_5, key=lambda x: x['similarity'])
    if best_match['similarity'] < 0.7:
        return "Desculpe, não encontramos uma resposta relevante na base de dados."

    return best_match['SOLUCAO']

faqs_df = load_faqs_from_db()

if not faqs_df.empty:
    faqs_df['ERRO'] = faqs_df['ERRO'].apply(preprocess_text)
    faqs_df['SOLUCAO'] = faqs_df['SOLUCAO'].apply(preprocess_text)
    faqs_df['embedding'] = faqs_df['ERRO'].apply(get_embedding)

def find_top_5(query, similarity_threshold=0.2):
    """
    Retorna as Top 5 FAQs mais relevantes para uma consulta.
    """
    query = preprocess_text(query)
    query_embedding = get_embedding(query).numpy()

    faqs_df['similarity'] = faqs_df['embedding'].apply(
        lambda emb: cosine_similarity([query_embedding], [emb.numpy()])[0][0]
    )

    faqs_sorted = faqs_df.sort_values(by='similarity', ascending=False)
    top_faqs = faqs_sorted[faqs_sorted['similarity'] >= similarity_threshold].head(5)

    if not top_faqs.empty:
        return top_faqs[['ERRO', 'SOLUCAO', 'similarity']].to_dict(orient='records')
    return []

