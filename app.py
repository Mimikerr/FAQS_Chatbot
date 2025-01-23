from flask import Flask, render_template, request, jsonify
from chatbot import find_top_5, load_cache, save_to_cache
import openai

app = Flask(__name__)

openai.api_key = ""

def process_with_chatgpt(query, top_faqs):
    """
    Processa a consulta do usuário com os FAQs retornados,
    gerando uma resposta consolidada.
    """
    if not top_faqs:
        return "Desculpe, não encontramos nada relacionado a essa pergunta no FAQ."

    print("\nTop 5 FAQs selecionadas:")
    for faq in top_faqs:
        print(f"Pergunta: {faq['ERRO']}, Similaridade: {faq['similarity']:.2f}")

    faqs_text = "\n\n".join(
        [f"Pergunta: {faq['ERRO']}\nResposta: {faq['SOLUCAO']}" for faq in top_faqs]
    )
    prompt = (
        f"Abaixo estão as perguntas e respostas mais relevantes da base de dados FAQ:\n\n"
        f"{faqs_text}\n\n"
        f"Pergunta do usuário: {query}\n\n"
        f"Baseado nessas informações, gere uma resposta consolidada que seja clara e útil para o usuário. "
        f"Se mais de uma resposta puder corrigir o problema, combine as soluções de maneira coesa."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Você é um assistente que responde apenas com informações da base de FAQs fornecida."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=400
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Erro ao processar a pergunta com ChatGPT: {e}")
        return "Desculpe, houve um erro ao processar sua pergunta."

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/ask', methods=['POST'])
def ask():
    """
    Processa a pergunta do usuário e retorna a melhor resposta
    com base nos FAQs relevantes.
    """
    query = request.get_json().get('question', '').strip()
    if not query:
        return jsonify({'solution': "Por favor, envie uma pergunta válida!"})

    try:
        top_faqs = find_top_5(query)
        if not top_faqs:
            return jsonify({'solution': "Desculpe, não encontramos nada relacionado a essa pergunta no FAQ."})

        solution = process_with_chatgpt(query, top_faqs)

        save_to_cache(query, solution)

        return jsonify({'solution': solution})
    except Exception as e:
        return jsonify({'solution': f"Erro ao processar a pergunta: {e}"})

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Rota para processar feedback do usuário sobre a resposta.
    """
    data = request.get_json()
    question = data.get('question', '').strip().lower()
    feedback = data.get('feedback', '').strip().lower()

    if not question or feedback not in ['sim', 'não']:
        return jsonify({'status': 'Erro: Dados inválidos.'}), 400

    cache = load_cache()

    if question in cache:
        entry = cache[question]
        entry['total'] += 1
        if feedback == 'sim':
            entry['positivos'] += 1

        save_to_cache(question, entry['resposta'], entry['positivos'], entry['total'])
        return jsonify({'status': 'Feedback salvo com sucesso!'})
    else:
        return jsonify({'status': 'Erro: Pergunta não encontrada no cache.'}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
