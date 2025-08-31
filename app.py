from flask import Flask, render_template, request, redirect, url_for, flash
import pdfplumber, joblib, os

app = Flask(__name__)
app.secret_key = "dev"  # só para flash messages em dev

MODEL_PATH = os.path.join("static", "model.pkl")
VEC_PATH = os.path.join("static", "vectorizer.pkl")

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
vectorizer = joblib.load(VEC_PATH) if os.path.exists(VEC_PATH) else None

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

def extract_text_from_pdf(file_storage):
    text = ""
    with pdfplumber.open(file_storage) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text.strip()

def auto_reply(pred_label):
    # 0 = produtivo / não spam | 1 = improdutivo / spam
    if pred_label == 0:
        return "Obrigado pelo seu contato. Recebemos sua mensagem e retornaremos em breve."
    return "Mensagem identificada como improdutiva. Nenhuma ação necessária."

@app.route("/classify", methods=["POST"])
def classify():
    # 1) tenta pegar texto do textarea
    text = (request.form.get("email_text") or "").trim() if hasattr(str, "trim") else (request.form.get("email_text") or "").strip()

    # 2) se não veio texto, tenta PDF
    if not text:
        file = request.files.get("file")
        if file and file.filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file)

    if not text:
        flash("Envie um PDF ou cole o texto do e-mail.")
        return redirect(url_for("index"))

    if not (model and vectorizer):
        return "⚠️ Modelo não encontrado. Rode 'python train_model.py' para gerar static/model.pkl e static/vectorizer.pkl."

    X = vectorizer.transform([text])
    pred = int(model.predict(X)[0])
    label = "✅ Não SPAM / Produtivo" if pred == 0 else "🚨 SPAM / Improdutivo"
    response = auto_reply(pred)

    snippet = text[:800]
    return render_template("result.html", result=label, content=snippet, auto_response=response)

if __name__ == "__main__":
    app.run(debug=True)
