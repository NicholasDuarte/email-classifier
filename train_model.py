from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# --- Base de exemplo (pequena só pra funcionar) ---
emails = [
    "Promoção imperdível, clique aqui para ganhar um prêmio",   # spam
    "Você foi selecionado para ganhar um cupom grátis",        # spam
    "Oferta especial somente hoje, não perca",                 # spam
    "Reunião amanhã às 10h com a equipe de projetos",          # não spam
    "Segue em anexo o relatório solicitado",                   # não spam
    "Confirmando nossa reunião da semana que vem",             # não spam
]

labels = [1, 1, 1, 0, 0, 0]  # 1 = spam, 0 = não spam

# --- Vetorização ---
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(emails)

# --- Treinamento ---
model = MultinomialNB()
model.fit(features, labels)

# --- Salvar os arquivos ---
joblib.dump(model, "static/model.pkl")
joblib.dump(vectorizer, "static/vectorizer.pkl")

print("✅ Modelo e vectorizer treinados e salvos em 'static/'")
