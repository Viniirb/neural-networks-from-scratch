from transformers import pipeline

print("--- Baixando Transformer (Hugging Face) ---")
classificador = pipeline("sentiment-analysis")

frases = [
    "This movie was fantastic really great acting",
    "Total waste of time terrible plot and boring",
    "The movie was okay, not great but not terrible either."
]

print("\n--- Análise com Transformer (BERT) ---")
for frase in frases:
    resultado = classificador(frase)[0]
    print(f"Frase: '{frase}'")
    print(f"Sentimento: {resultado['label']} | Confiança: {resultado['score']:.4f}\n")