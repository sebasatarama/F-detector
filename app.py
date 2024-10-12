from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Define el modelo y el tokenizer
model_name = "sebasatarama/F-DetectorModel"  # Reemplaza con el nombre de tu modelo en Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Inicializa la aplicación FastAPI
app = FastAPI()

# Define la estructura de los datos de entrada
class SentencesInput(BaseModel):
    sentences: list[str]

# Crea un endpoint para hacer predicciones
@app.post("/predict")
async def predict(input: SentencesInput):
    # Lista para almacenar los resultados de cada oración que cumplen los criterios
    results = []

    # Itera sobre cada oración
    for sentence in input.sentences:
        # Tokeniza la oración
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        
        # Realiza la predicción
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).tolist()[0]

        # Determina la etiqueta de mayor probabilidad
        max_prob = max(probabilities)
        label = probabilities.index(max_prob)
        
        # Ajusta la probabilidad multiplicándola por 0.65

        # Agrega el resultado solo si cumple con los criterios
        if label != 10 and max_prob > 0.7:
            adjusted_probability = max_prob * 0.65
            results.append({
                "label": label,
                "probability": round(adjusted_probability * 100, 2),  # Formato en porcentaje
                "sentence": sentence
            })

    # Devuelve directamente la lista de resultados
    return results

# Ejecuta el servidor
# uvicorn main:app --reload  # Para ejecutar el servidor desde la línea de comandos
