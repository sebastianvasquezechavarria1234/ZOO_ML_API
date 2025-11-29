from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "model.pkl"

# ======================================================
# 1) INTENTAR CARGAR EL MODELO
# ======================================================

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            return model
        except:
            return None
    return None

model = load_model()


# ======================================================
# 2) REGLAS DE FALLBACK SI NO EXISTE MODELO
# ======================================================

def rule_based_prediction(features):
    """
    Predicción simple SIN modelo:
    Devuelve animales reales según características del dataset zoo.data
    """

    hair = features.get("hair", 0)
    feathers = features.get("feathers", 0)
    milk = features.get("milk", 0)
    airborne = features.get("airborne", 0)
    aquatic = features.get("aquatic", 0)
    fins = features.get("fins", 0)
    legs = features.get("legs", 0)
    venomous = features.get("venomous", 0)
    domestic = features.get("domestic", 0)

    # Mamíferos
    if hair == 1 and milk == 1:
        if legs == 4 and domestic == 1:
            return "cat"
        if legs == 4 and domestic == 0:
            return "tiger"
        return "mammal"

    # Aves
    if feathers == 1:
        if airborne == 1:
            return "eagle"
        return "bird"

    # Peces
    if aquatic == 1 and fins == 1:
        if venomous == 1:
            return "stingray"
        return "fish"

    # Reptiles
    if venomous == 1 and hair == 0 and feathers == 0:
        return "snake"

    # Anfibios
    if aquatic == 1 and airborne == 0 and milk == 0 and fins == 0:
        return "frog"

    # Insectos
    if legs == 6:
        return "insect"

    # Por defecto
    return "unknown"


# ======================================================
# 3) RUTA PRINCIPAL "/"
# ======================================================

@app.get("/")
def home():
    return {
        "service": "API Zoológico – Clasificador de Animales",
        "description": "Envía características de un animal y la API devolverá la especie estimada.",
        "example_body": {
            "hair": 1,
            "feathers": 0,
            "eggs": 0,
            "milk": 1,
            "airborne": 0,
            "aquatic": 0,
            "predator": 0,
            "toothed": 1,
            "backbone": 1,
            "breathes": 1,
            "venomous": 0,
            "fins": 0,
            "legs": 4,
            "tail": 1,
            "domestic": 1
        },
        "model_loaded": model is not None,
        "notes": [
            "Usa POST /predict para clasificar animales.",
            "Si el modelo no está cargado, se usa un sistema inteligente basado en reglas.",
            "Puedes reentrenar enviando un CSV a /train (si lo habilitas)."
        ]
    }


# ======================================================
# 4) RUTA DE PREDICCIÓN
# ======================================================

@app.post("/predict")
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Body JSON requerido"}), 400

    features = [
        data.get("hair", 0),
        data.get("feathers", 0),
        data.get("eggs", 0),
        data.get("milk", 0),
        data.get("airborne", 0),
        data.get("aquatic", 0),
        data.get("predator", 0),
        data.get("toothed", 0),
        data.get("backbone", 0),
        data.get("breathes", 0),
        data.get("venomous", 0),
        data.get("fins", 0),
        data.get("legs", 0),
        data.get("tail", 0),
        data.get("domestic", 0)
    ]

    # Si hay modelo → usar modelo
    if model:
        try:
            prediction = model.predict([features])[0]
            return jsonify({"animal": str(prediction), "via": "model"})
        except:
            pass

    # Si falla el modelo → reglas
    animal = rule_based_prediction(data)
    return jsonify({"animal": animal, "via": "rule-based"})


# ======================================================
# 5) INICIAR SERVIDOR (Render usa PORT)
# ======================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
