# main.py
"""
Zoo ML API - versión sin pandas (lee zoo.data con csv).
- Usa scikit-learn si está disponible para entrenar un RandomForest.
- Si scikit-learn o numpy no están disponibles en el entorno, usa un clasificador por reglas (fallback).
- Endpoints: GET / , GET /schema , POST /predict , POST /train
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import csv
import traceback
import joblib

# Intentar importar numpy y sklearn; si fallan, habilitar fallback
HAS_NUMPY = False
HAS_SKLEARN = False
try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

try:
    if HAS_NUMPY:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# Config
DATA_FILE = "zoo.data"
MODEL_FILE = "zoo_rf_model.joblib"
SCALER_FILE = "zoo_scaler.joblib"

CLASS_MAP = {
    1: "mammal",
    2: "bird",
    3: "reptile",
    4: "fish",
    5: "amphibian",
    6: "insect",
    7: "other"
}

FEATURES = [
    "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator",
    "toothed", "backbone", "breathes", "venomous", "fins", "legs",
    "tail", "domestic", "catsize"
]

ALL_COLUMNS = ["name"] + FEATURES + ["type"]

app = Flask(__name__)
CORS(app)

MODEL = None
SCALER = None
USE_RULE_BASED = True  # by default until model loaded

# ---------- Helpers (CSV loader without pandas) ----------
def load_zoo_csv(path=DATA_FILE):
    """
    Lee zoo.data con csv.reader y retorna (names_list, X_list_of_lists, y_list)
    Espera sin cabecera; filas: name + 16 features + type
    """
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for r in rdr:
            if not r:
                continue
            rows.append(r)
    if not rows:
        raise ValueError("El archivo está vacío o no se pudo leer.")
    names = [r[0] for r in rows]
    X = []
    y = []
    for r in rows:
        # Aseguramos que tenga al menos 18 columnas (name + 16 + type)
        if len(r) < len(ALL_COLUMNS):
            # intentar rellenar con ceros si faltan
            r = r + ["0"] * (len(ALL_COLUMNS) - len(r))
        # features: indices 1..16 inclusive
        features = [int(float(v)) for v in r[1:1+len(FEATURES)]]
        cls = int(float(r[1+len(FEATURES)]))
        X.append(features)
        y.append(cls)
    return names, X, y

# ---------- ML training (if sklearn available) ----------
def train_model_from_file(path=DATA_FILE):
    global MODEL, SCALER, USE_RULE_BASED
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {path}")
    if not HAS_NUMPY or not HAS_SKLEARN:
        raise RuntimeError("NumPy o scikit-learn no están disponibles en este entorno.")
    names, X_list, y_list = load_zoo_csv(path)
    X = np.array(X_list, dtype=int)
    y = np.array(y_list, dtype=int)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(Xs, y)
    MODEL = clf
    SCALER = scaler
    joblib.dump(MODEL, MODEL_FILE)
    joblib.dump(SCALER, SCALER_FILE)
    USE_RULE_BASED = False
    return {"message": "trained", "samples": len(y)}

def try_load_saved_model():
    global MODEL, SCALER, USE_RULE_BASED
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
            MODEL = joblib.load(MODEL_FILE)
            SCALER = joblib.load(SCALER_FILE)
            USE_RULE_BASED = False
            return True
    except Exception:
        return False
    return False

# ---------- Rule-based fallback ----------
def rule_based_predict(payload):
    get = lambda k: int(float(payload.get(k, 0)))
    feathers = get("feathers")
    milk = get("milk")
    eggs = get("eggs")
    aquatic = get("aquatic")
    fins = get("fins")
    legs = get("legs")
    venomous = get("venomous")

    if feathers == 1:
        pred = 2
        reason = "tiene plumas -> bird"
    elif milk == 1:
        pred = 1
        reason = "produce leche -> mammal"
    elif aquatic == 1 and fins == 1:
        pred = 4
        reason = "acuático + aletas -> fish"
    elif eggs == 1 and venomous == 1:
        pred = 3
        reason = "pone huevos y es venenoso -> reptile"
    elif legs >= 6:
        pred = 6
        reason = "6+ patas -> insect"
    elif eggs == 1:
        pred = 3
        reason = "pone huevos -> reptile (fallback)"
    else:
        pred = 7
        reason = "no cumple reglas -> other"
    probs = {str(k): (1.0 if k == pred else 0.0) for k in CLASS_MAP.keys()}
    return {"pred_type": int(pred), "pred_label": CLASS_MAP[pred], "probs": probs, "explanation": reason}

# ---------- Model predict (uses MODEL if available) ----------
def predict_with_model(payload):
    global MODEL, SCALER
    if MODEL is None or SCALER is None:
        raise RuntimeError("Modelo no está cargado.")
    # Build feature vector in FEATURES order
    vec = []
    for f in FEATURES:
        val = payload.get(f, 0)
        try:
            vec.append(int(float(val)))
        except Exception:
            vec.append(0)
    if not HAS_NUMPY:
        raise RuntimeError("NumPy requerido para usar el modelo.")
    X = np.array([vec], dtype=float)
    Xs = SCALER.transform(X)
    pred = int(MODEL.predict(Xs)[0])
    probs = {}
    try:
        probs_arr = MODEL.predict_proba(Xs)[0]
        classes = MODEL.classes_
        for c, p in zip(classes, probs_arr):
            probs[str(int(c))] = float(p)
    except Exception:
        probs = {str(k): (1.0 if k == pred else 0.0) for k in CLASS_MAP.keys()}
    explanation = {}
    try:
        importances = MODEL.feature_importances_
        top_idx = np.argsort(importances)[::-1][:3]
        top_feats = [(FEATURES[i], float(importances[i])) for i in top_idx]
        explanation["top_features_global"] = top_feats
    except Exception:
        explanation["note"] = "no explanation"
    return {"pred_type": pred, "pred_label": CLASS_MAP.get(pred, "unknown"), "probs": probs, "explanation": explanation}

# ---------- Initialization ----------
try:
    if not try_load_saved_model():
        # if model files not present, but scikit available and zoo.data exists, train
        if HAS_NUMPY and HAS_SKLEARN and os.path.exists(DATA_FILE):
            train_model_from_file(DATA_FILE)
        else:
            USE_RULE_BASED = True
except Exception:
    USE_RULE_BASED = True

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Zoo classifier API (no-pandas)",
        "model_loaded": not USE_RULE_BASED,
        "notes": [
            "If model_loaded is false the API uses rule-based fallback.",
            "You can POST to /train with {'csv': '...'} to retrain (csv same format as zoo.data)."
        ]
    })

@app.route("/schema", methods=["GET"])
def schema():
    return jsonify({
        "features": FEATURES,
        "class_map": CLASS_MAP,
        "example": {f: 0 for f in FEATURES}
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON inválido, se esperaba objeto"}), 400

        # validate legs if present
        if "legs" in payload:
            try:
                if int(float(payload["legs"])) < 0:
                    return jsonify({"error": "legs debe ser >= 0"}), 400
            except Exception:
                return jsonify({"error": "legs debe ser numérico"}), 400

        if USE_RULE_BASED:
            res = rule_based_predict(payload)
            res["used_model"] = "rule_based"
        else:
            try:
                res = predict_with_model(payload)
                res["used_model"] = "trained_model"
            except Exception as e:
                # fallback if model prediction fails
                res = rule_based_predict(payload)
                res["used_model"] = "rule_based_after_error"
                res["model_error"] = str(e)
        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/train", methods=["POST"])
def train_endpoint():
    """
    POST {'csv': 'contenido del zoo.data (sin header)'} para re-entrenar.
    """
    global USE_RULE_BASED
    try:
        data = request.get_json(force=True)
        if not data or "csv" not in data:
            return jsonify({"error": "Enviar JSON con clave 'csv'"}), 400
        csv_text = data["csv"]
        tmp = "tmp_upload_zoo.data"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(csv_text)
        # try to train
        if not HAS_NUMPY or not HAS_SKLEARN:
            return jsonify({"error": "No hay soporte para entrenamiento en este entorno (numpy/sklearn faltan)."}), 400
        res = train_model_from_file(tmp)
        os.remove(tmp)
        return jsonify({"message": "modelo re-entrenado", "detail": res})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("Zoo API (no-pandas) arrancando. model_loaded:", not USE_RULE_BASED)
    app.run(host="0.0.0.0", port=port, debug=True)
