from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

app = Flask(__name__)


colnames = [
    "name","hair","feathers","eggs","milk","airborne","aquatic","predator",
    "toothed","backbone","breathes","venomous","fins","legs","tail","domestic",
    "catsize","class_type"
]

data = pd.read_csv("zoo.data", names=colnames)

X = data.drop(columns=["name", "class_type"])


y = data["class_type"]


class_to_name = data.groupby("class_type")["name"].first().to_dict()


model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)


@app.route("/predict", methods=["POST"])
def predict():
    body = request.json


    required = list(X.columns)
    for field in required:
        if field not in body:
            return jsonify({"error": f"Missing field: {field}"}), 400

    values = [body[col] for col in X.columns]


    predicted_class = int(model.predict([values])[0])


    animal_name = class_to_name.get(predicted_class, "Unknown")

    return jsonify({
        "predicted_class": predicted_class,
        "animal": animal_name,
        "input_data": body
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)