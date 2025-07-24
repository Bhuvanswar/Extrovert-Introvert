import pandas as pd
from SRC.utility import load_object

# Load model, preprocessor, and label encoder
model = load_object("artifacts/model.pkl")
preprocessor = load_object("artifacts/proprocessor.pkl")
label_encoder = load_object("artifacts/label_encoder.pkl")

# Prepare custom samples
sample_input = pd.DataFrame([
    # Likely extrovert
    {
        "Time_spent_Alone": 1,
        "Stage_fear": "no",
        "Social_event_attendance": 8,
        "Going_outside": 9,
        "Drained_after_socializing": "no",
        "Friends_circle_size": 10,
        "Post_frequency": 9
    },
    # Likely introvert
    {
        "Time_spent_Alone": 9,
        "Stage_fear": "yes",
        "Social_event_attendance": 2,
        "Going_outside": 2,
        "Drained_after_socializing": "yes",
        "Friends_circle_size": 2,
        "Post_frequency": 1
    }
])

# Transform input
input_scaled = preprocessor.transform(sample_input)

# Predict
encoded_preds = model.predict(input_scaled)
encoded_preds = encoded_preds.astype(int)
print("Classes:", label_encoder.classes_)
decoded_preds = label_encoder.inverse_transform(encoded_preds)

# Display results
for i, pred in enumerate(decoded_preds):
    print(f"Sample {i+1} -> Personality Type: {pred}")
