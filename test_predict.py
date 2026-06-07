from predictor import predict_disease

symptoms = ["anxiety", "depression"]

disease, confidence = predict_disease(symptoms)

print("Predicted Disease:", disease)
print(f"Confidence: {confidence:.2f}%")
