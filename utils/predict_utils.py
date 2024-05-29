def predict_class(probability: float, threshold: float = 0.5):
    predicted_class = int(probability > threshold)
    confidence = probability if predicted_class == 1 else 1 - probability

    return predicted_class, confidence
