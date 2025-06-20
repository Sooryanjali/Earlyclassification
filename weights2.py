import numpy as np

# Given model accuracies
model_accuracies = np.array([0.7747, 0.8266, 0.8634])  # [model_30, model_50, model_70]

# Normalize weights so they sum to 1
classifier_weights = model_accuracies / np.sum(model_accuracies)

# Save the computed weights for later use
np.save("classifier_weights2.npy", classifier_weights)

# Print results
print("✅ Classifier Weights Computed & Saved Successfully!")
print("Model Accuracies:", model_accuracies)
print("Final Normalized Weights:", classifier_weights)
