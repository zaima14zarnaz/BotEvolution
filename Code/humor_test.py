from transformers import pipeline

# Initialize the text classification pipeline with the humor detection model
classifier = pipeline("text-classification", model="mohameddhiab/humor-no-humor")

# Example text to classify
text = "Why don't scientists trust atoms? Because they make up everything."

# Perform classification
result = classifier(text)

# Output the result
print(result)
