import os
import nltk

# Set NLTK data path to a writable directory
nltk.data.path.append("./nltk_data")

# Create NLTK data directory if it doesn't exist
os.makedirs("./nltk_data", exist_ok=True)

# Download required NLTK data
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
