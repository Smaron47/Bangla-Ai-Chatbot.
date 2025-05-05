# Import libraries
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Path to saved model
MODEL_PATH = "/content/drive/My Drive/Chatbot_Models"

# Load the saved model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Chatbot pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Test the chatbot
def test_chatbot():
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        response = chatbot(user_input, max_length=100, num_return_sequences=1)
        print("Chatbot:", response[0]['generated_text'])

# Start chatbot testing
test_chatbot()
