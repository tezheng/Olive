import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-large-uncased'  # You can choose other BERT variants
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Example input text
text = "This is an example sentence for BERT inference."

# Tokenize the input text
encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

# Perform inference
with torch.inference_mode():  # Disable gradient calculation during inference
    outputs = model(**encoded_input)

# Extract the last hidden state (contextualized word embeddings)
last_hidden_states = outputs.last_hidden_state

# Extract the pooled output (sentence-level embedding)
pooled_output = outputs.pooler_output

# Print shapes for verification
print("Shape of last_hidden_states:", last_hidden_states.shape)  # [batch_size, sequence_length, hidden_size]
print("Shape of pooled_output:", pooled_output.shape)        # [batch_size, hidden_size]

# Example of getting the embedding for a specific token (e.g., the first token "this")
first_token_embedding = last_hidden_states[0, 0, :]
print("Shape of first_token_embedding:", first_token_embedding.shape)  # [hidden_size]

# If you have multiple sentences (batch inference)
sentences = [
    "This is the first sentence.",
    "This is the second sentence.",
]
encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    batch_outputs = model(**encoded_inputs)

batch_last_hidden_states = batch_outputs.last_hidden_state
batch_pooled_outputs = batch_outputs.pooler_output

print("\nBatch Inference:")
# [batch_size, sequence_length, hidden_size]
print("Shape of batch_last_hidden_states:", batch_last_hidden_states.shape)
print("Shape of batch_pooled_outputs:", batch_pooled_outputs.shape)       # [batch_size, hidden_size]


# Example of using the model for sentence classification (requires a different model head)

model_for_classification = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2)  # Example: 2 labels (e.g., positive/negative)
encoded_input_classification = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    classification_outputs = model_for_classification(**encoded_input_classification)

logits_classification = classification_outputs.logits
predicted_class_classification = torch.argmax(logits_classification)

print("\nSentence Classification:")
print("Logits:", logits_classification)
print("Predicted class:", predicted_class_classification)
