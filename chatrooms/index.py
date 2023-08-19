from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Charger le modèle et le tokenizer
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')

# Définir la requête
prompt = "Traduisez le texte suivant en anglais: 'Bonjour tout le monde.'"

# Convertir la requête en entrée pour le modèle
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Générer une réponse
output = model.generate(input_ids, max_length=150, num_return_sequences=1)

# Décoder la réponse en texte
response = tokenizer.decode(output[0], skip_special_tokens=True)

print(f"Réponse: {response}")