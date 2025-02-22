from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Charger le modèle et le tokenizer depuis le répertoire de votre bureau
model = GPTNeoForCausalLM.from_pretrained('wait/gpt-neo-2.7B')
tokenizer = GPT2Tokenizer.from_pretrained('wait/gpt-neo-2.7B')