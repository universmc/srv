from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Charger le modèle et le tokenizer
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')