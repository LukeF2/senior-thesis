import os

from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def infer_sentiment(text, labels=("positive", "negative"), examples=None, model="llama-3.1-8b-instant"):
	# 1. SYSTEM - instructions + output
	sys_msg = (
		"You are a strict sentiment classifier.\n"
		f"Allowed labels: {', '.join(labels)}.\n"
		"Reply with ONLY ONE WORD: Exactly ONE of the allowed labels."
	)
	
	# 2. Messages: system + user (zero shot?)
	messages = [
		{"role": "system", "content": sys_msg},
		{"role": "user", "content": text},
	]
	
	# 3. Call groq
	out = client.chat.completions.create(
		model=model,
		messages=messages,
		temperature=0,
	).choices[0].message.content.strip()
	
	# 4) Normalize to one of the allowed labels
	out_low = out.lower()
	for L in labels:
		if out_low == L.lower() or L.lower() in out_low:
			return L
	return labels[0]
