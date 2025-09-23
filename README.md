## Zero-shot Sentiment (Groq)

Returns one label from `labels` for input text using Groq’s chat API (no training/examples).

**Function:** `infer_sentiment(text, labels=("positive","negative"), model="llama-3.1-8b-instant")` (in `sentiment_infer.py`)

## Usage

Call `infer_sentiment(text, labels=...)` to get a single label (zero-shot).  

```{python}
from sentiment_infer import infer_sentiment

print(infer_sentiment("I love this!"))                         # -> "positive"
print(infer_sentiment("This was awful."))                      # -> "negative"
