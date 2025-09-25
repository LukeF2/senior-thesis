## Zero-shot Sentiment (Groq)

Returns one label from `labels` for input text using Groq’s chat API 

**Function:** `infer_sentiment(text, labels=("positive","negative"), model="llama-3.1-8b-instant")` (in `sentiment_infer.py`)

## Usage

```
$ python3 -i sentiment_infer.py
>>> infer_sentiment("I love this!")
"positive"
>>> infer_sentiment("This was awful.")
"negative"
```
