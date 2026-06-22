%pip install transformers torch
from transformers import pipeline

sentiment = pipeline(
    "sentiment-analysis",
    framework="pt"
)

reviews = [
    'The product quality is amazing! I am really happy with my purchase.',
    'Terrible customer service. I will not buy from here again.',
    'It was okay, not great but not terrible either.',
    'Absolutely love it! Fast shipping and great packaging.',
    'The item arrived broken and the support team was unhelpful.'
]

results = sentiment(reviews)

for review, result in zip(reviews, results):
    print(f"Review: {review}")
    print(f"Sentiment: {result['label']}, Confidence: {result['score']:.2f}\n")
