import cohere
import string
import gensim.downloader as api
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")

co = cohere.ClientV2("")

model = api.load("glove-wiki-gigaword-50")

def similar_words(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())

    result = []
    for word in words:
        if word not in stop_words and word not in string.punctuation:
            if word in model.key_to_index:
                sims = model.most_similar(word, topn=3)
                for sim_word, _ in sims:
                    result.append(sim_word)

    return " ".join(result)

def generate(prompt):
    response = co.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.message.content[0].text

prompt1 = "What is the use of Artificial Intelligence in modern society?"
print("\nCohere Response:\n", generate(prompt1))

original_prompt = "Describe the future of Artificial Intelligence in education."
enriched_prompt = original_prompt + " " + similar_words(original_prompt)

print("\nOriginal Prompt:", original_prompt)
print("\nGenerated Response (Original Prompt):\n", generate(original_prompt))

print("\nEnriched Prompt:", enriched_prompt)
print("\nGenerated Response (Enriched Prompt):\n", generate(enriched_prompt))
