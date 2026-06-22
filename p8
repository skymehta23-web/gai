%pip install cohere langchain langchain-cohere

with open('Text1.txt', 'w') as f:
    f.write("""
Generative AI is a branch of artificial intelligence that creates new, human-like outputs
such as writing, visuals, or sound based on vast datasets. It is used to automate content
creation, enhance productivity, enable personalized experiences, and power new product
innovations. Key use cases include marketing, product design, customer service, and
software development. Generative AI uses machine learning models, specifically deep
learning architectures such as GANs, transformers, and VAEs.
""")
print('Text1.txt created.')

import os
from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate

os.environ['COHERE_API_KEY'] = 'qnwGxMHhUwMeQRoDOU2MSBLSJcLKjAZu5TQYYHJN'   

with open('Text1.txt', 'r', encoding='utf-8') as f:
    document_text = f.read()
print('Original Document:\n', document_text)

llm = ChatCohere()

prompt = PromptTemplate(
    input_variables=['document'],
    template="""
You are a helpful assistant.
Given the following document, summarize it in bullet points.
{document}
Summary:
"""
)

chain = prompt | llm
response = chain.invoke({'document': document_text})
print('\nSummary:\n', response.content)
