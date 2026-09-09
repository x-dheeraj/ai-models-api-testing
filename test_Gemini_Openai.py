from openai import OpenAI

client = OpenAI(
    api_key="AQ.Ab8RNKLwvuoqzpKJK9zz1YmVV_", # api_key is modified not original
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-3.8-flash",
    messages=[
        {   "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Explain Vector Embeddings in few lines"
        }
    ]
)

print(response.choices[0].message)

