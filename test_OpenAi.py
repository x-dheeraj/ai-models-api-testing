from dotenv import load_dotenv
from openai import OpenAI

load_dotenv() # Loads variables from the .env file into the environment

# creating a client 
client = OpenAI()

# Making an API request
response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
             {"role": "user", "content": "Hey there"}
            ]
        )

# printing the response
print(response.choices[0].messages.content)
