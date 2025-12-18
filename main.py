from groq import Groq

client = Groq()

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": "Confirm Groq is working."}
    ],
)

print(response.choices[0].message.content)