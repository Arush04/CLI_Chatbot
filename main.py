import os
import requests

# will use lanchain huggingface integration
def chat_with_llm(query, client):
    """
    Sends the user query + history to an LLM endpoint and returns the response.
    """
    response = client.models.generate_content(
        model="gemini-1.5",
        contents=query,
    )
    return response.text

def main():
    print("🤖 LLM Chat CLI (type 'exit' or 'quit' to end)\n")
    history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break
        client = genai.Client()

        # Send to LLM
        reply = chat_with_llm(user_input, client)
        print(f"AI: {reply}\n")

        # Keep history
        # history.append({"user": user_input, "assistant": reply})

if __name__ == "__main__":
    main()
