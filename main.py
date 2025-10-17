import os
import torch
import requests
import transformers
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from typing import Optional

# ========================
# CONFIGURATION
# ========================
CHROMA_DIR = "./chroma_store"
TOMORROW_API_KEY = "your key here"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# ========================
# LOAD VECTOR DB
# ========================
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model
)

# ========================
# MODEL + TOKENIZER
# ========================
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

def get_weather(city: str):
    """Fetch live weather information using Tomorrow.io Realtime API."""
    try:
        url = f"https://api.tomorrow.io/v4/weather/realtime?location={city}&apikey={TOMORROW_API_KEY}"

        headers = {
            "accept": "application/json",
            "accept-encoding": "deflate, gzip, br"
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        values = data["data"]["values"]
        temp = values.get("temperature")
        weather_code = values.get("weatherCode")

        # Map weather code to description
        description = get_weather_description(weather_code) if weather_code else "unknown conditions"

        print(f"Weather fetched for {city}: {temp}°C")
        return f"The weather in {city} is currently {description} with a temperature of {temp}°C."

    except requests.RequestException as e:
        return f"Error fetching weather data: {e}"
    except KeyError as e:
        return f"Error: unexpected response format from weather API. Missing: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


def get_weather_description(code: int) -> str:
    """Map Tomorrow.io weather codes to human-readable descriptions."""
    weather_codes = {
        0: "Unknown",
        1000: "Clear, Sunny",
        1100: "Mostly Clear",
        1101: "Partly Cloudy",
        1102: "Mostly Cloudy",
        1001: "Cloudy",
        2000: "Fog",
        2100: "Light Fog",
        4000: "Drizzle",
        4001: "Rain",
        4200: "Light Rain",
        4201: "Heavy Rain",
        5000: "Snow",
        5001: "Flurries",
        5100: "Light Snow",
        5101: "Heavy Snow",
        6000: "Freezing Drizzle",
        6001: "Freezing Rain",
        6200: "Light Freezing Rain",
        6201: "Heavy Freezing Rain",
        7000: "Ice Pellets",
        7101: "Heavy Ice Pellets",
        7102: "Light Ice Pellets",
        8000: "Thunderstorm"
    }
    return weather_codes.get(code, f"Weather code {code}")

def tool_selector(user_query: str) -> tuple[str, str]:
    """
    Uses the LLM to classify the user's query into one of three categories (general, weather, both)
    Returns: (tool_name, reasoning)
    """
    classification_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Classify the query into **exactly** one of the following categories:
- weather: when asked about anything related to weather
- general: when asked about anything but the weather
- general+weather: when asked about weather and other information

Respond in this **exact** format:
Category: [category_name]
Reason: [one short sentence explaining why]<|eot_id|><|start_header_id|>user<|end_header_id|>

Query: What is Canonical?
Category:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Category: general
Reason: The query asks about a company/product, not weather.<|eot_id|><|start_header_id|>user<|end_header_id|>

Query: What's the weather in London?
Category:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Category: weather
Reason: The query explicitly asks about weather conditions.<|eot_id|><|start_header_id|>user<|end_header_id|>

Query: What is Canonical and how is the weather in London?
Category:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Category: general+weather
Reason: The query asks about both company information and weather.<|eot_id|><|start_header_id|>user<|end_header_id|>

Query: {user_query}
Category:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    output = pipeline(
        classification_prompt,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    full_output = output[0]["generated_text"][len(classification_prompt):].strip()

    category = "general"
    reason = "No classification found"

    # Extract category
    category_match = re.search(r"Category:\s*(general\+weather|weather|general)", full_output, re.IGNORECASE)
    if category_match:
        category = category_match.group(1).lower()
    else:
        full_output_lower = full_output.lower()
        if "general+weather" in full_output_lower or "general and weather" in full_output_lower:
            category = "general+weather"
        elif "weather" in full_output_lower:
            category = "weather"
        elif "general" in full_output_lower:
            category = "general"

    # Extract reason
    reason_match = re.search(r"Reason:\s*(.+?)(?:\n|<|$)", full_output, re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()

    return category, reason


def agent_respond(user_query: str, vectorstore: Optional[object] = None) -> str:
    """
    ReAct-style agent that:
      1. Uses LLM-based `tool_selector` to decide which tools to use.
      2. Collects only the required context.
      3. Generates the final answer using the chosen context(s).
    """
    print("🔍 Selecting tools...")
    selected_tool, reasoning = tool_selector(user_query)
    print(f"Tool selected: {selected_tool}")

    context = ""
    weather_info = ""

    # Retrieve from vectorstore if "general" or "general+weather"
    if selected_tool in ["general", "general+weather"] and vectorstore:
        try:
            results = vectorstore.similarity_search(user_query, k=5)
            if results:
                print("🧠 Getting data from knowledge base")
                context += "\n\n--- Retrieved Knowledge ---\n"
                for doc in results:
                    context += f"{doc.page_content}\n"
        except Exception as e:
            print(f"⚠️ Vector store error: {e}")

    # Get weather info if "weather" or "general+weather"
    if selected_tool in ["weather", "general+weather"]:
        detected_city = None
        cities = ["Bangalore", "Delhi", "London", "New York", "Tokyo", "Toronto", "Mumbai", "Paris"]

        for city in cities:
            if city.lower() in user_query.lower():
                detected_city = city
                break

        if detected_city:
            print(f"🌦️ Fetching weather for {detected_city}...")
            weather_info = get_weather(detected_city)
        else:
            weather_info = "Weather was requested, but no specific city name was detected."

    # System prompt
    system_prompt = """You are a helpful assistant. Follow these rules:
    1. Judge from the prompt if question can be answered with information from knowledge base or weather information or both. Below are some examples for your reference.
    Example:
    Context: "Canonical offers data solutions including PostgreSQL, MySQL, and MongoDB."
    Question: "What databases does Canonical support and how is the weather in london?"
    Answer: "Based on the information provided, Canonical offers support for several databases including PostgreSQL, MySQL, and MongoDB as part of their data solutions. The Weather fetched for Delhi is 28.6°C"

    Now answer the user's question using ONLY the context provided below. If no relevant context exists, say so."""

    # Include only the contexts relevant to the chosen tool(s)
    user_content = f"""
    Question: {user_query}

    Below information contains retrieved knowledge and weather information (based on selected tools):

    {f"Context from knowledge base:\n{context}" if context and selected_tool in ['general', 'general+weather'] else ""}
    {f"Current weather information:\n{weather_info}" if weather_info and selected_tool in ['weather', 'general+weather'] else ""}
    """

    # Compose full prompt
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    print("🧩 Generating final response...")
    output = pipeline(
        prompt,
        max_new_tokens=300,
        do_sample=False,
        temperature=None,
        top_p=None,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Extract generated text
    full_output = output[0]["generated_text"]
    response_text = full_output[len(prompt):].strip()
    response_text = response_text.split("<|eot_id|>")[0].strip()
    response_text = response_text.split("<|end_of_text|>")[0].strip()

    # Add reasoning note
    reasoning_note = f"\n\n[Agent Thought: The LLM decided to use '{selected_tool}' tool(s) because: {reasoning}]"
    return response_text + reasoning_note


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🤖 LLAMA-3.2 AGENT WITH WEATHER INTEGRATION")
    print("=" * 60)
    while True:
      user_query = input("You: ").strip()
      if user_query.lower() in ["exit", "quit"]:
          print("👋 Exiting chat. Goodbye!")
          break
      response = agent_respond(user_query, vectorstore)
      print(f"\n🧠 Agent response: {response}")