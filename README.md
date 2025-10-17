<p align="center"><h1 align="center">CLI_CHATBOT</h1></p>
<p align="center">
	<img src="https://img.shields.io/github/license/Arush04/CLI_Chatbot?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/Arush04/CLI_Chatbot?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/Arush04/CLI_Chatbot?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/Arush04/CLI_Chatbot?style=default&color=0080ff" alt="repo-language-count">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>

##  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
- [ Architecture](#-architecture)
- [ Trade offs](#-trade-offs-and-limitations)

---

##  Overview

<code>❯ This is a hybrid Retrieval-Augmented Generation (RAG) application that combines document-based question answering with real-time weather insights.</code>

---

##  Features

- **Retrieval-Augmented Generation (RAG)**: Queries are enhanced with context from your ingested documents stored in a Chroma vector DB.
- **Live Weather Integration**: Automatically detects weather-related queries (e.g., "What's the weather like in Delhi?") and fetches real-time data using the OpenWeatherMap API.
- **LLM-Powered Reasoning**: Uses Meta LLaMA 3 to synthesize context from retrieved documents and API responses into clear, factual answers.
- **CLI Chat Interface**: An interactive command-line interface allows users to chat with the agent continuously until they type exit.

---

##  Project Structure

```sh
└── CLI_Chatbot/
    ├── CLI_Agent.ipynb
    ├── dataIngest.py
    ├── main.py
    ├── requiremnts.txt
    └── webscrapper.py
```

---
##  Getting Started

###  Prerequisites

Before getting started with CLI_Chatbot, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python


###  Installation

Install CLI_Chatbot using one of the following methods:

**Build from source:**

1. Clone the CLI_Chatbot repository:
```sh
❯ git clone https://github.com/Arush04/CLI_Chatbot
```

2. Navigate to the project directory:
```sh
❯ cd CLI_Chatbot
```

3. Install the project dependencies:
```sh
❯ pip install -r requirements.txt
```
---

### Usage

1. Create folder which will store our scrapped documents:
```sh
❯ mkdir web_pages
```

2. Run the web scrapper:
```sh
❯ python webscrapper.py
```

3. Ingest web pages into the chroma vector db:
```sh
❯ python dataIngest.py
```

4. Run the agent and infer:
```sh
❯ python main.py
```

**Alternative approach instead of all these steps is to just run the jupyter notebook.**

---

### Architecture

<img width="1192" height="854" alt="Screenshot from 2025-10-16 22-33-18" src="https://github.com/user-attachments/assets/2c5f58e2-f82b-4977-b4dd-15dc8a88111d" />


**Reasoning:**

**LLM: Llama-3.2-3B-Instruct**
- Selected for its efficient resource utilization, since I am running this application on colab this is the ideal fit.
- Provides adequate performance for domain-specific Q&A when properly prompted and paired with good retrieved data.

**Vector Database: ChromaDB**
- Lightweight and easy to implement with minimal configurations, suitable for rapid prototyping.
- Scaling of this application can be easy due to following advantages of Chromadb:
  * Advanced search configurations (MMR, hybrid search, filtering)
  * Custom ingestion pipelines and embedding strategies
  * Flexible chunking methods (semantic, sliding window, recursive)
  * Collection management and persistence options
- Better retrieval quality directly improves LLM output by providing more relevant, focused context, reducing hallucinations and improving answer accuracy without requiring model upgrades

**Tomorrow.io Weather API**

- Offers forecast data, historical weather, and climate insights.
- Free
- Easy to integrate

---

### Trade-offs and Limitations 

1. Basic Preprocessing of Web Data    
The current pipeline for scraping and parsing web pages captures raw textual content without deep cleaning or semantic segmentation.
This means redundant text, navigation links, or unrelated metadata may still exist in the final corpus.
As a result, the retrieved context might occasionally be noisy or less relevant for the user query.

2. Minimal Vector Database Configuration  
The Chroma vector store has been implemented using default parameters, optimized for quick experimentation rather than high performance.
This setup works well for small-scale demos but limits retrieval accuracy and query speed as data volume grows.
Tuning the embedding dimensions, persistence strategy, and distance metrics could substantially improve overall system responsiveness and relevance.

3. Small-scale Models (LLM & Embedding Encoder)  
The application currently uses Meta LLaMA 3.2 3B for generation and all-MiniLM-L6-v2 for embedding.
While these models are lightweight and efficient, they lack the depth, reasoning capability, and domain adaptability of larger models.
As a result, responses may occasionally miss subtle contextual relationships or nuanced domain-specific understanding.

4. Simplistic Prompt Engineering and No Preprocessing Before Inference  
The current system and user prompts are relatively simple and don’t leverage advanced prompt engineering techniques like few-shot examples or contextual formatting.
Moreover, the retrieved data and weather responses are fed directly to the LLM without any preprocessing, ranking, or summarization.
This can increase hallucination risk, as the LLM might attempt to fill in missing context or generate speculative information.

---
