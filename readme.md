# 🧠 Ollama Chatbot

A conversational AI chatbot built with **Python**, **Gradio**, and **Ollama** — enabling interactive, real-time conversations with a locally running language model.

---

## ✨ Features

- 💬 **Live Chat UI** with Gradio
- 🧠 **LLM-powered responses** via Ollama
- 🔐 Uses `.env` for secure config
- ⚡ **Streaming responses** for faster interaction
- 🔧 **Easy local deployment**

---

## 📂 Project Structure

📁 ollama-chatbot/
├── app.py # Main Python script
├── .env # Environment variables
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## ⚙️ Environment Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/ollama-chatbot.git
cd ollama-chatbot

Model Configuration:

The model used is hardcoded inside app.py. Default is:

model = "llama3.2"
To change the model, ensure it is available locally with:

ollama pull llama3.2
You can replace "llama3.2" with any model you've downloaded.


Create a virtual environment:

python -m venv .venv
source .venv/bin/activate  #Windows: .venv\Scripts\activate

License:

This project is licensed under the MIT License.

Acknowledgements:

ollama
Gradio

Feel free to fork this repo and build something awesome ✨
For feedback or contributions, raise an issue or open a pull request.