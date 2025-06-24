# Ollama Chatbot

A conversational AI chatbot built with Python, Gradio, and the Ollama LLM. This project allows users to interact with a large language model through a simple web interface, enabling real-time, natural language conversations.

---

## Features

- **Live Chat UI** with Gradio
- **LLM-powered responses** via Ollama
- Uses `.env` for secure config
- **Streaming responses** for faster interaction
- **Easy local deployment**

---

## Project Structure

📁 ollama-chatbot/<br>
├── app.py <br>
├── .env <br>
├── requirements.txt <br>
└── README.md 


---

## ⚙️ Environment Setup

1. **Clone the repository**<br>
git clone https://github.com/palaknagar-07/Ollama-chatbot.git <br>
cd ollama-chatbot

2. **Install dependencies** <br>
pip install -r requirements.txt

3. **Create a virtual environment**<br>
python -m venv .venv<br>
source .venv/bin/activate  #Windows: .venv\Scripts\activate

## Model Configuration

The model used is hardcoded inside app.py. Default is:<br>
model = "llama3.2"<br>
To change the model, ensure it is available locally with:<br>
ollama pull llama3.2<br>
You can replace "llama3.2" with any model you've downloaded.

## Sample Interaction

You: What is AI?  
Bot: AI stands for Artificial Intelligence...

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Ollama](https://ollama.com/)
- [Gradio](https://gradio.app/)


---
Feel free to fork this repo and build something awesome. <br>
For feedback or contributions, raise an issue or open a pull request.
