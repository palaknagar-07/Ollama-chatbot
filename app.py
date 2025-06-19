import requests
import gradio as gr
import json

# Test if Ollama server is running
response = requests.post("http://localhost:11434/api/generate", json={
    "model": "llama3.2",
    "prompt": "Hello, who are you?",
    "stream": False
})
print("Initial test response:", response.json()["response"])

# === Chat setup ===
system_message = "You're an helpful assistant. If you don't know something just say so"
MODEL = "llama3.2"

def chat(message, history):
    # Build conversation history for Ollama
    formatted_history = [{"role": "system", "content": system_message}]
    
    # Add previous messages
    for user_msg, bot_msg in history:
        formatted_history.append({"role": "user", "content": user_msg})
        formatted_history.append({"role": "assistant", "content": bot_msg})
    
    # Add current message
    formatted_history.append({"role": "user", "content": message})
    
    # Get streaming response
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": MODEL,
            "messages": formatted_history,
            "stream": True
        },
        stream=True
    )
    
    reply = ""
    
    # Process each chunk
    for line in response.iter_lines():
        if line:
            try:
                chunk = json.loads(line.decode("utf-8"))
                
                # Extract message content
                if 'message' in chunk and 'content' in chunk['message']:
                    reply += chunk['message']['content']
                    yield reply
                
                # Check if done
                if chunk.get('done', False):
                    break
                    
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue

# === Gradio Interface ===
if __name__ == "__main__":
    gr.ChatInterface(fn=chat).launch()