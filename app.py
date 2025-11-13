import requests
import gradio as gr
import json
import tempfile
import os
from pathlib import Path
import speech_recognition as sr
from gtts import gTTS
import pygame
import io
from datetime import datetime
import logging
from typing import List, Dict, Optional, Tuple, Any
import threading
import queue
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize pygame mixer for audio playback
pygame.mixer.init()

class VoicePersonality:
    """Define different voice personalities"""
    def __init__(self, name: str, lang: str, tld: str, slow: bool, system_prompt: str, description: str):
        self.name = name
        self.lang = lang  # Language code
        self.tld = tld    # Top-level domain for accent
        self.slow = slow  # Speech speed
        self.system_prompt = system_prompt
        self.description = description

# Define available voice personalities
VOICE_PERSONALITIES = {
    "friendly": VoicePersonality(
        name="Friendly Assistant",
        lang="en",
        tld="com",
        slow=False,
        system_prompt="You're a warm, friendly, and enthusiastic assistant. Use casual language, show excitement about helping, and be encouraging. Add some personality to your responses with expressions like 'Great question!', 'I'd love to help!', etc.",
        description="🌟 Warm and enthusiastic helper"
    ),
    "professional": VoicePersonality(
        name="Professional Expert",
        lang="en",
        tld="co.uk",
        slow=False,
        system_prompt="You're a professional, knowledgeable expert. Speak in a clear, authoritative manner. Be concise, accurate, and formal. Focus on providing precise information and well-structured responses.",
        description="💼 Formal and authoritative expert"
    ),
    "casual": VoicePersonality(
        name="Casual Buddy",
        lang="en",
        tld="com.au",
        slow=False,
        system_prompt="You're a laid-back, casual friend. Use informal language, contractions, and speak like you're chatting with a good friend. Be relaxed, use humor when appropriate, and keep things conversational.",
        description="😎 Laid-back conversational friend"
    ),
    "wise": VoicePersonality(
        name="Wise Mentor",
        lang="en",
        tld="co.uk",
        slow=True,
        system_prompt="You're a wise, thoughtful mentor with years of experience. Speak deliberately and thoughtfully. Share insights, ask probing questions, and help users think deeply about topics. Use metaphors and life lessons when relevant.",
        description="🧙‍♂️ Thoughtful and philosophical guide"
    ),
    "energetic": VoicePersonality(
        name="Energetic Coach",
        lang="en",
        tld="us",
        slow=False,
        system_prompt="You're an energetic, motivational coach! Be upbeat, positive, and encouraging. Use exclamation points, motivational language, and help pump up the user. Focus on action, progress, and achievement!",
        description="⚡ High-energy motivational coach"
    ),
    "calm": VoicePersonality(
        name="Calm Therapist",
        lang="en",
        tld="com",
        slow=True,
        system_prompt="You're a calm, soothing presence. Speak gently and thoughtfully. Be patient, understanding, and provide comfort. Use calming language and help users feel at ease. Focus on mindfulness and emotional well-being.",
        description="🧘‍♀️ Peaceful and soothing presence"
    ),
    "scientist": VoicePersonality(
        name="Research Scientist",
        lang="en",
        tld="edu",
        slow=False,
        system_prompt="You're a curious research scientist who loves explaining complex topics. Be precise, use scientific terminology when appropriate, and break down complex ideas into understandable parts. Show enthusiasm for discovery and learning.",
        description="🔬 Analytical and curious researcher"
    ),
    "storyteller": VoicePersonality(
        name="Creative Storyteller",
        lang="en",
        tld="ie",
        slow=False,
        system_prompt="You're a creative storyteller who loves narratives and imagination. Use vivid descriptions, metaphors, and paint pictures with words. Make responses engaging and memorable. Bring creativity to every interaction.",
        description="📚 Imaginative and creative narrator"
    )
}

class ConversationAgent:
    """Agent to manage conversation state, context, and multimodal interactions"""
    
    def __init__(self, model: str = "llama3.2", personality: str = "friendly"):
        self.model = model
        self.current_personality = personality
        self.personality_obj = VOICE_PERSONALITIES.get(personality, VOICE_PERSONALITIES["friendly"])
        self.system_message = self.personality_obj.system_prompt
        self.conversation_history: List[Dict[str, str]] = []
        self.audio_enabled = True
        self.speech_recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize microphone
        with self.microphone as source:
            self.speech_recognizer.adjust_for_ambient_noise(source)
    
    def change_personality(self, personality: str):
        """Change the voice personality"""
        if personality in VOICE_PERSONALITIES:
            self.current_personality = personality
            self.personality_obj = VOICE_PERSONALITIES[personality]
            self.system_message = self.personality_obj.system_prompt
            logger.info(f"Changed personality to: {self.personality_obj.name}")
            return f"Personality changed to: {self.personality_obj.name}"
        else:
            return f"Personality '{personality}' not found"
    
    def get_personality_info(self):
        """Get current personality information"""
        return {
            "name": self.personality_obj.name,
            "description": self.personality_obj.description,
            "current": self.current_personality
        }
    
    def add_to_history(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def build_formatted_history(self) -> List[Dict[str, str]]:
        """Build formatted history for Ollama API"""
        formatted_history = [{"role": "system", "content": self.system_message}]
        
        for msg in self.conversation_history:
            formatted_history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return formatted_history
    
    def speech_to_text(self, audio_file) -> Optional[str]:
        """Convert speech to text using speech recognition"""
        if audio_file is None:
            return None
        
        try:
            # Load audio file
            with sr.AudioFile(audio_file) as source:
                audio_data = self.speech_recognizer.record(source)
            
            # Convert to text
            text = self.speech_recognizer.recognize_google(audio_data)
            logger.info(f"Speech recognized: {text}")
            return text
        
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return "Sorry, I couldn't understand the audio. Please try again."
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            return f"Speech recognition service error: {e}"
        except Exception as e:
            logger.error(f"Error in speech to text: {e}")
            return f"Error processing audio: {e}"
    
    def text_to_speech(self, text: str) -> Optional[str]:
        """Convert text to speech using current personality voice settings"""
        try:
            # Create TTS object with personality-specific settings
            tts = gTTS(
                text=text, 
                lang=self.personality_obj.lang,
                tld=self.personality_obj.tld,
                slow=self.personality_obj.slow
            )
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tts.save(temp_file.name)
            
            logger.info(f"TTS audio saved with {self.personality_obj.name} voice: {temp_file.name}")
            return temp_file.name
        
        except Exception as e:
            logger.error(f"Error in text to speech: {e}")
            return None
    
    def get_ollama_response(self, message: str):
        """Get streaming response from Ollama"""
        # Add user message to history
        self.add_to_history("user", message)
        
        # Build conversation context
        formatted_history = self.build_formatted_history()
        formatted_history.append({"role": "user", "content": message})
        
        try:
            # Get streaming response from Ollama
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.model,
                    "messages": formatted_history,
                    "stream": True
                },
                stream=True,
                timeout=30
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
                        logger.error(f"Error decoding JSON: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing chunk: {e}")
                        continue
            
            # Add assistant response to history
            if reply:
                self.add_to_history("assistant", reply)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Connection error to Ollama server: {e}"
            logger.error(error_msg)
            yield error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(error_msg)
            yield error_msg

# Global agent instance
conversation_agent = ConversationAgent()

def change_voice_personality(personality_key):
    """Change the voice personality"""
    global conversation_agent
    conversation_agent.change_personality(personality_key)
    personality = conversation_agent.personality_obj
    return f"Changed to {personality.name} personality", personality.description

def test_ollama_connection():
    """Test if Ollama server is running"""
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2",
            "prompt": "Hello, who are you?",
            "stream": False
        }, timeout=10)
        
        if response.status_code == 200:
            result = response.json()["response"]
            logger.info(f"Ollama connection successful: {result[:100]}...")
            return True, result
        else:
            logger.error(f"Ollama server returned status code: {response.status_code}")
            return False, f"Server error: {response.status_code}"
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Cannot connect to Ollama server: {e}")
        return False, f"Connection failed: {e}"

def generate_audio_response(history):
    """Generate audio for the last bot response"""
    if not history or not isinstance(history, list):
        return None
    
    # The last message in history is a tuple of (user_message, bot_response)
    last_response = ""
    for user_msg, bot_msg in reversed(history):
        if bot_msg and isinstance(bot_msg, str) and bot_msg.strip():
            last_response = bot_msg
            break
    
    if not last_response.strip():
        return None
    
    try:
        # Generate TTS audio
        audio_file = conversation_agent.text_to_speech(last_response)
        return audio_file
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return None

def clear_conversation():
    """Clear conversation history while maintaining current personality"""
    global conversation_agent
    current_personality = conversation_agent.current_personality
    conversation_agent = ConversationAgent(personality=current_personality)
    return [], "", None

def toggle_audio_mode():
    """Toggle audio response mode"""
    conversation_agent.audio_enabled = not conversation_agent.audio_enabled
    status = "enabled" if conversation_agent.audio_enabled else "disabled"
    return f"Audio responses {status}"

# Test connection on startup
connection_status, connection_message = test_ollama_connection()
if connection_status:
    print(f"✅ Ollama connection successful!")
    print(f"Initial response: {connection_message}")
else:
    print(f"❌ Ollama connection failed: {connection_message}")
    print("Please ensure Ollama is running with: ollama serve")

# === Enhanced Gradio Interface ===
def create_interface():
    """Create the enhanced Gradio interface"""
    
    with gr.Blocks(
        title="Enhanced Multimodal Chat Assistant",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .audio-section { background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 10px 0; }
        .status-box { background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 5px 0; }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🤖 Enhanced Multimodal Chat Assistant
        
        **Features:**
        - 💬 **Text Chat**: Type your messages normally
        - 🎤 **Speech Input**: Record audio messages that get transcribed
        - 🔊 **Text-to-Speech**: Get audio responses from the assistant
        - 🎭 **Voice Personalities**: Choose from 8 different AI personalities with unique voices
        - 🧠 **Agent Management**: Intelligent conversation context handling
        - 📊 **Conversation History**: Maintains context across interactions
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Main chat interface - Fixed for Gradio 3.x (list of tuples)
                chatbot = gr.Chatbot(
                    height=500,
                    label="Conversation",
                    show_label=True,
                    container=True
                )
                
                # Text input
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message here...",
                        scale=4,
                        container=False,
                        show_label=False
                    )
                    send_btn = gr.Button("Send 📤", scale=1, variant="primary")
                
                # Audio input section
                with gr.Group(elem_classes=["audio-section"]):
                    gr.Markdown("### 🎤 Voice Input")
                    with gr.Row():
                        audio_input = gr.Audio(
                            source="microphone",
                            type="filepath",
                            label="Record or Upload Audio"
                        )
                        process_audio_btn = gr.Button("Process Audio 🎵", variant="secondary")
                
                # Audio output section
                with gr.Group(elem_classes=["audio-section"]):
                    gr.Markdown("### 🔊 Audio Response")
                    audio_output = gr.Audio(
                        label="Generated Speech",
                        autoplay=False
                    )
                    generate_audio_btn = gr.Button("Generate Audio Response 🔊", variant="secondary")
            
            with gr.Column(scale=1):
                # Control panel
                gr.Markdown("### ⚙️ Controls")
                
                with gr.Group():
                    # Voice personality selector
                    gr.Markdown("### 🎭 Voice Personality")
                    personality_dropdown = gr.Dropdown(
                        choices=[(f"{p.name} - {p.description}", key) for key, p in VOICE_PERSONALITIES.items()],
                        value="friendly",
                        label="Choose Personality",
                        interactive=True
                    )
                    personality_status = gr.Textbox(
                        label="Current Personality",
                        value=VOICE_PERSONALITIES["friendly"].description,
                        interactive=False
                    )
                    
                    clear_btn = gr.Button("Clear Conversation 🗑️", variant="stop")
                    audio_toggle_btn = gr.Button("Toggle Audio Mode 🔄")
                    
                    # Status display
                    status_display = gr.Textbox(
                        label="Status",
                        value=f"Ready - Ollama: {'✅' if connection_status else '❌'}",
                        interactive=False,
                        elem_classes=["status-box"]
                    )
                
                # Conversation info
                gr.Markdown("### 📊 Session Info")
                session_info = gr.Textbox(
                    label="Session ID",
                    value=conversation_agent.conversation_id,
                    interactive=False
                )
        
        # Event handlers - Fixed to use tuples format for Gradio 3.x
        def handle_text_submit(message, history):
            """Handle text message submission - returns list of tuples"""
            if not message.strip():
                return history or [], ""
            
            # Ensure history is a list
            if history is None:
                history = []
            
            # Get response from agent
            response_generator = conversation_agent.get_ollama_response(message)
            
            # Get the complete response
            complete_response = ""
            try:
                for partial_response in response_generator:
                    complete_response = partial_response
            except Exception as e:
                complete_response = f"Error: {str(e)}"
            
            # Add user message and bot response as a tuple
            history.append((message, complete_response))
            
            return history, ""  # Return updated history and clear input
        
        def handle_audio_submit(audio, history):
            """Handle audio message submission - returns list of tuples"""
            if audio is None:
                return history or [], "No audio file provided"
            
            # Ensure history is a list
            if history is None:
                history = []
            
            try:
                # Convert speech to text
                transcribed_text = conversation_agent.speech_to_text(audio)
                
                if not transcribed_text or transcribed_text.startswith("Sorry") or transcribed_text.startswith("Error"):
                    return history, transcribed_text or "Failed to transcribe audio"
                
                # Get response from agent
                response_generator = conversation_agent.get_ollama_response(transcribed_text)
                
                # Get the complete response
                complete_response = ""
                for partial_response in response_generator:
                    complete_response = partial_response
                
                # Add user message (with microphone emoji) and bot response as a tuple
                history.append((f"🎤 {transcribed_text}", complete_response))
                
                return history, f"Transcribed: {transcribed_text}"
                
            except Exception as e:
                return history, f"Error processing audio: {str(e)}"
        
        # Text input events
        send_btn.click(
            handle_text_submit,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            handle_text_submit,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        # Audio input events
        process_audio_btn.click(
            handle_audio_submit,
            inputs=[audio_input, chatbot],
            outputs=[chatbot, status_display]
        )
        
        # Audio output events
        generate_audio_btn.click(
            generate_audio_response,
            inputs=[chatbot],
            outputs=[audio_output]
        )
        
        # Personality change event
        personality_dropdown.change(
            change_voice_personality,
            inputs=[personality_dropdown],
            outputs=[status_display, personality_status]
        )
        
        # Control events
        clear_btn.click(
            clear_conversation,
            outputs=[chatbot, msg_input, audio_output]
        )
        
        audio_toggle_btn.click(
            toggle_audio_mode,
            outputs=[status_display]
        )
    
    return demo

# === Main Application ===
if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    
    print("\n🚀 Launching Multimodal Chat Assistant...")
    print("📋 Features available:")
    print("  - Text-based conversation")
    print("  - Speech-to-text input")
    print("  - Text-to-speech output")
    print("  - Agent-managed conversations")
    print("  - Conversation history management")
    
    # Launch with appropriate settings
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        debug=True
    )