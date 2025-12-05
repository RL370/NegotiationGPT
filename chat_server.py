#!/usr/bin/env python3
"""
Flask Server for NegotiationGPT Chat Interface
Provides REST API for chat management and inference
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from typing import Dict, List
import pandas as pd
import io

from chat_inference import ChatInference


app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize chat inference
chat_inference = None

# Storage for chat sessions (in production, use a database)
CHATS_FILE = "chat_sessions.json"


class ChatStorage:
    """Manages storage and retrieval of chat sessions."""

    def __init__(self, storage_file: str = CHATS_FILE):
        self.storage_file = storage_file
        self.chats = self._load_chats()

    def _load_chats(self) -> Dict:
        """Load chats from storage file."""
        if Path(self.storage_file).exists():
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading chats: {e}")
                return {}
        return {}

    def _save_chats(self):
        """Save chats to storage file."""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.chats, f, indent=2)
        except Exception as e:
            print(f"Error saving chats: {e}")

    def create_chat(self, title: str = None) -> Dict:
        """Create a new chat session.

        Args:
            title: Optional title for the chat

        Returns:
            New chat session dictionary
        """
        chat_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        chat = {
            'id': chat_id,
            'title': title or f"Negotiation Chat {len(self.chats) + 1}",
            'created_at': timestamp,
            'updated_at': timestamp,
            'messages': []
        }

        self.chats[chat_id] = chat
        self._save_chats()

        return chat

    def get_chat(self, chat_id: str) -> Dict:
        """Get a chat session by ID.

        Args:
            chat_id: Chat session ID

        Returns:
            Chat session dictionary or None
        """
        return self.chats.get(chat_id)

    def get_all_chats(self) -> List[Dict]:
        """Get all chat sessions.

        Returns:
            List of chat session dictionaries
        """
        # Return sorted by updated_at (most recent first)
        chats = list(self.chats.values())
        chats.sort(key=lambda x: x['updated_at'], reverse=True)
        return chats

    def update_chat(self, chat_id: str, messages: List[Dict] = None, title: str = None):
        """Update a chat session.

        Args:
            chat_id: Chat session ID
            messages: Updated messages list
            title: Updated title
        """
        if chat_id not in self.chats:
            return False

        if messages is not None:
            self.chats[chat_id]['messages'] = messages

        if title is not None:
            self.chats[chat_id]['title'] = title

        self.chats[chat_id]['updated_at'] = datetime.now().isoformat()
        self._save_chats()

        return True

    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat session.

        Args:
            chat_id: Chat session ID

        Returns:
            True if deleted, False otherwise
        """
        if chat_id in self.chats:
            del self.chats[chat_id]
            self._save_chats()
            return True
        return False

    def add_message(self, chat_id: str, message: Dict) -> bool:
        """Add a message to a chat session.

        Args:
            chat_id: Chat session ID
            message: Message dictionary with 'role' and 'content'

        Returns:
            True if added, False otherwise
        """
        if chat_id not in self.chats:
            return False

        message['timestamp'] = datetime.now().isoformat()
        self.chats[chat_id]['messages'].append(message)
        self.chats[chat_id]['updated_at'] = datetime.now().isoformat()
        self._save_chats()

        return True


# Initialize storage
chat_storage = ChatStorage()


@app.route('/')
def index():
    """Serve the main page."""
    return send_from_directory('static', 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': chat_inference is not None
    })


@app.route('/api/chats', methods=['GET'])
def get_chats():
    """Get all chat sessions."""
    chats = chat_storage.get_all_chats()
    return jsonify({'chats': chats})


@app.route('/api/chats', methods=['POST'])
def create_chat():
    """Create a new chat session."""
    data = request.get_json() or {}
    title = data.get('title')

    chat = chat_storage.create_chat(title=title)
    return jsonify({'chat': chat}), 201


@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    """Get a specific chat session."""
    chat = chat_storage.get_chat(chat_id)

    if not chat:
        return jsonify({'error': 'Chat not found'}), 404

    return jsonify({'chat': chat})


@app.route('/api/chats/<chat_id>', methods=['PUT'])
def update_chat(chat_id):
    """Update a chat session."""
    data = request.get_json() or {}

    if not chat_storage.get_chat(chat_id):
        return jsonify({'error': 'Chat not found'}), 404

    title = data.get('title')
    messages = data.get('messages')

    success = chat_storage.update_chat(chat_id, messages=messages, title=title)

    if success:
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Failed to update chat'}), 500


@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Delete a chat session."""
    success = chat_storage.delete_chat(chat_id)

    if success:
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Chat not found'}), 404


@app.route('/api/chats/<chat_id>/messages', methods=['POST'])
def add_message(chat_id):
    """Add a message to a chat and get AI response."""
    data = request.get_json()

    if not data or 'content' not in data:
        return jsonify({'error': 'Missing content'}), 400

    chat = chat_storage.get_chat(chat_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404

    # Check the mode
    mode = data.get('mode', 'practice')
    is_advice_mode = mode == 'advice'
    is_user_only = mode == 'user_only'  # User just adding their response, no AI reply

    # Add user message
    message_role = data.get('role', 'user')
    user_message = {
        'role': message_role,
        'content': data['content']
    }
    chat_storage.add_message(chat_id, user_message)

    # If user_only mode, just return without generating AI response
    if is_user_only:
        chat = chat_storage.get_chat(chat_id)
        return jsonify({'chat': chat})

    # Get AI response if model is loaded
    if chat_inference:
        try:
            # Get updated messages
            chat = chat_storage.get_chat(chat_id)

            # In advice mode, we need the ACTUAL user's role for suggestions
            # The message_role is the speaker (could be counterpart in advice mode)
            # We need to pass the user's actual role for generating suggestions
            actual_user_role = data.get('actual_user_role', message_role)

            # Determine mode for analyze_conversation
            analysis_mode = 'advice' if is_advice_mode else 'practice'

            # Analyze conversation and get prediction, passing actual user's role and mode
            analysis = chat_inference.analyze_conversation(
                chat['messages'],
                actual_user_role,
                mode=analysis_mode
            )

            if is_advice_mode:
                # In advice mode, return multiple suggestions
                ai_response = {
                    'role': 'assistant',
                    'content': 'Multiple response options:',
                    'metadata': {
                        'predicted_code': analysis['next_suggestion']['predicted_code'],
                        'code_description': analysis['next_suggestion']['code_description'],
                        'recommendations': analysis['recommendations'],
                        'stats': analysis['conversation_stats'],
                        'suggestions': analysis.get('multiple_suggestions', []),
                        'is_advice': True
                    }
                }
            else:
                # In practice mode, return single response from counterpart
                ai_response = {
                    'role': 'assistant',
                    'content': analysis['next_suggestion']['generated_text'],
                    'metadata': {
                        'predicted_code': analysis['next_suggestion']['predicted_code'],
                        'code_description': analysis['next_suggestion']['code_description'],
                        'recommendations': analysis['recommendations'],
                        'stats': analysis['conversation_stats']
                    }
                }

            # Add AI message to chat
            chat_storage.add_message(chat_id, ai_response)

            # Return updated chat
            chat = chat_storage.get_chat(chat_id)
            return jsonify({'chat': chat})

        except Exception as e:
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()

            # Add error message
            error_message = {
                'role': 'assistant',
                'content': f"I apologize, but I encountered an error generating a response: {str(e)}",
                'metadata': {'error': True}
            }
            chat_storage.add_message(chat_id, error_message)

            chat = chat_storage.get_chat(chat_id)
            return jsonify({'chat': chat})
    else:
        # Model not loaded, return echo message
        echo_message = {
            'role': 'assistant',
            'content': 'Model not loaded. Please start the server with the model loaded.',
            'metadata': {'error': True}
        }
        chat_storage.add_message(chat_id, echo_message)

        chat = chat_storage.get_chat(chat_id)
        return jsonify({'chat': chat})


@app.route('/api/chats/<chat_id>/respond', methods=['POST'])
def respond_to_suggestion(chat_id):
    """User responds with selected or custom message, get next steps."""
    data = request.get_json()

    if not data or 'content' not in data:
        return jsonify({'error': 'Missing content'}), 400

    chat = chat_storage.get_chat(chat_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404

    # Add user's chosen response
    message_role = data.get('role', 'user')
    user_response = {
        'role': message_role,
        'content': data['content'],
        'is_user_response': True
    }
    chat_storage.add_message(chat_id, user_response)

    # Get AI analysis of next steps
    if chat_inference:
        try:
            chat = chat_storage.get_chat(chat_id)
            # Use actual user role for generating analysis
            actual_user_role = data.get('actual_user_role', message_role)
            # This is part of advice mode flow, so mode is 'practice' to show what counterpart might say
            analysis = chat_inference.analyze_conversation(
                chat['messages'],
                actual_user_role,
                mode='practice'
            )

            # Provide next steps analysis
            next_steps_message = {
                'role': 'assistant',
                'content': '📋 Next Steps Analysis',
                'metadata': {
                    'predicted_code': analysis['next_suggestion']['predicted_code'],
                    'code_description': analysis['next_suggestion']['code_description'],
                    'recommendations': analysis['recommendations'],
                    'stats': analysis['conversation_stats'],
                    'is_next_steps': True
                }
            }

            chat_storage.add_message(chat_id, next_steps_message)

            chat = chat_storage.get_chat(chat_id)
            return jsonify({'chat': chat})

        except Exception as e:
            print(f"Error generating next steps: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Model not loaded'}), 503


@app.route('/api/predict', methods=['POST'])
def predict():
    """Get prediction for given context."""
    data = request.get_json()

    if not data or 'context' not in data:
        return jsonify({'error': 'Missing context'}), 400

    if not chat_inference:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        context = data['context']
        max_tokens = data.get('max_tokens', 80)
        temperature = data.get('temperature', 0.9)
        top_p = data.get('top_p', 0.9)

        result = chat_inference.predict_next_content(
            context,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )

        return jsonify({'result': result})

    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze a conversation."""
    data = request.get_json()

    if not data or 'messages' not in data:
        return jsonify({'error': 'Missing messages'}), 400

    if not chat_inference:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        messages = data['messages']
        analysis = chat_inference.analyze_conversation(messages)

        return jsonify({'analysis': analysis})

    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and analyze a conversation file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not chat_inference:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        # Read file content
        filename = file.filename.lower()

        if filename.endswith('.csv'):
            # Parse CSV file
            df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))
            messages = parse_csv_to_messages(df)

        elif filename.endswith('.txt'):
            # Parse text file
            content = file.stream.read().decode('utf-8')
            messages = parse_text_to_messages(content)

        else:
            return jsonify({'error': 'Unsupported file format. Please upload CSV or TXT file.'}), 400

        if not messages:
            return jsonify({'error': 'No valid messages found in file'}), 400

        # Analyze the conversation
        analysis = chat_inference.analyze_conversation(messages)

        # Generate future tips
        tips = generate_future_tips(messages, analysis)

        return jsonify({
            'success': True,
            'message_count': len(messages),
            'analysis': analysis,
            'future_tips': tips
        })

    except Exception as e:
        print(f"Error processing upload: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to process file: {str(e)}'}), 500


@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available pretrained models."""
    try:
        models = []
        for key, info in ChatInference.AVAILABLE_MODELS.items():
            models.append({
                'id': key,
                'name': key,
                'description': info['model_description']
            })

        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/current', methods=['GET'])
def get_current_model():
    """Get currently selected model."""
    global chat_inference

    if not chat_inference:
        return jsonify({
            'success': True,
            'model': None,
            'message': 'No model loaded'
        })

    return jsonify({
        'success': True,
        'model': {
            'name': chat_inference.model_name,
            'description': ChatInference.AVAILABLE_MODELS.get(
                chat_inference.model_name, {}
            ).get('model_description', 'Custom Model')
        }
    })


@app.route('/api/models/select', methods=['POST'])
def select_model():
    """Switch to a different model."""
    global chat_inference

    data = request.json
    model_name = data.get('model_name')

    if not model_name:
        return jsonify({'error': 'model_name is required'}), 400

    if model_name not in ChatInference.AVAILABLE_MODELS:
        return jsonify({
            'error': f'Invalid model. Available: {", ".join(ChatInference.AVAILABLE_MODELS.keys())}'
        }), 400

    try:
        print(f"\n[Model Switch] Loading {model_name}...")

        # Initialize new model
        new_inference = ChatInference(model_name=model_name)

        # Replace global instance
        chat_inference = new_inference

        print(f"[Model Switch] Successfully switched to {model_name}")

        return jsonify({
            'success': True,
            'message': f'Successfully switched to {model_name}',
            'model': {
                'name': model_name,
                'description': ChatInference.AVAILABLE_MODELS[model_name]['model_description']
            }
        })

    except Exception as e:
        print(f"[Model Switch] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to load model: {str(e)}'}), 500


def parse_csv_to_messages(df: pd.DataFrame) -> List[Dict]:
    """Parse CSV DataFrame to message format."""
    messages = []

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Try to find speaker and content columns
    speaker_col = None
    content_col = None

    for col in df.columns:
        if 'speaker' in col or 'role' in col:
            speaker_col = col
        elif 'content' in col or 'text' in col or 'message' in col:
            content_col = col

    if not speaker_col or not content_col:
        # If no clear columns, use first two columns
        if len(df.columns) >= 2:
            speaker_col = df.columns[0]
            content_col = df.columns[1]
        else:
            return []

    for _, row in df.iterrows():
        speaker = str(row[speaker_col]).strip().lower()
        content = str(row[content_col]).strip()

        if content and content != 'nan':
            messages.append({
                'role': speaker if speaker in ['buyer', 'seller'] else 'user',
                'content': content
            })

    return messages


def parse_text_to_messages(content: str) -> List[Dict]:
    """Parse text file to message format."""
    messages = []
    lines = content.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to parse "Speaker: Message" format
        if ':' in line:
            parts = line.split(':', 1)
            speaker = parts[0].strip().lower()
            message = parts[1].strip()

            messages.append({
                'role': speaker if speaker in ['buyer', 'seller'] else 'user',
                'content': message
            })
        else:
            # If no clear format, treat as user message
            messages.append({
                'role': 'user',
                'content': line
            })

    return messages


def generate_future_tips(messages: List[Dict], analysis: Dict) -> List[str]:
    """Generate tips for future conversations based on analysis."""
    tips = []

    stats = analysis.get('conversation_stats', {})
    predictions = analysis.get('next_suggestion', {})

    # Analyze balance
    buyer_msgs = stats.get('buyer_messages', 0)
    seller_msgs = stats.get('seller_messages', 0)

    if buyer_msgs > seller_msgs * 2:
        tips.append("In future negotiations, allow more space for the other party to speak. Active listening builds trust.")
    elif seller_msgs > buyer_msgs * 2:
        tips.append("You're doing most of the talking. In future negotiations, ask more questions to understand the other party's needs.")

    # Analyze predicted code
    predicted_code = predictions.get('predicted_code', '')

    if predicted_code in ['dis', 'diff']:
        tips.append("This conversation shows disagreement patterns. For future negotiations, try finding common ground early.")
    elif predicted_code in ['agr', 'mu']:
        tips.append("Great! This conversation shows agreement patterns. Maintain this collaborative approach in future negotiations.")
    elif predicted_code.startswith('q'):
        tips.append("Many questions detected. In future negotiations, balance inquiry with information sharing.")
    elif predicted_code.startswith('o'):
        tips.append("Offer patterns detected. In future negotiations, justify your offers with clear reasoning.")

    # General tips based on length
    total_msgs = stats.get('total_messages', 0)
    if total_msgs < 5:
        tips.append("This conversation is brief. In future negotiations, take time to build rapport and understand all perspectives.")
    elif total_msgs > 20:
        tips.append("Long conversation detected. In future negotiations, try to reach agreements more efficiently by clarifying objectives early.")

    # Add strategic tips
    tips.append("Always prepare your BATNA (Best Alternative To Negotiated Agreement) before negotiations.")
    tips.append("Focus on interests, not positions. Ask 'why' to understand underlying needs.")
    tips.append("Look for win-win solutions that create value for both parties.")

    return tips


def initialize_model():
    """Initialize the chat inference model."""
    global chat_inference

    try:
        print("\n" + "="*60)
        print("Initializing NegotiationGPT Chat Server")
        print("="*60 + "\n")

        # Initialize with default model (T5)
        # Users can switch models from the web interface
        print("Initializing with default model (T5)")
        print("You can switch models from the web interface Settings")
        chat_inference = ChatInference(model_name="T5")

        print("\n" + "="*60)
        print("Server Ready!")
        print("="*60 + "\n")

        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        print("Server will start without model. Chat features will be limited.")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # Initialize model
    initialize_model()

    # Start server
    print("Starting Flask server on http://localhost:5001")
    print("Press Ctrl+C to stop the server\n")

    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
