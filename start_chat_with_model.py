#!/usr/bin/env python3
"""
Enhanced startup script for NegotiationGPT Chat Interface
Allows model selection before starting the server
"""

import sys
import os
import subprocess
from chat_inference import ChatInference


def select_model():
    """Interactive model selection."""
    print("\n" + "="*70)
    print("NegotiationGPT - Model Selection")
    print("="*70)

    # List available models
    models = list(ChatInference.AVAILABLE_MODELS.keys())

    print("\nAvailable Models:\n")
    for i, (key, info) in enumerate(ChatInference.AVAILABLE_MODELS.items(), 1):
        print(f"{i}. {key:15} - {info['model_description']}")

    print(f"\n{len(models) + 1}. Default (T5)")
    print(f"{len(models) + 2}. Custom Path")

    print("\n" + "="*70)

    # Get user choice
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models) + 2}): ").strip()

            if not choice:
                print("Defaulting to T5...")
                return "T5", None, None

            choice_num = int(choice)

            if 1 <= choice_num <= len(models):
                model_name = models[choice_num - 1]
                print(f"\nSelected: {model_name}")
                return model_name, None, None

            elif choice_num == len(models) + 1:
                print("Using default T5 model...")
                return "T5", None, None

            elif choice_num == len(models) + 2:
                model_path = input("Enter model path: ").strip()
                vocab_path = input("Enter vocab path: ").strip()
                return None, model_path, vocab_path

            else:
                print(f"Invalid choice. Please enter 1-{len(models) + 2}")

        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nCancelled.")
            sys.exit(0)


def start_server(model_name=None, model_path=None, vocab_path=None):
    """Start the chat server with selected model."""

    print("\n" + "="*70)
    print("Starting NegotiationGPT Chat Server")
    print("="*70 + "\n")

    # Set environment variables for the server to use
    if model_name:
        os.environ['NEGOTIATION_MODEL'] = model_name
        print(f"Model: {model_name}")
    elif model_path:
        os.environ['NEGOTIATION_MODEL_PATH'] = model_path
        os.environ['NEGOTIATION_VOCAB_PATH'] = vocab_path or ""
        print(f"Custom Model Path: {model_path}")

    print("Server will be available at: http://localhost:5001")
    print("Press Ctrl+C to stop the server\n")
    print("="*70 + "\n")

    # Start the server
    try:
        subprocess.run([sys.executable, "chat_server.py"])
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        sys.exit(0)


def main():
    """Main function."""
    print("\n" + "="*70)
    print("NegotiationGPT Chat Interface Launcher")
    print("="*70 + "\n")

    # Check if model argument provided
    if len(sys.argv) > 1:
        model_arg = sys.argv[1]

        # Check if it's a valid model name
        if model_arg in ChatInference.AVAILABLE_MODELS:
            print(f"Using model: {model_arg}")
            start_server(model_name=model_arg)
        else:
            print(f"Invalid model: {model_arg}")
            print(f"Available models: {', '.join(ChatInference.AVAILABLE_MODELS.keys())}")
            sys.exit(1)
    else:
        # Interactive selection
        model_name, model_path, vocab_path = select_model()
        start_server(model_name, model_path, vocab_path)


if __name__ == "__main__":
    main()
