#!/usr/bin/env python3
"""
Example script demonstrating how to use different pretrained models
with the NegotiationGPT chat inference system.
"""

from chat_inference import ChatInference


def main():
    """Demonstrate model selection and usage."""

    print("\n" + "="*70)
    print("NegotiationGPT - Pretrained Model Selection Demo")
    print("="*70 + "\n")

    # 1. List all available models
    print("Step 1: Viewing available models\n")
    ChatInference.list_available_models()

    # 2. Example: Using T5 model
    print("\n" + "="*70)
    print("Example 1: Using T5 Model")
    print("="*70 + "\n")

    chat_t5 = ChatInference(model_name="T5")

    test_context = "Buyer: I'm interested in this product, but the price is too high."
    print(f"Context: {test_context}\n")

    result_t5 = chat_t5.predict_next_content(test_context)
    print(f"T5 Response: {result_t5['generated_text']}")
    print(f"Predicted Code: {result_t5['predicted_code']}")

    # 3. Example: Using BERT model
    print("\n" + "="*70)
    print("Example 2: Using BERT Model")
    print("="*70 + "\n")

    chat_bert = ChatInference(model_name="BERT")

    result_bert = chat_bert.predict_next_content(test_context)
    print(f"BERT Response: {result_bert['generated_text']}")
    print(f"Predicted Code: {result_bert['predicted_code']}")

    # 4. Example: Using RoBERTa model
    print("\n" + "="*70)
    print("Example 3: Using RoBERTa Model")
    print("="*70 + "\n")

    chat_roberta = ChatInference(model_name="RoBERTa")

    result_roberta = chat_roberta.predict_next_content(test_context)
    print(f"RoBERTa Response: {result_roberta['generated_text']}")
    print(f"Predicted Code: {result_roberta['predicted_code']}")

    # 5. Example: Using ALBERT model
    print("\n" + "="*70)
    print("Example 4: Using ALBERT Model")
    print("="*70 + "\n")

    chat_albert = ChatInference(model_name="ALBERT")

    result_albert = chat_albert.predict_next_content(test_context)
    print(f"ALBERT Response: {result_albert['generated_text']}")
    print(f"Predicted Code: {result_albert['predicted_code']}")

    # 6. Example: Using DistilBERT model
    print("\n" + "="*70)
    print("Example 5: Using DistilBERT Model")
    print("="*70 + "\n")

    chat_distilbert = ChatInference(model_name="DistilBERT")

    result_distilbert = chat_distilbert.predict_next_content(test_context)
    print(f"DistilBERT Response: {result_distilbert['generated_text']}")
    print(f"Predicted Code: {result_distilbert['predicted_code']}")

    # 7. Example: Using DistilGPT2 model
    print("\n" + "="*70)
    print("Example 6: Using DistilGPT2 Model")
    print("="*70 + "\n")

    chat_distilgpt2 = ChatInference(model_name="DistilGPT2")

    result_distilgpt2 = chat_distilgpt2.predict_next_content(test_context)
    print(f"DistilGPT2 Response: {result_distilgpt2['generated_text']}")
    print(f"Predicted Code: {result_distilgpt2['predicted_code']}")

    print("\n" + "="*70)
    print("Model Selection Demo Complete!")
    print("="*70 + "\n")

    # Usage summary
    print("USAGE SUMMARY:")
    print("-" * 70)
    print("To use a specific model in your code:")
    print("")
    print("  # Option 1: Specify model by name")
    print("  chat = ChatInference(model_name='T5')")
    print("  chat = ChatInference(model_name='BERT')")
    print("  chat = ChatInference(model_name='RoBERTa')")
    print("  chat = ChatInference(model_name='ALBERT')")
    print("  chat = ChatInference(model_name='DistilBERT')")
    print("  chat = ChatInference(model_name='DistilGPT2')")
    print("")
    print("  # Option 2: Use default (T5)")
    print("  chat = ChatInference()")
    print("")
    print("  # Option 3: Use custom model path")
    print("  chat = ChatInference(model_path='path/to/model.pt', vocab_path='path/to/vocab.json')")
    print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
