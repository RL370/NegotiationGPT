#!/usr/bin/env python3
"""
Chat Inference Module for NegotiationGPT
Loads trained model and generates responses for negotiation context
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Import model components from main file
from NegotiationGPT import (
    NegotiationGPT,
    NegotiationTokenizer,
    NEGOTIATION_CODES,
    CODE_ID_TO_STR,
    DEVICE,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P
)

# Import research-backed analytics
from negotiation_analytics import (
    IssueTracker,
    ConcessionTracker,
    AnchoringDetector,
    QuestionStrategyAnalyzer,
    FramingGenerator
)


class ChatInference:
    """Handles chat inference with the trained NegotiationGPT model."""

    # Available pretrained models
    # Note: All models use the shared NegotiationGPT trained weights
    # The model folders contain base tokenizers, not compatible model architectures
    AVAILABLE_MODELS = {
        "T5": {
            "name": "T5",
            "vocab_path": "checkpoints/best_vocab.json",
            "model_path": "checkpoints/best_model.pt",
            "tokenizer_path": "T5/best_tokenizer",
            "model_description": "T5 (Text-to-Text Transfer Transformer) - NegotiationGPT"
        },
        "RoBERTa": {
            "name": "RoBERTa",
            "vocab_path": "checkpoints/best_vocab.json",
            "model_path": "checkpoints/best_model.pt",
            "tokenizer_path": "RoBERTa/best_tokenizer",
            "model_description": "RoBERTa (Robustly Optimized BERT) - NegotiationGPT"
        },
        "DistilGPT2": {
            "name": "DistilGPT2",
            "vocab_path": "checkpoints/best_vocab.json",
            "model_path": "checkpoints/best_model.pt",
            "tokenizer_path": "DistilBERTGPT2/best_tokenizer",
            "model_description": "DistilGPT2 (Distilled GPT-2) - NegotiationGPT"
        },
        "DistilBERT": {
            "name": "DistilBERT",
            "vocab_path": "checkpoints/best_vocab.json",
            "model_path": "checkpoints/best_model.pt",
            "tokenizer_path": "DistilBERT/best_tokenizer",
            "model_description": "DistilBERT (Distilled BERT) - NegotiationGPT"
        },
        "BERT": {
            "name": "BERT",
            "vocab_path": "checkpoints/best_vocab.json",
            "model_path": "checkpoints/best_model.pt",
            "tokenizer_path": "BERT/best_tokenizer",
            "model_description": "BERT (Bidirectional Encoder) - NegotiationGPT"
        },
        "ALBERT": {
            "name": "ALBERT",
            "vocab_path": "checkpoints/best_vocab.json",
            "model_path": "checkpoints/best_model.pt",
            "tokenizer_path": "ALBERT/best_tokenizer",
            "model_description": "ALBERT (A Lite BERT) - NegotiationGPT"
        }
    }

    def __init__(self, model_name: str = None, model_path: str = None, vocab_path: str = None):
        """Initialize the chat inference system with a pretrained model.

        Args:
            model_name: Name of the pretrained model to use. Options: T5, RoBERTa, DistilGPT2, DistilBERT, BERT, ALBERT
                       If not specified, defaults to custom model_path/vocab_path or T5
            model_path: Path to a custom trained model checkpoint (optional, overrides model_name)
            vocab_path: Path to a custom vocabulary JSON file (optional, overrides model_name)
        """
        # List available models
        if model_name is None and model_path is None:
            print("\n[ChatInference] Available pretrained models:")
            for i, (key, info) in enumerate(self.AVAILABLE_MODELS.items(), 1):
                print(f"  {i}. {key}: {info['model_description']}")
            print("\nDefaulting to T5...\n")
            model_name = "T5"

        # Use specified pretrained model
        if model_name and model_name in self.AVAILABLE_MODELS:
            model_info = self.AVAILABLE_MODELS[model_name]
            if vocab_path is None:
                vocab_path = model_info["vocab_path"]
            if model_path is None:
                model_path = model_info["model_path"]
            tokenizer_path = model_info["tokenizer_path"]
            self.model_name = model_name
            print(f"[ChatInference] Using pretrained model: {model_info['model_description']}")
        elif model_name and model_name not in self.AVAILABLE_MODELS:
            print(f"[ChatInference] Warning: '{model_name}' not found. Available models: {', '.join(self.AVAILABLE_MODELS.keys())}")
            print(f"[ChatInference] Defaulting to T5...")
            model_name = "T5"
            model_info = self.AVAILABLE_MODELS[model_name]
            vocab_path = model_info["vocab_path"]
            model_path = model_info["model_path"]
            tokenizer_path = model_info["tokenizer_path"]
            self.model_name = model_name
        else:
            # Use custom paths
            if model_path is None:
                model_path = "checkpoints/best_model.pt"
            if vocab_path is None:
                vocab_path = "checkpoints/best_vocab.json"
            self.model_name = "Custom"
            print(f"[ChatInference] Using custom model paths")

        print(f"[ChatInference] Loading vocab from {vocab_path}")

        # Load tokenizer
        self.tokenizer = self._load_tokenizer(vocab_path)

        # Load model (with fail-safe)
        self.model = None
        self.model_loaded = False

        try:
            self.model = NegotiationGPT(vocab_size=self.tokenizer.vocab_size)

            if model_path and Path(model_path).exists():
                print(f"[ChatInference] Loading model weights from {model_path}")

                try:
                    # Handle different model file formats
                    if model_path.endswith('.safetensors'):
                        try:
                            from safetensors.torch import load_file
                            state_dict = load_file(model_path, device=str(DEVICE))
                            self.model.load_state_dict(state_dict, strict=False)
                        except ImportError:
                            print("[ChatInference] Warning: safetensors not installed")
                            print("[ChatInference] Using initialized model")
                    else:
                        # Load .pt or .pth files
                        state_dict = torch.load(model_path, map_location=DEVICE)
                        self.model.load_state_dict(state_dict, strict=False)

                    self.model_loaded = True
                    print(f"[ChatInference] Model weights loaded successfully")

                except Exception as e:
                    print(f"[ChatInference] Warning: Could not load model weights: {e}")
                    print(f"[ChatInference] Continuing with initialized model")
            else:
                print(f"[ChatInference] No model weights file found at: {model_path}")
                print(f"[ChatInference] Using initialized model")

            self.model.to(DEVICE)
            self.model.eval()

        except Exception as e:
            print(f"[ChatInference] ERROR initializing model: {e}")
            print(f"[ChatInference] System will continue with fallback responses")
            self.model = None

        print(f"[ChatInference] Model loaded successfully on {DEVICE}")
        print(f"[ChatInference] Vocab size: {self.tokenizer.vocab_size}")
        print(f"[ChatInference] Model: {self.model_name}")

        # Initialize analytics tools
        self.issue_tracker = IssueTracker()
        self.concession_tracker = ConcessionTracker()
        self.anchoring_detector = AnchoringDetector()
        self.question_analyzer = QuestionStrategyAnalyzer()
        self.framing_generator = FramingGenerator()

        print(f"[ChatInference] Analytics tools initialized")

    @staticmethod
    def list_available_models():
        """List all available pretrained models."""
        print("\n" + "="*60)
        print("Available Pretrained Models")
        print("="*60)
        for i, (key, info) in enumerate(ChatInference.AVAILABLE_MODELS.items(), 1):
            print(f"\n{i}. {key}")
            print(f"   Description: {info['model_description']}")
            print(f"   Vocab: {info['vocab_path']}")
            print(f"   Tokenizer: {info['tokenizer_path']}")
        print("\n" + "="*60 + "\n")

    def _load_tokenizer(self, vocab_path: str) -> NegotiationTokenizer:
        """Load tokenizer from vocabulary JSON."""
        tokenizer = NegotiationTokenizer()

        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)

        tokenizer.itos = {int(k): v for k, v in vocab_data['itos'].items()}
        tokenizer.stoi = vocab_data['stoi']

        return tokenizer

    def predict_next_content(
        self,
        context: str,
        max_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        use_high_quality: bool = True,
        speaker_role: str = 'user',
        messages: List[Dict[str, str]] = None
    ) -> Dict[str, any]:
        """Predict next negotiation content based on context.

        Args:
            context: The negotiation context/history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            use_high_quality: Use high-quality template responses instead of raw generation
            speaker_role: Role of the speaker who will say this response
            messages: Full message history for better context

        Returns:
            Dictionary containing generated text and predicted code
        """
        # Fail-safe: Return fallback response if model not loaded
        if self.model is None:
            return self._get_fallback_response(context)

        try:
            with torch.no_grad():
                # Encode input
                input_ids = self.tokenizer.encode(context, add_special_tokens=True)
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
                attention_mask = torch.ones_like(input_tensor)

                # Get initial predictions
                _, code_logits, _ = self.model(input_tensor, attention_mask)

                # Predict negotiation code
                code_pred = torch.argmax(code_logits, dim=1).item()
                predicted_code = CODE_ID_TO_STR.get(code_pred, "unknown")

                # Generate high-quality contextual response
                if use_high_quality:
                    last_message = self._extract_last_message(context)
                    negotiation_stage = self._determine_negotiation_stage(messages) if messages else 'middle'

                    # Use contextual generator for much better quality
                    import random
                    response_type = random.choice(['direct', 'collaborative', 'inquisitive'])

                    if response_type == 'direct':
                        generated_text = self._generate_contextual_direct_response(
                            predicted_code, last_message, context, speaker_role, negotiation_stage
                        )
                    elif response_type == 'collaborative':
                        generated_text = self._generate_contextual_collaborative_response(
                            predicted_code, last_message, speaker_role, negotiation_stage
                        )
                    else:  # inquisitive
                        generated_text = self._generate_contextual_question_response(
                            predicted_code, last_message, speaker_role, negotiation_stage
                        )
                else:
                    # Generate text using beam search (legacy method)
                    generated_text = self._generate_with_beam_search(
                        input_tensor,
                        max_tokens=20,
                        beam_size=5
                    )

                return {
                    'generated_text': generated_text,
                    'predicted_code': predicted_code,
                    'code_description': self._get_code_description(predicted_code)
                }
        except Exception as e:
            print(f"[ChatInference] Error during prediction: {e}")
            return self._get_fallback_response(context)

    def _generate_text(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> str:
        """Generate text using the language model head with improved decoding.

        Args:
            input_ids: Input token IDs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            Generated text string
        """
        generated = input_ids.clone()

        # Track generated tokens to avoid repetition
        generated_tokens = []

        for step in range(max_tokens):
            # Get model predictions
            attention_mask = torch.ones_like(generated)
            lm_logits, _, _ = self.model(generated, attention_mask)

            # Get logits for next token (last position)
            next_token_logits = lm_logits[0, -1, :].clone()

            # Apply repetition penalty to avoid loops
            if len(generated_tokens) > 0:
                for prev_token in set(generated_tokens[-10:]):  # Last 10 tokens
                    next_token_logits[prev_token] *= 0.8  # Penalty factor

            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Convert to probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # Apply top-p (nucleus) filtering
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            # Zero out probabilities for removed tokens
            probs[sorted_indices[sorted_indices_to_remove]] = 0

            # Renormalize
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                # Fallback: use original distribution
                probs = F.softmax(next_token_logits, dim=-1)

            # Sample next token
            if temperature == 0:
                # Greedy decoding
                next_token = torch.argmax(probs).unsqueeze(0)
            else:
                # Sample from distribution
                next_token = torch.multinomial(probs, num_samples=1)

            # Stop conditions
            next_token_id = next_token.item()

            if next_token_id == self.tokenizer.eos_token_id:
                break
            if next_token_id == self.tokenizer.pad_token_id:
                break

            # Add to generated tokens
            generated_tokens.append(next_token_id)

            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            # Stop if sequence gets too long
            if generated.size(1) > 256:
                break

            # Early stopping if we see sentence end punctuation
            decoded_so_far = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            if decoded_so_far.strip() in ['.', '!', '?'] and len(generated_tokens) > 5:
                break

        # Decode generated text (excluding the input)
        generated_ids = generated[0, input_ids.size(1):].tolist()
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Clean up the text
        generated_text = self._clean_generated_text(generated_text)

        return generated_text

    def _generate_with_beam_search(self, input_ids: torch.Tensor, max_tokens: int, beam_size: int = 5) -> str:
        """Generate text using beam search for better quality.

        Args:
            input_ids: Input token IDs
            max_tokens: Maximum tokens to generate
            beam_size: Number of beams for beam search

        Returns:
            Generated text string
        """
        batch_size = input_ids.size(0)
        vocab_size = self.tokenizer.vocab_size

        # Start with the input sequence
        sequences = [(input_ids, 0.0)]  # (sequence, score)

        for step in range(max_tokens):
            all_candidates = []

            for seq, score in sequences:
                # Skip if sequence has ended
                if seq[0, -1].item() == self.tokenizer.eos_token_id:
                    all_candidates.append((seq, score))
                    continue

                # Get model predictions
                attention_mask = torch.ones_like(seq)
                lm_logits, _, _ = self.model(seq, attention_mask)

                # Get logits for next token
                next_token_logits = lm_logits[0, -1, :]

                # Apply repetition penalty
                for prev_token_id in seq[0, input_ids.size(1):].tolist():
                    if prev_token_id in [self.tokenizer.pad_token_id, self.tokenizer.bos_token_id]:
                        continue
                    next_token_logits[prev_token_id] *= 0.7  # Strong penalty

                # Get log probabilities
                log_probs = F.log_softmax(next_token_logits, dim=-1)

                # Get top-k candidates
                top_log_probs, top_indices = torch.topk(log_probs, beam_size)

                for i in range(beam_size):
                    token_id = top_indices[i].item()
                    token_score = top_log_probs[i].item()

                    # Skip pad and unknown tokens
                    if token_id in [self.tokenizer.pad_token_id, self.tokenizer.unk_token_id]:
                        continue

                    new_seq = torch.cat([seq, torch.tensor([[token_id]], device=seq.device)], dim=1)
                    new_score = score + token_score

                    all_candidates.append((new_seq, new_score))

            # Select top beam_size sequences with diversity penalty
            # Penalize candidates that are too similar to already selected beams
            selected_sequences = []
            for seq, score in sorted(all_candidates, key=lambda x: x[1] / (x[0].size(1) - input_ids.size(1) + 1), reverse=True):
                # Diversity penalty: penalize if last tokens match existing beams
                diversity_penalty = 0.0
                last_tokens = seq[0, -min(3, seq.size(1)):].tolist()  # Last 3 tokens

                for selected_seq, _ in selected_sequences:
                    selected_last = selected_seq[0, -min(3, selected_seq.size(1)):].tolist()
                    # Count matching tokens
                    matches = sum(1 for a, b in zip(last_tokens, selected_last) if a == b)
                    diversity_penalty += matches * 0.5  # Penalty per matching token

                adjusted_score = score - diversity_penalty
                selected_sequences.append((seq, adjusted_score))

                if len(selected_sequences) >= beam_size:
                    break

            # Re-sort by adjusted scores
            sequences = sorted(selected_sequences, key=lambda x: x[1], reverse=True)[:beam_size]

            # Check if all beams have ended
            if all(seq[0, -1].item() == self.tokenizer.eos_token_id for seq, _ in sequences):
                break

        # Get the best sequence
        best_seq, _ = sequences[0]

        # Decode the generated part only
        generated_ids = best_seq[0, input_ids.size(1):].tolist()

        # Remove EOS token if present
        if generated_ids and generated_ids[-1] == self.tokenizer.eos_token_id:
            generated_ids = generated_ids[:-1]

        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Clean and validate the text
        generated_text = self._clean_and_validate_text(generated_text)

        return generated_text

    def _clean_and_validate_text(self, text: str) -> str:
        """Clean and validate generated text with quality filtering.

        Args:
            text: Raw generated text

        Returns:
            Cleaned text
        """
        # Remove extra spaces
        text = ' '.join(text.split())

        # Quality filters - reject low-quality outputs
        words = text.split()

        # Filter 1: Minimum length
        if len(words) < 2:
            return "I understand. Let's discuss this further."

        # Filter 2: Reject if too many repeated words
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1

        max_repetitions = max(word_counts.values()) if word_counts else 0
        if max_repetitions > len(words) * 0.3:  # More than 30% repetition
            return "Let me rephrase that. I think we should consider alternative approaches."

        # Filter 3: Reject if mostly punctuation or symbols
        alphanumeric_count = sum(1 for c in text if c.isalnum() or c.isspace())
        if alphanumeric_count < len(text) * 0.7:  # Less than 70% alphanumeric
            return "I'd like to clarify my position on this matter."

        # Filter 4: Check for common nonsensical patterns
        text_lower = text.lower()
        nonsensical_patterns = [
            ('the the', 'of of', 'to to', 'a a', 'and and'),  # Word repetition
            ('like like', 'kind kind', 'sort sort'),
            ('um um', 'uh uh', 'er er')
        ]
        for patterns in nonsensical_patterns:
            if any(pattern in text_lower for pattern in patterns):
                return "Let me clarify my thoughts on this."

        # Filter 5: Ensure it has verb-like structure (basic coherence check)
        # Check if there's at least one common verb pattern
        common_verbs = ['can', 'will', 'would', 'should', 'could', 'is', 'are', 'have', 'need', 'want', 'think', 'believe', 'offer', 'propose']
        has_verb = any(verb in text_lower.split() for verb in common_verbs)
        if not has_verb and len(words) > 3:
            return "I'd like to offer a perspective on this."

        # Capitalize first letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        # Ensure sentence ends properly
        if text and text[-1] not in '.!?':
            text = text + '.'

        # Remove any remaining artifacts
        text = text.replace('  ', ' ').strip()

        return text

    def _generate_template_response(self, predicted_code: str, context: str) -> str:
        """Generate grammatically correct response based on predicted code.

        Args:
            predicted_code: The predicted negotiation code
            context: The conversation context

        Returns:
            Grammatically correct response text
        """
        # Templates organized by negotiation code
        templates = {
            'agr': [
                "I agree with your proposal. Let's move forward with this.",
                "That sounds reasonable to me. I accept these terms.",
                "I'm on board with this approach. Let's proceed."
            ],
            'dis': [
                "I respectfully disagree with that point. Here's my concern:",
                "I see things differently. Let me explain my perspective.",
                "I'm not comfortable with that proposal. Can we explore alternatives?"
            ],
            'os': [
                "I'd like to make an offer: I can provide this at a fair price.",
                "Here's what I'm proposing: Let's set the terms at a mutually beneficial level.",
                "My initial offer is competitive and reasonable for both parties."
            ],
            'om': [
                "Let me counter with a modified proposal that addresses both our needs.",
                "I'd like to adjust the terms slightly to make this work better.",
                "How about we revise the offer to find middle ground?"
            ],
            'ip': [
                "What's your position on this matter?",
                "Can you clarify your stance on the main points?",
                "Where do you stand on the key issues we're discussing?"
            ],
            'ir': [
                "Can you explain your reasoning behind that proposal?",
                "What factors led you to this position?",
                "I'd like to understand the rationale for your offer."
            ],
            'ib': [
                "Could you provide some background on this situation?",
                "What's the context behind this negotiation?",
                "Can you share more details about the circumstances?"
            ],
            'sb': [
                "Let me provide some context: The situation requires careful consideration.",
                "Here's the background: This has been an ongoing discussion.",
                "To give you perspective: There are several factors at play here."
            ],
            'sf': [
                "I feel that this approach would work best for everyone involved.",
                "My concern is that we reach an agreement that feels fair to both parties.",
                "I believe we can find a solution that satisfies both our interests."
            ],
            'qo': [
                "Can you clarify the details of your offer?",
                "What exactly are you proposing in terms of the agreement?",
                "Could you break down the specifics of what you're suggesting?"
            ],
            'qp': [
                "What's your position on the main terms?",
                "How do you view the current proposal?",
                "Where do you stand on the key aspects of this deal?"
            ],
            'qb': [
                "Can you provide more information about the background?",
                "What's the history behind this situation?",
                "Could you explain the context leading up to this?"
            ],
            'in': [
                "Let me share some relevant information: This is important to consider.",
                "Here are the key facts: We need to account for these details.",
                "For your information: These points are crucial to our discussion."
            ],
            'mu': [
                "I understand your perspective. Let's work together to find common ground.",
                "We're on the same page about the main objectives.",
                "I see where you're coming from. Let's build on this understanding."
            ],
            'cs': [
                "To summarize, we've agreed on the following terms.",
                "Let's finalize this agreement with the points we've discussed.",
                "In conclusion, I believe we have a solid foundation for moving forward."
            ]
        }

        # Get templates for the predicted code
        code_templates = templates.get(predicted_code, [
            "Thank you for sharing. Let's continue our discussion.",
            "I appreciate your input. What are your thoughts on next steps?",
            "Let's explore this further and find a mutually beneficial solution."
        ])

        # Select a template (use modulo for variety)
        import random
        template = random.choice(code_templates)

        return template

    def _get_code_description(self, code: str) -> str:
        """Get a human-readable description of the negotiation code.

        Args:
            code: Negotiation code string

        Returns:
            Description of the code
        """
        code_descriptions = {
            "agr": "Agreement - Expressing agreement or acceptance",
            "dis": "Disagreement - Expressing disagreement or rejection",
            "coer": "Coercion - Using pressure or threats",
            "diff": "Differentiation - Highlighting differences",
            "sim": "Similarity - Highlighting similarities",
            "int_proc": "Procedural - Discussing process or procedure",
            "misc": "Miscellaneous communication",
            "os": "Offer Simple - Making a simple offer",
            "om": "Offer Modified - Making a modified offer",
            "ip": "Inquiry Position - Asking about position",
            "ir": "Inquiry Reasoning - Asking about reasoning",
            "ib": "Inquiry Background - Asking about background",
            "sb": "Statement Background - Stating background information",
            "sf": "Statement Feeling - Expressing feelings",
            "qo": "Question Offer - Questioning an offer",
            "qp": "Question Position - Questioning a position",
            "qr": "Question Reasoning - Questioning reasoning",
            "qb": "Question Background - Questioning background",
            "qd": "Question Demand - Questioning a demand",
            "qm": "Question Miscellaneous",
            "in": "Information - Providing information",
            "mu": "Mutual Understanding - Establishing mutual understanding",
            "p1": "Priority 1 - High priority item",
            "pm": "Priority Medium - Medium priority item",
            "pt": "Priority Total - Total priority consideration",
            "cs": "Closing Statement - Making a closing statement",
            "misc_2": "Miscellaneous secondary communication"
        }
        return code_descriptions.get(code, "Unknown negotiation code")

    def _get_fallback_response(self, context: str) -> Dict[str, any]:
        """Generate fallback response when model is not available.

        Args:
            context: The negotiation context

        Returns:
            Dictionary with fallback response
        """
        import random

        # Extract keywords from context for context-aware responses
        context_lower = context.lower()

        # Generic but helpful negotiation responses
        fallback_responses = [
            "I understand your position. Could you elaborate on your priorities?",
            "That's an interesting point. What factors are most important to you in this negotiation?",
            "I appreciate you sharing that. Let's explore options that could work for both parties.",
            "Thank you for that perspective. What would make this a successful outcome for you?",
            "I hear your concerns. Perhaps we can find a middle ground that addresses both our needs.",
            "That's worth considering. What alternatives might you suggest?",
            "I'd like to better understand your constraints. Can you provide more context?",
            "Let's discuss this further. What aspects are most flexible for you?",
            "I value finding a mutually beneficial solution. What's your ideal scenario?",
            "Good point. How do you see us moving forward from here?",
        ]

        # Context-aware additions
        if "price" in context_lower or "$" in context or "cost" in context_lower:
            fallback_responses.extend([
                "I understand the pricing is a concern. Let's see if we can find a price point that works.",
                "The cost is an important factor. What budget range are you working with?",
                "Regarding the price, perhaps we could discuss payment terms or package options.",
            ])

        if "time" in context_lower or "deadline" in context_lower or "schedule" in context_lower:
            fallback_responses.extend([
                "Timing is important. What's your ideal timeline for this?",
                "I understand the schedule matters. Let's see if we can accommodate your timeframe.",
                "The deadline is noted. Can you provide some flexibility if needed?",
            ])

        if "quality" in context_lower or "standard" in context_lower:
            fallback_responses.extend([
                "Quality is essential. Let me assure you we can meet your standards.",
                "I understand the importance of quality. What specific requirements do you have?",
            ])

        response = random.choice(fallback_responses)

        return {
            'generated_text': response,
            'predicted_code': 'q',  # Question code as safe default
            'code_description': 'Question - Asking for information',
            'note': 'Using fallback response (model not fully initialized)'
        }

    def analyze_conversation(self, messages: List[Dict[str, str]], user_role: str = 'user', mode: str = 'advice') -> Dict[str, any]:
        """Analyze a conversation history and provide insights.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            user_role: The role of the user (buyer/seller/user)
            mode: 'advice' or 'practice' - determines who should speak next

        Returns:
            Analysis results with suggestions and research-backed analytics
        """
        # Combine messages into context
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        # Determine who should speak next
        if mode == 'practice':
            # In practice mode, AI responds as the counterpart
            if user_role.lower() == 'buyer':
                next_speaker_role = 'seller'
            elif user_role.lower() == 'seller':
                next_speaker_role = 'buyer'
            else:
                # Alternate based on last speaker
                last_role = messages[-1]['role'] if messages else 'user'
                next_speaker_role = 'buyer' if last_role == 'seller' else 'seller'
        else:
            # In advice mode, user speaks next (suggestions are for them)
            next_speaker_role = user_role

        # Get prediction for next move with high-quality generation
        prediction = self.predict_next_content(
            context,
            use_high_quality=True,
            speaker_role=next_speaker_role,
            messages=messages
        )

        # Determine negotiation stage
        negotiation_stage = self._determine_negotiation_stage(messages)

        # Run research-backed analytics
        issues_analysis = self.issue_tracker.extract_issues(messages)
        concessions_analysis = self.concession_tracker.analyze_concessions(messages)
        anchoring_analysis = self.anchoring_detector.detect_anchors(messages)
        questions_analysis = self.question_analyzer.analyze_questions(messages)

        # Generate multiple suggestions based on user's role
        multiple_suggestions = self._generate_multiple_suggestions(
            context,
            prediction['predicted_code'],
            user_role,
            negotiation_stage,
            messages
        )

        # Analyze conversation patterns
        buyer_count = sum(1 for msg in messages if 'buyer' in msg['role'].lower())
        seller_count = sum(1 for msg in messages if 'seller' in msg['role'].lower())

        return {
            'next_suggestion': prediction,
            'multiple_suggestions': multiple_suggestions,
            'conversation_stats': {
                'total_messages': len(messages),
                'buyer_messages': buyer_count,
                'seller_messages': seller_count,
                'last_speaker': messages[-1]['role'] if messages else 'none',
                'negotiation_stage': negotiation_stage
            },
            'recommendations': self._generate_recommendations(prediction, messages, user_role),
            # Research-backed analytics
            'analytics': {
                'issues': {
                    'identified_issues': issues_analysis,
                    'tradeoff_opportunities': self.issue_tracker.suggest_tradeoffs()
                },
                'concessions': concessions_analysis,
                'anchoring': anchoring_analysis,
                'questions': {
                    'analysis': questions_analysis,
                    'suggested_questions': questions_analysis.get('suggested_questions', [])
                }
            }
        }

    def _determine_negotiation_stage(self, messages: List[Dict[str, str]]) -> str:
        """Determine what stage of negotiation we're in.

        Args:
            messages: List of message dictionaries

        Returns:
            Stage: 'opening', 'middle', 'closing'
        """
        num_messages = len(messages)

        if num_messages <= 3:
            return 'opening'
        elif num_messages <= 8:
            return 'middle'
        else:
            # Check for closing signals in recent messages
            recent_messages = messages[-3:]
            closing_keywords = ['agree', 'deal', 'accept', 'finalize', 'confirm', 'settled']
            for msg in recent_messages:
                content_lower = msg['content'].lower()
                if any(keyword in content_lower for keyword in closing_keywords):
                    return 'closing'
            return 'middle'

    def _generate_multiple_suggestions(
        self,
        context: str,
        predicted_code: str,
        user_role: str,
        negotiation_stage: str,
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Generate multiple response suggestions tailored to the conversation and user's role.

        Args:
            context: Conversation context
            predicted_code: Predicted negotiation code
            user_role: The role of the user (buyer/seller/user)
            negotiation_stage: Current stage of negotiation
            messages: Full conversation history

        Returns:
            List of suggestion dictionaries
        """
        # Extract the last message to understand what was said
        last_message = self._extract_last_message(context)

        # Get different response options based on the code, context, and USER'S ROLE
        suggestions = []

        # Generate reasoning based on predicted code and negotiation stage
        code_reasoning = self._get_code_reasoning(predicted_code, negotiation_stage)

        # Strategy 1: Direct/Assertive
        suggestions.append({
            'style': 'Direct',
            'text': self._generate_contextual_direct_response(
                predicted_code, last_message, context, user_role, negotiation_stage
            ),
            'description': 'Clear and assertive approach',
            'why': f"Direct approach recommended because {code_reasoning['direct']} This style works well in {negotiation_stage} stage when you need clarity and confidence."
        })

        # Strategy 2: Collaborative
        suggestions.append({
            'style': 'Collaborative',
            'text': self._generate_contextual_collaborative_response(
                predicted_code, last_message, user_role, negotiation_stage
            ),
            'description': 'Focus on mutual benefit',
            'why': f"Collaborative approach recommended because {code_reasoning['collaborative']} This builds trust and long-term relationships, especially important in {negotiation_stage} stage."
        })

        # Strategy 3: Inquisitive
        suggestions.append({
            'style': 'Inquisitive',
            'text': self._generate_contextual_question_response(
                predicted_code, last_message, user_role, negotiation_stage
            ),
            'description': 'Gather more information',
            'why': f"Inquisitive approach recommended because {code_reasoning['inquisitive']} Asking questions helps you understand their needs and constraints before committing."
        })

        # Add framing variations to the first suggestion (Direct)
        # Research: Kahneman & Tversky (1979) - Framing effects
        direct_suggestion = suggestions[0]
        framed_versions = self.framing_generator.generate_framed_versions(
            direct_suggestion['text'],
            {'negotiation_stage': negotiation_stage, 'user_role': user_role}
        )

        # Add the top 2 most relevant framings as additional suggestions
        # Gain frame for risk-averse situations
        suggestions.append({
            'style': 'Gain-Framed',
            'text': framed_versions[0]['text'],  # gain frame
            'description': framed_versions[0]['description'],
            'why': f"{framed_versions[0]['when_to_use']}. {framed_versions[0]['research']}"
        })

        # Relationship frame for collaborative context
        if negotiation_stage in ['opening', 'middle']:
            suggestions.append({
                'style': 'Relationship-Focused',
                'text': framed_versions[4]['text'],  # relationship frame
                'description': framed_versions[4]['description'],
                'why': f"{framed_versions[4]['when_to_use']}. {framed_versions[4]['research']}"
            })

        return suggestions

    def _get_code_reasoning(self, code: str, negotiation_stage: str) -> Dict[str, str]:
        """Get reasoning for each suggestion style based on negotiation code.

        Args:
            code: Predicted negotiation code
            negotiation_stage: Current stage of negotiation

        Returns:
            Dictionary with reasoning for each style
        """
        # Code-specific reasoning
        code_reasoning_map = {
            'agr': {
                'direct': "the other party is showing agreement, so you can confidently move forward.",
                'collaborative': "agreement creates momentum, and collaboration ensures both parties stay aligned.",
                'inquisitive': "you should clarify the details of the agreement before finalizing."
            },
            'dis': {
                'direct': "there's disagreement, and you need to clearly state your position.",
                'collaborative': "disagreement requires finding common ground and win-win solutions.",
                'inquisitive': "you need to understand the root causes of their concerns."
            },
            'os': {
                'direct': "an offer is being made, and you should respond with clarity about your position.",
                'collaborative': "offers present opportunities to build mutual value.",
                'inquisitive': "you should understand the full details and rationale behind their offer."
            },
            'om': {
                'direct': "a modified offer shows flexibility, so you can be assertive about what you need.",
                'collaborative': "modified offers signal willingness to compromise, making collaboration more effective.",
                'inquisitive': "you should explore what drove the modification and what else is negotiable."
            },
            'ip': {
                'direct': "they're inquiring about your position, so you should state it clearly.",
                'collaborative': "sharing your position collaboratively builds transparency and trust.",
                'inquisitive': "before revealing your position, you can learn more about theirs."
            },
            'ir': {
                'direct': "they want to understand your reasoning, so be clear and confident.",
                'collaborative': "explaining reasoning collaboratively shows respect for their perspective.",
                'inquisitive': "you can explore their reasoning first to tailor your response."
            },
            'ib': {
                'direct': "they're asking for background, so provide it concisely and confidently.",
                'collaborative': "sharing background collaboratively creates mutual understanding.",
                'inquisitive': "you can gather their background first to provide relevant context."
            },
            'qo': {
                'direct': "they're questioning your offer, so defend it with confidence.",
                'collaborative': "questions about offers are opportunities to find mutual value.",
                'inquisitive': "you should understand their concerns before adjusting your offer."
            },
            'qp': {
                'direct': "they're questioning your position, so reaffirm it clearly.",
                'collaborative': "position questions are chances to find alignment.",
                'inquisitive': "you should explore their position more before responding."
            },
            'mu': {
                'direct': "mutual understanding exists, so you can be direct about next steps.",
                'collaborative': "mutual understanding is perfect for collaborative problem-solving.",
                'inquisitive': "you should confirm the details of your shared understanding."
            },
            'cs': {
                'direct': "closing signals require clear commitment and action.",
                'collaborative': "closing collaboratively ensures both parties are satisfied.",
                'inquisitive': "you should verify all details before closing."
            }
        }

        # Default reasoning if code not found
        default_reasoning = {
            'direct': "clear communication prevents misunderstandings.",
            'collaborative': "collaboration builds better long-term outcomes.",
            'inquisitive': "gathering information gives you strategic advantage."
        }

        return code_reasoning_map.get(code, default_reasoning)

    def _extract_last_message(self, context: str) -> str:
        """Extract the last message from context.

        Args:
            context: Full conversation context

        Returns:
            Last message content
        """
        lines = context.strip().split('\n')
        if not lines:
            return ""

        # Get the last line and extract just the message content
        last_line = lines[-1]
        if ':' in last_line:
            return last_line.split(':', 1)[1].strip()
        return last_line.strip()

    def _analyze_message_sentiment(self, message: str) -> Dict[str, any]:
        """Analyze the sentiment and intent of a message with comprehensive emotion detection.

        Args:
            message: The message text to analyze

        Returns:
            Dictionary with sentiment analysis including primary and secondary emotions
        """
        message_lower = message.lower()

        # Strong disagreement indicators
        strong_disagree = [
            "don't think", "can't accept", "won't work", "not the deal",
            "no way", "absolutely not", "can't do", "won't do", "reject",
            "unacceptable", "too high", "too low", "too much", "too little",
            "not interested", "not going to", "impossible", "can't agree"
        ]

        # Disagreement indicators
        disagree = [
            "don't", "can't", "won't", "not", "no", "disagree", "but",
            "however", "unfortunately", "issue", "difficult", "challenging", "cannot"
        ]

        # Agreement indicators
        agree = [
            "agree", "yes", "accept", "deal", "sounds good", "perfect",
            "excellent", "great", "love it", "works for me", "i'm in",
            "let's do it", "fantastic", "wonderful", "absolutely", "definitely"
        ]

        # Uncertainty/consideration indicators
        uncertain = [
            "maybe", "perhaps", "might", "could", "consider", "think about",
            "not sure", "uncertain", "possibly", "potential"
        ]

        # Question indicators
        question = [
            "?", "what", "how", "why", "when", "where", "who", "can you",
            "could you", "would you", "do you"
        ]

        # Frustration/Anger indicators
        frustrated = [
            "frustrated", "annoyed", "irritated", "sick of", "tired of",
            "enough", "ridiculous", "waste of time", "this is getting",
            "unreasonable", "unfair", "insulting", "outrageous"
        ]

        # Excitement/Enthusiasm indicators
        excited = [
            "excited", "thrilled", "can't wait", "amazing", "awesome",
            "incredible", "looking forward", "eager", "pumped", "stoked",
            "fantastic opportunity", "perfect timing", "exactly what"
        ]

        # Concern/Worry indicators
        concerned = [
            "concerned", "worried", "nervous", "anxious", "afraid",
            "hesitant", "cautious", "wary", "skeptical", "doubt",
            "risk", "risky", "danger", "scared", "uncomfortable"
        ]

        # Confusion indicators
        confused = [
            "confused", "don't understand", "unclear", "not clear",
            "what do you mean", "don't follow", "lost", "puzzled",
            "doesn't make sense", "can you explain", "not following"
        ]

        # Disappointment indicators
        disappointed = [
            "disappointed", "let down", "expected more", "hoped for",
            "unfortunate", "sad", "unhappy", "dissatisfied", "not what i",
            "expected better", "thought it would", "was hoping"
        ]

        # Interest/Curiosity indicators
        interested = [
            "interesting", "intrigued", "curious", "tell me more",
            "fascinating", "want to know", "would like to hear",
            "sounds interesting", "compelling", "caught my attention"
        ]

        # Urgency/Pressure indicators
        urgent = [
            "urgent", "asap", "immediately", "right away", "now",
            "deadline", "running out", "time sensitive", "critical",
            "emergency", "pressing", "quickly", "hurry", "rush"
        ]

        # Skepticism/Doubt indicators
        skeptical = [
            "skeptical", "doubtful", "sounds too good", "suspicious",
            "hard to believe", "really?", "seriously?", "prove it",
            "show me", "evidence", "not convinced", "need proof"
        ]

        # Apologetic/Conciliatory indicators
        apologetic = [
            "sorry", "apologize", "my fault", "my mistake", "didn't mean",
            "regret", "shouldn't have", "understand your", "see your point",
            "you're right", "fair enough", "makes sense"
        ]

        # Satisfaction/Contentment indicators
        satisfied = [
            "satisfied", "happy with", "content", "pleased", "glad",
            "good enough", "acceptable", "fine with", "okay with",
            "comfortable", "works", "fair"
        ]

        # Count all emotion indicators
        emotion_counts = {
            'strong_disagree': sum(1 for phrase in strong_disagree if phrase in message_lower),
            'disagree': sum(1 for phrase in disagree if phrase in message_lower),
            'agree': sum(1 for phrase in agree if phrase in message_lower),
            'uncertain': sum(1 for phrase in uncertain if phrase in message_lower),
            'question': sum(1 for phrase in question if phrase in message_lower),
            'frustrated': sum(1 for phrase in frustrated if phrase in message_lower),
            'excited': sum(1 for phrase in excited if phrase in message_lower),
            'concerned': sum(1 for phrase in concerned if phrase in message_lower),
            'confused': sum(1 for phrase in confused if phrase in message_lower),
            'disappointed': sum(1 for phrase in disappointed if phrase in message_lower),
            'interested': sum(1 for phrase in interested if phrase in message_lower),
            'urgent': sum(1 for phrase in urgent if phrase in message_lower),
            'skeptical': sum(1 for phrase in skeptical if phrase in message_lower),
            'apologetic': sum(1 for phrase in apologetic if phrase in message_lower),
            'satisfied': sum(1 for phrase in satisfied if phrase in message_lower)
        }

        # Determine primary sentiment (highest count)
        max_count = max(emotion_counts.values())

        # Priority order for ties (more specific emotions first)
        priority_order = [
            'frustrated', 'excited', 'confused', 'disappointed', 'apologetic',
            'concerned', 'skeptical', 'urgent', 'interested', 'strong_disagree',
            'satisfied', 'agree', 'disagree', 'uncertain', 'question'
        ]

        sentiment = 'neutral'
        confidence = 'low'

        if max_count > 0:
            # Find highest priority emotion with max count
            for emotion in priority_order:
                if emotion_counts[emotion] == max_count:
                    sentiment = emotion
                    confidence = 'high' if max_count >= 2 else 'medium'
                    break

        # Detect secondary emotions
        secondary_emotions = [
            emotion for emotion, count in emotion_counts.items()
            if count > 0 and emotion != sentiment
        ]

        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'secondary_emotions': secondary_emotions,
            'is_question': emotion_counts['question'] > 0,
            'is_negative': (emotion_counts['disagree'] + emotion_counts['strong_disagree'] +
                          emotion_counts['frustrated'] + emotion_counts['disappointed']) >
                          (emotion_counts['agree'] + emotion_counts['excited'] + emotion_counts['satisfied']),
            'emotion_intensity': max_count
        }

    def _generate_contextual_direct_response(
        self,
        code: str,
        last_message: str,
        context: str,
        user_role: str,
        negotiation_stage: str
    ) -> str:
        """Generate a direct response from the user's role perspective.

        Args:
            code: Negotiation code
            last_message: The last thing the other party said
            context: Full conversation context
            user_role: User's role (buyer/seller/user)
            negotiation_stage: Current negotiation stage

        Returns:
            Contextual direct response from user's perspective
        """
        last_lower = last_message.lower()
        import re
        numbers = re.findall(r'\d+', last_message)

        # Analyze sentiment of what they said
        sentiment_analysis = self._analyze_message_sentiment(last_message)
        sentiment = sentiment_analysis['sentiment']

        # BUYER perspective - trying to get better price/terms
        if user_role.lower() == 'buyer':
            # Handle emotion-based responses (override code if emotion is detected)
            if sentiment == 'frustrated':
                # They're frustrated - acknowledge and de-escalate
                return f"I can hear this is frustrating for you. Let's take a step back and figure out what's not working. I want to find a solution that works for both of us without the stress."

            elif sentiment == 'excited':
                # They're excited - match enthusiasm and close
                if 'price' in last_lower or '$' in last_message:
                    return f"I love your enthusiasm! I'm excited too. Let's lock in this price and move forward while we're both energized about this."
                return f"Your excitement is contagious! I'm ready to move forward. Let's finalize the details and make this happen."

            elif sentiment == 'confused':
                # They're confused - clarify and simplify
                return f"I can see there's some confusion here. Let me clarify what I'm looking for as a buyer: [clear summary]. Does that help clear things up?"

            elif sentiment == 'disappointed':
                # They're disappointed - acknowledge and pivot
                if 'price' in last_lower or '$' in last_message:
                    return f"I understand you're disappointed with the price I proposed. What price point would make you feel better about this? Let's find something that works."
                return f"I hear the disappointment. What were you hoping for instead? Let's see if we can bridge the gap."

            elif sentiment == 'concerned':
                # They're concerned/worried - address concerns directly
                return f"I understand your concerns. As a buyer, I want to make sure we address those properly. What specifically worries you, and how can we mitigate those risks?"

            elif sentiment == 'urgent':
                # They're expressing urgency - acknowledge timeline
                return f"I hear the urgency. I can work quickly on my end. What's driving this timeline, and what's the absolute deadline we need to meet?"

            elif sentiment == 'skeptical':
                # They're skeptical - provide assurance
                return f"I understand your skepticism. What would help build your confidence in this deal? I'm happy to provide references, guarantees, or whatever you need to feel secure."

            elif sentiment == 'apologetic':
                # They're apologizing - accept graciously and move forward
                return f"I appreciate that. No problem at all - let's focus on moving forward. What do we need to do to make this work?"

            elif sentiment == 'interested':
                # They're interested/curious - encourage and provide info
                return f"I'm glad this interests you! Let me share more details about what I'm looking for and why this could be a great fit for both of us."

            elif sentiment in ['strong_disagree', 'disagree']:
                # They're expressing disagreement/rejection - address their concerns
                if 'price' in last_lower or 'cost' in last_lower or '$' in last_message:
                    if numbers:
                        return f"I understand you're hesitant about the price. What if I offered ${numbers[0]}? Would that work better for you?"
                    return f"I hear your concerns about pricing. What would be a fair price from your perspective? Let's find middle ground."
                elif 'timeline' in last_lower or 'delivery' in last_lower or 'schedule' in last_lower:
                    return f"I understand the timeline is an issue for you. What delivery schedule would work better? Let's see if we can accommodate that."
                elif 'deal' in last_lower or 'agreement' in last_lower:
                    return f"I hear you're not comfortable with this deal. What specific concerns do you have? Let's address them directly."
                else:
                    return f"I understand you have reservations. What would it take to make this work for you? I'm willing to discuss options."

            elif sentiment in ['agree', 'satisfied']:
                # They're showing agreement - close or confirm
                if 'price' in last_lower or '$' in last_message:
                    return f"Excellent! I'm glad we agree on the price. Let's finalize this purchase."
                return f"Great! I'm happy we're aligned. Let's move forward with this agreement."

            elif sentiment == 'question':
                # They're asking questions - provide information
                return f"Good question. Let me clarify: I'm looking for the best value as a buyer, and I'm willing to be flexible if it makes sense for both of us."

            # Code-based responses (when sentiment is neutral or uncertain)
            if code in ['agr', 'mu']:
                if 'price' in last_lower or '$' in last_message:
                    return f"I can accept that price. Let's finalize the purchase at the terms you've outlined."
                return f"That works for me. Let's move forward with this agreement."

            elif code in ['dis', 'coer']:
                if 'price' in last_lower or '$' in last_message:
                    if numbers:
                        return f"That price is too high for my budget. I need you to come down from ${numbers[0]} to make this work."
                    return f"That pricing exceeds my budget. I need a lower price point to proceed."
                elif 'timeline' in last_lower or 'delivery' in last_lower:
                    return f"That delivery timeline is too long. I need it sooner to meet my project deadlines."
                return f"I can't accept those terms. As a buyer, I need better conditions to justify this purchase."

            elif code in ['os', 'om']:
                if 'price' in last_lower or '$' in last_message:
                    if numbers:
                        proposed = int(numbers[0])
                        counter = int(proposed * 0.85)  # Buyer counters lower
                        return f"I appreciate the offer of ${proposed}, but my budget is around ${counter}. Can you work with that?"
                    return f"That's a starting point, but I need a better price. What's the lowest you can go?"
                return f"Here's my counter-offer: I'm willing to buy if you can sweeten the deal on what you've proposed."

            elif code in ['ip', 'ir', 'ib', 'qo', 'qp', 'qb']:
                if 'price' in last_lower:
                    return f"Before I commit to this price, what's included? Are there any additional costs I should know about?"
                return f"I need more details about what you're offering. What exactly am I getting for my money?"

        # SELLER perspective - trying to maximize value/price
        elif user_role.lower() == 'seller':
            # Handle emotion-based responses (override code if emotion is detected)
            if sentiment == 'frustrated':
                # They're frustrated - de-escalate and empathize
                return f"I can hear the frustration, and I want to resolve this. Let's pause and address what's bothering you. I'm committed to making this work without all the stress."

            elif sentiment == 'excited':
                # They're excited - match energy and close
                if 'price' in last_lower or '$' in last_message:
                    return f"I love your enthusiasm! This is exactly the kind of win-win deal I enjoy. Let's finalize the price and get you started right away."
                return f"Your excitement tells me we're on the right track! Let's capitalize on this momentum and get everything finalized."

            elif sentiment == 'confused':
                # They're confused - simplify and clarify
                return f"Let me clarify - I can see there's some confusion. Here's what I'm offering in simple terms: [clear explanation]. Does that make more sense?"

            elif sentiment == 'disappointed':
                # They're disappointed - pivot and add value
                if 'price' in last_lower or '$' in last_message:
                    return f"I understand you're disappointed with the pricing. Let me show you what else I can include to make this feel like better value. What would turn this around for you?"
                return f"I hear the disappointment. What were you expecting instead? Let me see how I can adjust to meet those expectations."

            elif sentiment == 'concerned':
                # They're concerned - provide reassurance
                return f"I understand your concerns, and they're valid. Let me address them head-on and show you how I mitigate these risks. What specifically worries you most?"

            elif sentiment == 'urgent':
                # They're expressing urgency - show you can deliver
                return f"I understand the urgency. I can expedite this on my end. What's the critical deadline, and what would it take to make this happen quickly?"

            elif sentiment == 'skeptical':
                # They're skeptical - provide proof and credibility
                return f"I appreciate healthy skepticism. Let me provide evidence - here's proof of quality, testimonials, and guarantees. What specific concerns can I address with facts?"

            elif sentiment == 'apologetic':
                # They're apologizing - be gracious and move forward
                return f"I appreciate that - no worries at all. Let's move past it and focus on what matters: making this deal work. What do you need from me?"

            elif sentiment == 'interested':
                # They're interested - build on it and sell value
                return f"I'm glad this caught your interest! Let me tell you more about why this is such a great opportunity and how it specifically benefits you."

            elif sentiment in ['strong_disagree', 'disagree']:
                # They're expressing disagreement/rejection - address their concerns
                if 'price' in last_lower or 'cost' in last_lower or '$' in last_message:
                    if numbers:
                        return f"I understand ${numbers[0]} seems high to you. Let me explain the value: this includes premium quality and features that justify the investment. What aspects are most important to you?"
                    return f"I hear your concern about the price. Let me break down what you're getting for this investment and why it's worth it. What would make the value clearer to you?"
                elif 'timeline' in last_lower or 'delivery' in last_lower or 'schedule' in last_lower:
                    return f"I understand timing is critical for you. While I can't compromise quality, let me see what adjustments I can make to the schedule. What's your absolute deadline?"
                elif 'deal' in last_lower or 'agreement' in last_lower:
                    return f"I hear you're not satisfied with the current terms. Let me understand what's not working for you so we can find a solution that works for both of us."
                else:
                    return f"I understand your hesitation. Let me address your concerns directly. What specifically would make this work better for you?"

            elif sentiment in ['agree', 'satisfied']:
                # They're showing agreement - close the deal
                if 'price' in last_lower or '$' in last_message:
                    return f"Excellent! I'm glad we agree on the price. Let's finalize this sale and get you set up."
                return f"Perfect! I'm pleased we could reach this agreement. Let's move forward."

            elif sentiment == 'question':
                # They're asking questions - provide information to build value
                return f"Great question. Let me explain: what I'm offering provides exceptional value because of [quality/features/service], and I'm confident it meets your needs."

            # Code-based responses (when sentiment is neutral or uncertain)
            if code in ['agr', 'mu']:
                if 'price' in last_lower or '$' in last_message:
                    return f"Excellent. I'm glad we agree on the price. Let's finalize this sale."
                return f"Perfect. I'm pleased we could reach this agreement."

            elif code in ['dis', 'coer']:
                if 'price' in last_lower or '$' in last_message:
                    if numbers:
                        return f"I can't go as low as ${numbers[0]}. That doesn't cover my costs and the value I'm providing."
                    return f"That price is below what this is worth. The quality and value justify a higher price point."
                elif 'timeline' in last_lower or 'delivery' in last_lower:
                    return f"I can't rush the timeline that much. Quality work takes time, and I won't compromise on that."
                return f"I have to push back on that. What I'm offering is worth more than what you're suggesting."

            elif code in ['os', 'om']:
                if 'price' in last_lower or '$' in last_message:
                    if numbers:
                        proposed = int(numbers[0])
                        counter = int(proposed * 1.15)  # Seller counters higher
                        return f"Your offer of ${proposed} is below market value. I can come down to ${counter}, which is fair given the quality."
                    return f"I appreciate your interest, but I need a higher price to make this work. Here's what I can do..."
                return f"Based on what you've said, here's my counter-proposal with better terms for the value I'm providing."

            elif code in ['ip', 'ir', 'ib', 'qo', 'qp', 'qb']:
                if 'price' in last_lower:
                    return f"Let me explain the pricing: this includes premium features and quality that justify the cost."
                return f"Let me provide more context about what makes this offering valuable and worth the investment."

        # Default/USER role
        else:
            if code in ['agr', 'mu']:
                return f"I agree with what you've said. Let's move forward with this approach."
            elif code in ['dis', 'coer']:
                return f"I have to push back on what you've suggested. That doesn't align with my needs."
            elif code in ['os', 'om']:
                return f"In response to your point, here's my counter-proposal."
            else:
                return f"Regarding what you just said - I need to clarify a few points before responding."

    def _generate_contextual_collaborative_response(
        self,
        code: str,
        last_message: str,
        user_role: str,
        negotiation_stage: str
    ) -> str:
        """Generate a collaborative response from the user's role perspective.

        Args:
            code: Negotiation code
            last_message: The last thing the other party said
            user_role: User's role (buyer/seller/user)
            negotiation_stage: Current negotiation stage

        Returns:
            Contextual collaborative response
        """
        last_lower = last_message.lower()
        import re
        numbers = re.findall(r'\d+', last_message)

        # Analyze sentiment of what they said
        sentiment_analysis = self._analyze_message_sentiment(last_message)
        sentiment = sentiment_analysis['sentiment']

        # BUYER perspective - collaborative but still cost-conscious
        if user_role.lower() == 'buyer':
            # Handle emotion-based collaborative responses
            if sentiment == 'frustrated':
                return f"I can feel the frustration. Let's reset and work through this together calmly. We're on the same team here - what can we do collaboratively to ease the tension and find a solution?"

            elif sentiment == 'excited':
                return f"I love that energy! Let's channel that excitement into making sure this works perfectly for both of us. What else should we align on to make this amazing?"

            elif sentiment == 'confused':
                return f"I can see there's confusion. Let's slow down and work through this together step by step. What specific parts need clarification so we're on the same page?"

            elif sentiment == 'disappointed':
                return f"I hear the disappointment. Let's collaborate to turn this around - what can we adjust together to make this meet your expectations while working for me too?"

            elif sentiment == 'concerned':
                return f"Your concerns are valid. Let's work together to address them. How can we partner to mitigate these risks in a way that protects both of us?"

            elif sentiment == 'urgent':
                return f"I understand the urgency. Let's collaborate to expedite this efficiently. How can we work together to meet your timeline while ensuring quality?"

            elif sentiment == 'skeptical':
                return f"I appreciate you sharing your doubts. Let's work together transparently - what information or guarantees would help us build trust and move forward collaboratively?"

            elif sentiment == 'apologetic':
                return f"I appreciate that - no worries. Let's move forward together positively. How can we collaborate to make sure we're both happy with the outcome?"

            elif sentiment == 'interested':
                return f"I'm glad this interests you! Let's explore this together. How can we collaborate to make sure this works wonderfully for both of us?"

            elif sentiment in ['strong_disagree', 'disagree']:
                # They're disagreeing - acknowledge and find common ground
                if 'price' in last_lower or 'cost' in last_lower or '$' in last_message:
                    return f"I hear your concerns about the pricing. Let's work together to find a number that works for both of us. What if we explored different options or packages that fit my budget while meeting your needs?"
                elif 'deal' in last_lower or 'agreement' in last_lower:
                    return f"I appreciate your honesty that this deal isn't working for you. Let's take a step back - what would make this a win-win? I'm committed to finding something that works for both of us."
                else:
                    return f"I understand you have concerns. Let's work through them together. What aspects are most important to you, and how can we find middle ground?"

            elif sentiment in ['agree', 'satisfied']:
                # They're agreeing - build on it collaboratively
                return f"I'm glad we're on the same page! Let's build on this agreement and make sure we've covered everything to make this work well for both of us."

            # Existing collaborative logic for neutral sentiment
            if 'price' in last_lower or '$' in last_message:
                if numbers:
                    proposed = int(numbers[0])
                    counter = int(proposed * 0.90)
                    return f"I appreciate your offer of ${proposed}. Let's work together - if you can meet me at ${counter}, we have a deal that works for both of us."
                return f"I value what you're offering. Let's find a price point that fits my budget while being fair to you."
            elif 'timeline' in last_lower or 'delivery' in last_lower:
                return f"I understand your timeline constraints. Let's collaborate on a schedule - perhaps we can be flexible if it means better quality?"
            return f"I like your approach. As a buyer, I want to make sure we both win here. How can we structure this to meet both our needs?"

        # SELLER perspective - collaborative but value-focused
        elif user_role.lower() == 'seller':
            # Handle emotion-based collaborative responses
            if sentiment == 'frustrated':
                return f"I can sense the frustration. Let's take a collaborative approach to work through this without stress. We're partners in finding a solution - how can we make this easier together?"

            elif sentiment == 'excited':
                return f"Your excitement is fantastic! Let's work together to keep this momentum going and finalize something that exceeds both our expectations."

            elif sentiment == 'confused':
                return f"I can see some confusion here. Let's work through this together clearly. What specific aspects should we clarify collaboratively?"

            elif sentiment == 'disappointed':
                return f"I understand the disappointment. Let's collaborate to turn this around - what can we co-create together that would better meet your expectations?"

            elif sentiment == 'concerned':
                return f"Your concerns make sense. Let's work together to address them. How can we partner to create safeguards that protect both of us?"

            elif sentiment == 'urgent':
                return f"I hear the urgency. Let's collaborate on an accelerated timeline that works for both of us. How can we work together to meet your deadline?"

            elif sentiment == 'skeptical':
                return f"I appreciate your honesty about doubts. Let's work together transparently to build confidence. What can we collaboratively do to earn your trust?"

            elif sentiment == 'apologetic':
                return f"I appreciate that - it's all good. Let's move forward together constructively. How can we collaborate to ensure we're both satisfied?"

            elif sentiment == 'interested':
                return f"I'm excited you're interested! Let's explore this opportunity together. How can we collaborate to make sure this delivers value for both of us?"

            elif sentiment in ['strong_disagree', 'disagree']:
                # They're disagreeing - acknowledge and work toward solution
                if 'price' in last_lower or 'cost' in last_lower or '$' in last_message:
                    return f"I understand the price is a sticking point for you. Let's work together to find a solution - perhaps we can adjust the package or add value in other ways that make this work within your budget?"
                elif 'deal' in last_lower or 'agreement' in last_lower:
                    return f"I appreciate you being upfront that this isn't the right deal yet. Let's collaborate to reshape it - what would make this a win-win for both of us?"
                else:
                    return f"I hear your concerns. Let's work through them together. I'm confident we can find an approach that addresses your needs while working for both of us."

            elif sentiment in ['agree', 'satisfied']:
                # They're agreeing - close collaboratively
                return f"Excellent! I'm glad we found common ground. Let's work together to finalize the details and make sure this works perfectly for both of us."

            # Existing collaborative logic for neutral sentiment
            if 'price' in last_lower or '$' in last_message:
                if numbers:
                    proposed = int(numbers[0])
                    counter = int(proposed * 1.10)
                    return f"I hear your budget of ${proposed}. Let's work together - I can add extra value and meet you at ${counter} to make this mutually beneficial."
                return f"I appreciate your position. Let's find a price that works - I'm willing to discuss options that reflect the quality while fitting your needs."
            elif 'timeline' in last_lower or 'delivery' in last_lower:
                return f"I understand timing is important to you. Let's collaborate on a schedule that allows me to deliver quality while meeting your deadline."
            return f"I value working together on this. Let me show you how we can structure this deal to maximize value for both of us."

        # Default
        else:
            if 'price' in last_lower or '$' in last_message:
                return f"I understand your position on pricing. Let's find a middle ground that works for both of us."
            return f"Building on what you've said, let's work together to find a win-win solution."

    def _generate_contextual_question_response(
        self,
        code: str,
        last_message: str,
        user_role: str,
        negotiation_stage: str
    ) -> str:
        """Generate an inquisitive response from the user's role perspective.

        Args:
            code: Negotiation code
            last_message: The last thing the other party said
            user_role: User's role (buyer/seller/user)
            negotiation_stage: Current negotiation stage

        Returns:
            Contextual question response
        """
        last_lower = last_message.lower()
        import re
        numbers = re.findall(r'\d+', last_message)

        # Analyze sentiment of what they said
        sentiment_analysis = self._analyze_message_sentiment(last_message)
        sentiment = sentiment_analysis['sentiment']

        # BUYER perspective - questions focus on value, costs, guarantees
        if user_role.lower() == 'buyer':
            # Handle emotion-based questions
            if sentiment == 'frustrated':
                return f"I can hear the frustration. Can you help me understand what's causing it? What specific issues should we tackle first to make this less stressful?"

            elif sentiment == 'excited':
                return f"I love the enthusiasm! Before we move forward, can you confirm the key details? What makes you most excited about this?"

            elif sentiment == 'confused':
                return f"I can sense some confusion. What specifically isn't clear? Let me ask the right questions to help clarify things."

            elif sentiment == 'disappointed':
                return f"I hear disappointment. Can you tell me what fell short of your expectations? What would have made this better from your perspective?"

            elif sentiment == 'concerned':
                return f"I understand you're concerned. What specific risks worry you most? What guarantees or assurances would help address those concerns?"

            elif sentiment == 'urgent':
                return f"I understand the time pressure. What's driving this urgency? Is there flexibility if we need it, or is the deadline absolute?"

            elif sentiment == 'skeptical':
                return f"I appreciate your skepticism. What specific claims are you doubting? What evidence or proof would help convince you?"

            elif sentiment == 'apologetic':
                return f"I appreciate that. No worries - can we move forward? What do you need from me at this point?"

            elif sentiment == 'interested':
                return f"I'm glad this caught your interest! What aspects intrigue you most? What else would you like to know?"

            elif sentiment in ['strong_disagree', 'disagree']:
                # They're disagreeing - probe to understand why
                if 'price' in last_lower or 'cost' in last_lower or '$' in last_message:
                    return f"Help me understand what's not working with the pricing from your perspective. What would you need to see to make this viable?"
                elif 'deal' in last_lower or 'agreement' in last_lower:
                    return f"I hear this deal isn't right for you. Can you tell me specifically what concerns you have? What would need to change to make this work?"
                else:
                    return f"I sense you're not comfortable with this. Can you walk me through your concerns? What would address them?"

            elif sentiment in ['agree', 'satisfied']:
                # They're agreeing - confirm details
                return f"Great! Before we finalize, can you confirm all the details are clear? What's included, timeline, and any other important points?"

            # Existing question logic for neutral sentiment
            if 'price' in last_lower or '$' in last_message:
                if numbers:
                    return f"Can you break down how you arrived at ${numbers[0]}? What's included in that price, and are there any hidden costs?"
                return f"What factors went into your pricing? Is there any flexibility if I commit to a larger purchase or longer term?"

            elif 'timeline' in last_lower or 'delivery' in last_lower:
                return f"What's driving that timeline? If I need it faster, what would that cost? And what guarantees do I have on the delivery date?"

            elif 'warranty' in last_lower or 'guarantee' in last_lower or 'quality' in last_lower:
                return f"What kind of guarantees or warranties come with this? What happens if there are issues?"

            return f"Before I commit, can you clarify: What exactly am I getting for my money? What's included and what would cost extra?"

        # SELLER perspective - questions focus on requirements, budget, urgency
        elif user_role.lower() == 'seller':
            # Handle emotion-based questions
            if sentiment == 'frustrated':
                return f"I can sense frustration. Can you help me understand what's causing it? What specific issues should I address to make this smoother for you?"

            elif sentiment == 'excited':
                return f"I love your excitement! What aspects appeal to you most? Before we proceed, can you confirm your key requirements and timeline?"

            elif sentiment == 'confused':
                return f"I can sense some confusion. What parts aren't clear to you? How can I explain this better so it makes sense?"

            elif sentiment == 'disappointed':
                return f"I hear disappointment. What were you hoping for that I haven't delivered? How can I adjust to better meet your expectations?"

            elif sentiment == 'concerned':
                return f"I understand your concerns. What specific risks worry you? What safeguards or guarantees would give you confidence in moving forward?"

            elif sentiment == 'urgent':
                return f"I understand the urgency. What's the deadline we're working with? Is there any flexibility, or is this time-critical?"

            elif sentiment == 'skeptical':
                return f"I appreciate honest skepticism. What specific aspects do you doubt? What proof or evidence would help you feel confident about this?"

            elif sentiment == 'apologetic':
                return f"I appreciate that - it's all good. Can we move forward? What do you need from me to proceed?"

            elif sentiment == 'interested':
                return f"I'm thrilled you're interested! What caught your attention? What other information would help you make a decision?"

            elif sentiment in ['strong_disagree', 'disagree']:
                # They're disagreeing - understand their objections
                if 'price' in last_lower or 'cost' in last_lower or '$' in last_message:
                    return f"I understand you have concerns about the price. Can you help me understand what's driving that? Is it budget constraints, or do you not see the value yet? What would make the pricing work for you?"
                elif 'deal' in last_lower or 'agreement' in last_lower:
                    return f"I hear you're not satisfied with the current offer. What specifically isn't working? If you could redesign this deal, what would it look like?"
                else:
                    return f"I sense some hesitation. Can you share what's giving you pause? Understanding your concerns will help me address them properly."

            elif sentiment in ['agree', 'satisfied']:
                # They're agreeing - confirm and close
                return f"Excellent! Let me make sure I have everything right. What are your key requirements, and when would you like to move forward?"

            # Existing question logic for neutral sentiment
            if 'price' in last_lower or '$' in last_message or 'budget' in last_lower:
                if numbers:
                    return f"I see your budget is around ${numbers[0]}. What's your priority - are you flexible on price if I add more value, or is this a hard limit?"
                return f"Help me understand your budget constraints. If I can demonstrate additional value, is there flexibility? What's most important to you?"

            elif 'timeline' in last_lower or 'deadline' in last_lower or 'urgent' in last_lower:
                return f"What's driving your timeline? Is there flexibility if it means better quality? And how firm is that deadline?"

            elif 'features' in last_lower or 'requirements' in last_lower:
                return f"Can you prioritize your requirements? What's must-have versus nice-to-have? This helps me tailor the best solution for you."

            return f"To ensure I'm offering you the best solution - what are your top priorities here? What would make this a definite yes for you?"

        # Default
        else:
            if 'price' in last_lower or '$' in last_message:
                return f"Can you explain how you arrived at that figure? What factors are you considering?"
            return f"Help me understand what's most important to you in what you've outlined. What are your priorities?"

    def _generate_contextual_tactical_response(self, code: str, last_message: str) -> str:
        """Generate a tactical/strategic response that addresses what was said.

        Args:
            code: Negotiation code
            last_message: The last thing the other party said

        Returns:
            Contextual tactical response
        """
        last_lower = last_message.lower()

        if code in ['agr', 'mu']:
            return f"I agree with the direction you've outlined. Strategically, I suggest we document this with clear deliverables and milestones."

        elif code in ['dis', 'coer']:
            if 'price' in last_lower or '$' in last_message:
                return f"I hear your pricing position. Let me present market data and comparable benchmarks that might shift our discussion."
            elif 'timeline' in last_lower:
                return f"Regarding the timeline you mentioned, let me explain the quality trade-offs with that schedule and suggest a strategic alternative."
            else:
                return f"I understand your position, but let me present an alternative perspective with supporting rationale that may change the equation."

        elif code in ['os', 'om']:
            if 'price' in last_lower or '$' in last_message:
                return f"Based on the pricing you've mentioned, here's a strategic counter: I can offer value in other areas if we adjust that figure."
            return f"That's movement in the right direction. Here's what would make this work strategically from my end."

        elif code in ['ip', 'ir', 'ib']:
            return f"You've raised an important question. Let me provide context and data that frames this strategically."

        else:
            # Tactical response that references their message
            if 'must' in last_lower or 'need to' in last_lower or 'have to' in last_lower:
                return f"I understand you have requirements. Let me propose a strategic approach that addresses those while protecting both our interests."
            elif 'can\'t' in last_lower or 'won\'t' in last_lower or 'not' in last_lower:
                return f"I hear the limitations you've mentioned. Let me reframe this strategically to find a path forward that works."
            else:
                return f"Taking a strategic view of what you've said, here's what I propose and why it makes sense for both parties."

    def _generate_collaborative_response(self, code: str) -> str:
        """Generate collaborative response."""
        responses = {
            'agr': "I appreciate your perspective. Let's work together to make this beneficial for both of us.",
            'dis': "I understand your concerns. Perhaps we can find a middle ground that works for everyone.",
            'os': "That's an interesting proposal. Can we explore how this benefits both parties?",
            'om': "I see value in your offer. Let's discuss how we can adjust it to meet both our needs.",
            'ip': "I'd like to understand your position better so we can find common ground.",
            'qb': "Let's make sure we're both on the same page about the context and background.",
        }
        return responses.get(code, "I'm committed to finding a solution that works well for both of us. What aspects are most important to you?")

    def _generate_question_response(self, code: str) -> str:
        """Generate question-based response."""
        responses = {
            'agr': "Before we finalize, can you clarify a few details about implementation?",
            'dis': "What concerns you most about this approach? I'd like to understand your perspective.",
            'os': "Can you help me understand the reasoning behind this offer?",
            'om': "What would make this proposal more acceptable from your standpoint?",
            'ip': "What are the key priorities driving your position on this?",
            'qb': "Can you share more about the circumstances that led to this situation?",
        }
        return responses.get(code, "What information would help you feel more comfortable moving forward?")

    def _generate_tactical_response(self, code: str) -> str:
        """Generate tactical/strategic response."""
        responses = {
            'agr': "I agree in principle. Let me suggest we document this agreement with specific terms.",
            'dis': "I hear your concerns. Let me present an alternative that might address them.",
            'os': "I appreciate the offer. Based on market standards, I'd like to propose a slight adjustment.",
            'om': "That's movement in the right direction. Here's what would make this work from my end.",
            'ip': "My position is based on industry benchmarks and comparable situations.",
            'qb': "The context here is important. Let me provide relevant background that may influence our discussion.",
        }
        return responses.get(code, "Let me take a strategic approach: here's what I propose and why it makes sense.")

    def _generate_recommendations(
        self,
        prediction: Dict[str, any],
        messages: List[Dict[str, str]],
        user_role: str = 'user'
    ) -> List[str]:
        """Generate negotiation recommendations based on context and user role.

        Args:
            prediction: Prediction results
            messages: Conversation history
            user_role: The role of the user

        Returns:
            List of recommendation strings
        """
        recommendations = []

        code = prediction['predicted_code']

        # Add code-specific recommendations
        if code in ['agr', 'sim', 'mu']:
            recommendations.append("The conversation is moving toward agreement. Consider solidifying the deal.")
        elif code in ['dis', 'diff']:
            recommendations.append("There's disagreement. Try to understand the other party's concerns.")
        elif code.startswith('q'):
            recommendations.append("Questions are being asked. Provide clear, honest information.")
        elif code.startswith('o'):
            recommendations.append("An offer is being made. Evaluate it carefully against your goals.")
        elif code in ['ip', 'ir', 'ib']:
            recommendations.append("Inquiry detected. Be prepared to explain your position clearly.")

        # Add general negotiation tips
        if len(messages) > 10:
            recommendations.append("Long negotiation detected. Consider summarizing agreed points.")

        return recommendations


def test_inference():
    """Test the chat inference system."""
    print("\n" + "="*60)
    print("Testing Chat Inference System")
    print("="*60 + "\n")

    # List available models
    ChatInference.list_available_models()

    # Initialize with a specific model (e.g., BERT)
    print("Initializing with BERT model...\n")
    chat = ChatInference(model_name="BERT")

    # Test context
    test_context = "Buyer: I'm interested in purchasing this car, but the price seems high. Can we discuss?"

    print(f"Context: {test_context}\n")

    # Get prediction
    result = chat.predict_next_content(test_context)

    print("Prediction Results:")
    print(f"  Generated Text: {result['generated_text']}")
    print(f"  Predicted Code: {result['predicted_code']}")
    print(f"  Code Description: {result['code_description']}")

    # Test conversation analysis
    messages = [
        {"role": "buyer", "content": "I'm interested in this car."},
        {"role": "seller", "content": "Great! It's a fantastic vehicle."},
        {"role": "buyer", "content": "What's your best price?"}
    ]

    print("\n" + "="*60)
    print("Conversation Analysis")
    print("="*60 + "\n")

    analysis = chat.analyze_conversation(messages)

    print("Stats:")
    for key, value in analysis['conversation_stats'].items():
        print(f"  {key}: {value}")

    print("\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"  - {rec}")

    print("\n" + "="*60)


if __name__ == "__main__":
    test_inference()
