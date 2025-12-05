#!/usr/bin/env python3
"""
RLHF (Reinforcement Learning from Human Feedback) Trainer for NegotiationGPT

Based on:
- Christiano et al. (2017): Deep reinforcement learning from human preferences
- Ouyang et al. (2022): Training language models to follow instructions with human feedback (InstructGPT)
- Stiennon et al. (2020): Learning to summarize with human feedback

This module enables:
1. Collecting human preferences on model suggestions
2. Training a reward model from preferences
3. Fine-tuning the policy model using PPO (Proximal Policy Optimization)
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from NegotiationGPT import (
    NegotiationGPT,
    NegotiationTokenizer,
    D_MODEL,
    DEVICE,
    USE_OUTCOME_HEAD
)


# =========================================================================
# 1. PREFERENCE DATA COLLECTION
# =========================================================================

@dataclass
class PreferenceExample:
    """A single preference comparison.

    Attributes:
        context: The negotiation context/history
        response_a: First candidate response
        response_b: Second candidate response
        preferred: Which response is preferred ('a', 'b', or 'tie')
        quality_score_a: Optional quality rating for response A (0-10)
        quality_score_b: Optional quality rating for response B (0-10)
        feedback: Optional text feedback explaining preference
    """
    context: str
    response_a: str
    response_b: str
    preferred: str  # 'a', 'b', or 'tie'
    quality_score_a: Optional[int] = None
    quality_score_b: Optional[int] = None
    feedback: Optional[str] = None
    metadata: Optional[Dict] = None


class PreferenceCollector:
    """Collects and manages human preference data.

    Research: Christiano et al. (2017) - Learning from human preferences
    """

    def __init__(self, storage_file: str = "preferences.json"):
        self.storage_file = storage_file
        self.preferences = self._load_preferences()

    def _load_preferences(self) -> List[Dict]:
        """Load existing preferences from file."""
        if Path(self.storage_file).exists():
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading preferences: {e}")
                return []
        return []

    def _save_preferences(self):
        """Save preferences to file."""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.preferences, f, indent=2)
        except Exception as e:
            print(f"Error saving preferences: {e}")

    def add_preference(self, example: PreferenceExample):
        """Add a new preference example.

        Args:
            example: PreferenceExample instance
        """
        self.preferences.append({
            'context': example.context,
            'response_a': example.response_a,
            'response_b': example.response_b,
            'preferred': example.preferred,
            'quality_score_a': example.quality_score_a,
            'quality_score_b': example.quality_score_b,
            'feedback': example.feedback,
            'metadata': example.metadata or {}
        })
        self._save_preferences()

    def get_preferences(self, min_count: int = 0) -> List[Dict]:
        """Get all preferences with at least min_count examples.

        Args:
            min_count: Minimum number of preferences to return

        Returns:
            List of preference dictionaries
        """
        return self.preferences if len(self.preferences) >= min_count else []

    def export_for_training(self) -> List[Tuple[str, str, str, float]]:
        """Export preferences in format for reward model training.

        Returns:
            List of (context, response_a, response_b, preference_label) tuples
            preference_label: 1.0 if A preferred, 0.0 if B preferred, 0.5 if tie
        """
        training_data = []

        for pref in self.preferences:
            if pref['preferred'] == 'a':
                label = 1.0
            elif pref['preferred'] == 'b':
                label = 0.0
            else:  # tie
                label = 0.5

            training_data.append((
                pref['context'],
                pref['response_a'],
                pref['response_b'],
                label
            ))

        return training_data


# =========================================================================
# 2. REWARD MODEL
# =========================================================================

class RewardModel(nn.Module):
    """Reward model that predicts quality scores for responses.

    Research: Stiennon et al. (2020) - Learning to summarize with human feedback

    The reward model learns to predict human preferences by:
    1. Encoding context + response
    2. Predicting a scalar reward score
    3. Training on pairwise comparisons
    """

    def __init__(self, base_model: NegotiationGPT, tokenizer: NegotiationTokenizer):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer

        # Freeze base model initially (can fine-tune later)
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Reward head: maps from D_MODEL to scalar reward
        self.reward_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(D_MODEL // 2, 1)
        )

    def forward(self, context: str, response: str) -> torch.Tensor:
        """Compute reward for a context-response pair.

        Args:
            context: Negotiation context/history
            response: Candidate response

        Returns:
            Scalar reward tensor
        """
        # Combine context and response
        full_text = f"{context}\n{response}"

        # Tokenize
        input_ids = self.tokenizer.encode(full_text, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
        attention_mask = torch.ones_like(input_tensor)

        # Get base model outputs
        with torch.no_grad():
            outputs = self.base_model(input_tensor, attention_mask)
            lm_logits = outputs[0]

        # Pool the final layer representations
        # Use mean pooling over sequence
        last_hidden = lm_logits.mean(dim=1)  # (batch_size, vocab_size) -> (batch_size, D_MODEL)

        # Predict reward
        reward = self.reward_head(last_hidden)

        return reward.squeeze(-1)  # (batch_size,)

    def compute_pairwise_loss(
        self,
        context: str,
        response_a: str,
        response_b: str,
        preference: float
    ) -> torch.Tensor:
        """Compute loss for pairwise preference.

        Research: Bradley-Terry model for pairwise comparisons

        Args:
            context: Negotiation context
            response_a: First response
            response_b: Second response
            preference: 1.0 if A preferred, 0.0 if B preferred, 0.5 if tie

        Returns:
            Loss tensor
        """
        reward_a = self.forward(context, response_a)
        reward_b = self.forward(context, response_b)

        # Bradley-Terry model: P(A > B) = sigmoid(reward_a - reward_b)
        logit = reward_a - reward_b

        if preference == 1.0:
            # A preferred: maximize P(A > B)
            loss = F.binary_cross_entropy_with_logits(
                logit,
                torch.ones_like(logit),
                reduction='mean'
            )
        elif preference == 0.0:
            # B preferred: maximize P(B > A)
            loss = F.binary_cross_entropy_with_logits(
                logit,
                torch.zeros_like(logit),
                reduction='mean'
            )
        else:
            # Tie: minimize difference
            loss = torch.abs(logit).mean()

        return loss


class PreferenceDataset(Dataset):
    """Dataset for training reward model from preferences."""

    def __init__(self, preferences: List[Tuple[str, str, str, float]]):
        self.preferences = preferences

    def __len__(self):
        return len(self.preferences)

    def __getitem__(self, idx):
        context, response_a, response_b, preference = self.preferences[idx]
        return {
            'context': context,
            'response_a': response_a,
            'response_b': response_b,
            'preference': preference
        }


def train_reward_model(
    reward_model: RewardModel,
    preferences: List[Tuple[str, str, str, float]],
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    batch_size: int = 4
):
    """Train reward model on preference data.

    Args:
        reward_model: RewardModel instance
        preferences: List of (context, response_a, response_b, preference) tuples
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size

    Returns:
        Trained reward model
    """
    print(f"\n[Reward Model Training] Starting with {len(preferences)} preferences")

    dataset = PreferenceDataset(preferences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(reward_model.reward_head.parameters(), lr=learning_rate)

    reward_model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            optimizer.zero_grad()

            # Compute loss for each example in batch
            batch_loss = 0.0
            for i in range(len(batch['context'])):
                loss = reward_model.compute_pairwise_loss(
                    batch['context'][i],
                    batch['response_a'][i],
                    batch['response_b'][i],
                    batch['preference'][i]
                )
                batch_loss += loss

            batch_loss = batch_loss / len(batch['context'])
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")

    print("[Reward Model Training] Complete!")
    return reward_model


# =========================================================================
# 3. PPO (PROXIMAL POLICY OPTIMIZATION)
# =========================================================================

class PPOTrainer:
    """PPO trainer for fine-tuning policy with learned reward.

    Research: Schulman et al. (2017) - Proximal Policy Optimization

    This is a simplified version. For production, consider using:
    - TRL (Transformer Reinforcement Learning) library
    - RLHF frameworks like trlX
    """

    def __init__(
        self,
        policy_model: NegotiationGPT,
        reward_model: RewardModel,
        tokenizer: NegotiationTokenizer,
        clip_epsilon: float = 0.2,
        kl_coef: float = 0.1
    ):
        self.policy = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef

        # Keep a copy of the original policy for KL divergence
        self.reference_policy = type(policy_model)(policy_model.embedding.num_embeddings)
        self.reference_policy.load_state_dict(policy_model.state_dict())
        self.reference_policy.eval()

        for param in self.reference_policy.parameters():
            param.requires_grad = False

    def compute_rewards(self, contexts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute rewards for generated responses using reward model.

        Args:
            contexts: List of context strings
            responses: List of generated response strings

        Returns:
            Tensor of rewards
        """
        rewards = []
        self.reward_model.eval()

        with torch.no_grad():
            for context, response in zip(contexts, responses):
                reward = self.reward_model(context, response)
                rewards.append(reward.item())

        return torch.tensor(rewards, dtype=torch.float32)

    def compute_kl_penalty(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between current policy and reference.

        This prevents the policy from deviating too far from original.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            KL divergence penalty
        """
        # Get logits from current policy
        policy_outputs = self.policy(input_ids, attention_mask)
        policy_logits = policy_outputs[0]  # LM logits

        # Get logits from reference policy
        with torch.no_grad():
            ref_outputs = self.reference_policy(input_ids, attention_mask)
            ref_logits = ref_outputs[0]

        # Compute KL divergence
        policy_probs = F.softmax(policy_logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)

        kl_div = F.kl_div(
            policy_probs.log(),
            ref_probs,
            reduction='batchmean',
            log_target=False
        )

        return kl_div


# =========================================================================
# 4. UTILITIES
# =========================================================================

def create_preference_ui_data(
    context: str,
    suggestions: List[Dict[str, str]]
) -> Dict:
    """Create data structure for preference collection UI.

    Args:
        context: Negotiation context
        suggestions: List of suggestion dictionaries with 'text', 'style', etc.

    Returns:
        Dictionary formatted for UI preference collection
    """
    return {
        'context': context,
        'candidates': [
            {
                'id': i,
                'text': sug['text'],
                'style': sug.get('style', 'Unknown'),
                'description': sug.get('description', '')
            }
            for i, sug in enumerate(suggestions)
        ]
    }


def save_preference_from_ui(
    collector: PreferenceCollector,
    context: str,
    response_a: str,
    response_b: str,
    preferred: str,
    quality_a: Optional[int] = None,
    quality_b: Optional[int] = None,
    feedback: Optional[str] = None
):
    """Save preference data from UI.

    Args:
        collector: PreferenceCollector instance
        context: Negotiation context
        response_a: First candidate response
        response_b: Second candidate response
        preferred: Which is preferred ('a', 'b', or 'tie')
        quality_a: Quality score for A (0-10)
        quality_b: Quality score for B (0-10)
        feedback: Optional text feedback
    """
    example = PreferenceExample(
        context=context,
        response_a=response_a,
        response_b=response_b,
        preferred=preferred,
        quality_score_a=quality_a,
        quality_score_b=quality_b,
        feedback=feedback
    )

    collector.add_preference(example)
    print(f"[Preference Collected] Total: {len(collector.preferences)}")


# =========================================================================
# 5. EXAMPLE USAGE
# =========================================================================

if __name__ == "__main__":
    print("="*60)
    print("RLHF Infrastructure for NegotiationGPT")
    print("="*60)

    print("\n1. Collecting Preferences:")
    print("   - Users compare suggestion pairs")
    print("   - Preferences saved to preferences.json")

    print("\n2. Training Reward Model:")
    print("   - Learns to predict human preferences")
    print("   - Uses pairwise comparisons (Bradley-Terry model)")

    print("\n3. PPO Fine-tuning:")
    print("   - Optimizes policy to maximize learned reward")
    print("   - Maintains KL divergence penalty for stability")

    print("\n" + "="*60)
    print("To enable RLHF:")
    print("  1. Set USE_OUTCOME_HEAD = True in NegotiationGPT.py")
    print("  2. Collect preferences using the UI")
    print("  3. Train reward model: python rlhf_trainer.py")
    print("  4. Fine-tune policy with PPO")
    print("="*60)
