#!/usr/bin/env python3
"""
Negotiation Analytics Module
Research-backed analysis tools for negotiation conversations
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter


class IssueTracker:
    """Extract and track negotiation issues throughout conversation.

    Based on: Lax & Sebenius (1986) - Multi-issue negotiation enables value creation
    """

    def __init__(self):
        self.issues = {}
        self.issue_keywords = {
            'price': ['price', 'cost', '$', 'payment', 'fee', 'charge', 'dollar', 'money'],
            'delivery': ['delivery', 'timeline', 'schedule', 'deadline', 'shipping', 'arrival', 'when'],
            'warranty': ['warranty', 'guarantee', 'support', 'coverage', 'protection'],
            'quality': ['quality', 'specs', 'features', 'condition', 'grade', 'standard'],
            'quantity': ['quantity', 'volume', 'amount', 'how much', 'how many', 'units'],
            'terms': ['terms', 'conditions', 'contract', 'agreement', 'clause'],
            'customization': ['custom', 'modification', 'adjust', 'tailor', 'personalize'],
            'maintenance': ['maintenance', 'service', 'repair', 'upkeep', 'support']
        }

    def extract_issues(self, conversation: List[Dict]) -> Dict:
        """Identify and track negotiation issues from conversation.

        Args:
            conversation: List of message dictionaries

        Returns:
            Dictionary of identified issues with metadata
        """
        self.issues = {}

        for msg_idx, msg in enumerate(conversation):
            content_lower = msg['content'].lower()
            role = msg.get('role', 'user')

            for issue, keywords in self.issue_keywords.items():
                if any(kw in content_lower for kw in keywords):
                    if issue not in self.issues:
                        self.issues[issue] = {
                            'first_mentioned': msg_idx,
                            'buyer_mentions': 0,
                            'seller_mentions': 0,
                            'total_mentions': 0,
                            'positions': [],
                            'importance_score': 0
                        }

                    # Track who mentioned it
                    if 'buyer' in role.lower():
                        self.issues[issue]['buyer_mentions'] += 1
                    elif 'seller' in role.lower():
                        self.issues[issue]['seller_mentions'] += 1

                    self.issues[issue]['total_mentions'] += 1
                    self.issues[issue]['positions'].append({
                        'turn': msg_idx,
                        'role': role,
                        'snippet': content_lower[:100]
                    })

        # Calculate importance scores (more mentions = more important)
        max_mentions = max([issue['total_mentions'] for issue in self.issues.values()]) if self.issues else 1
        for issue in self.issues.values():
            issue['importance_score'] = issue['total_mentions'] / max_mentions

        return self.issues

    def suggest_tradeoffs(self) -> List[Dict]:
        """Suggest potential tradeoffs based on inferred priorities.

        Based on: Pruitt (1981) - Logrolling creates value by trading different-priority issues

        Returns:
            List of tradeoff suggestions
        """
        if len(self.issues) < 2:
            return []

        suggestions = []

        # Find issues with asymmetric importance
        buyer_priorities = sorted(
            [(issue, data['buyer_mentions']) for issue, data in self.issues.items()],
            key=lambda x: x[1],
            reverse=True
        )

        seller_priorities = sorted(
            [(issue, data['seller_mentions']) for issue, data in self.issues.items()],
            key=lambda x: x[1],
            reverse=True
        )

        if buyer_priorities and seller_priorities:
            top_buyer_issue = buyer_priorities[0][0]
            top_seller_issue = seller_priorities[0][0]

            if top_buyer_issue != top_seller_issue:
                suggestions.append({
                    'type': 'logrolling_opportunity',
                    'suggestion': f"Trade flexibility on {top_seller_issue} for gains on {top_buyer_issue}",
                    'why': f"Buyer focuses on {top_buyer_issue} ({buyer_priorities[0][1]} mentions), seller on {top_seller_issue} ({seller_priorities[0][1]} mentions). Different priorities = value creation.",
                    'research': 'Pruitt (1981): Trading issues of different importance creates integrative value'
                })

        # Suggest unexplored issues
        all_possible_issues = set(self.issue_keywords.keys())
        discussed_issues = set(self.issues.keys())
        unexplored = all_possible_issues - discussed_issues

        if unexplored and len(self.issues) >= 1:
            suggestions.append({
                'type': 'expand_issues',
                'suggestion': f"Consider discussing: {', '.join(list(unexplored)[:3])}",
                'why': "More issues create more opportunities for creative solutions",
                'research': 'Lax & Sebenius (1986): Multi-issue negotiations enable integrative bargaining'
            })

        return suggestions


class ConcessionTracker:
    """Track concessions and reciprocity patterns.

    Based on: Yukl (1974) - Gradual concessions signal firmness
              Cialdini (2006) - Reciprocity norm in negotiation
    """

    def __init__(self):
        self.concessions = []
        self.concession_keywords = {
            'accept': ['accept', 'agree', 'okay', 'ok', 'fine', 'yes'],
            'reduce': ['lower', 'reduce', 'decrease', 'down to', 'bring down'],
            'increase': ['raise', 'increase', 'up to', 'bring up'],
            'flexibility': ['flexible', 'willing', 'could', 'might', 'consider'],
            'compromise': ['compromise', 'meet you', 'middle ground', 'halfway']
        }

    def analyze_concessions(self, conversation: List[Dict]) -> Dict:
        """Analyze concession patterns throughout conversation.

        Args:
            conversation: List of message dictionaries

        Returns:
            Analysis of concession patterns
        """
        self.concessions = []

        for i, msg in enumerate(conversation):
            content_lower = msg['content'].lower()
            role = msg.get('role', 'user')

            # Detect concessions
            concession_type = None
            for ctype, keywords in self.concession_keywords.items():
                if any(kw in content_lower for kw in keywords):
                    concession_type = ctype
                    break

            if concession_type:
                # Estimate magnitude by looking for numeric changes
                magnitude = self._estimate_magnitude(msg, conversation[:i])

                self.concessions.append({
                    'turn': i,
                    'role': role,
                    'type': concession_type,
                    'magnitude': magnitude,
                    'content_snippet': msg['content'][:100]
                })

        # Analyze patterns
        buyer_concessions = [c for c in self.concessions if 'buyer' in c['role'].lower()]
        seller_concessions = [c for c in self.concessions if 'seller' in c['role'].lower()]

        reciprocity_score = self._calculate_reciprocity(buyer_concessions, seller_concessions)

        return {
            'total_concessions': len(self.concessions),
            'buyer_concessions': len(buyer_concessions),
            'seller_concessions': len(seller_concessions),
            'reciprocity_score': reciprocity_score,
            'pattern': self._identify_pattern(self.concessions),
            'recommendation': self._suggest_next_concession(buyer_concessions, seller_concessions),
            'concessions': self.concessions
        }

    def _estimate_magnitude(self, current_msg: Dict, previous_msgs: List[Dict]) -> str:
        """Estimate concession magnitude by comparing numeric values."""
        current_numbers = re.findall(r'\d+', current_msg['content'])

        if not current_numbers:
            return 'small'

        # Look for previous numbers to compare
        for prev_msg in reversed(previous_msgs[-3:]):  # Last 3 messages
            prev_numbers = re.findall(r'\d+', prev_msg['content'])
            if prev_numbers and current_numbers:
                try:
                    current_val = float(current_numbers[0])
                    prev_val = float(prev_numbers[0])
                    change_pct = abs(current_val - prev_val) / prev_val if prev_val > 0 else 0

                    if change_pct > 0.15:
                        return 'large'
                    elif change_pct > 0.05:
                        return 'medium'
                    else:
                        return 'small'
                except:
                    pass

        return 'small'

    def _calculate_reciprocity(self, buyer_concessions: List, seller_concessions: List) -> float:
        """Calculate reciprocity score (0-1).

        Research: Cialdini (2006) - Reciprocity norm
        High score = balanced give-and-take
        """
        if not buyer_concessions and not seller_concessions:
            return 1.0

        total = len(buyer_concessions) + len(seller_concessions)
        if total == 0:
            return 1.0

        balance = 1 - abs(len(buyer_concessions) - len(seller_concessions)) / total
        return balance

    def _identify_pattern(self, concessions: List[Dict]) -> str:
        """Identify concession pattern."""
        if len(concessions) < 2:
            return 'insufficient_data'

        # Check if concessions are alternating (good reciprocity)
        roles = [c['role'] for c in concessions]
        alternating = all(roles[i] != roles[i+1] for i in range(len(roles)-1))

        if alternating:
            return 'reciprocal'

        # Check if one party is making most concessions
        role_counts = Counter(roles)
        most_common_count = role_counts.most_common(1)[0][1]
        if most_common_count > len(concessions) * 0.7:
            return 'one_sided'

        return 'mixed'

    def _suggest_next_concession(self, buyer_concessions: List, seller_concessions: List) -> str:
        """Suggest next concession based on reciprocity.

        Research: Yukl (1974) - Gradual, decreasing concessions
        """
        if len(buyer_concessions) > len(seller_concessions) + 1:
            return "Wait for the other party to reciprocate before making another concession. Research shows balanced reciprocity leads to better outcomes."
        elif len(seller_concessions) > len(buyer_concessions) + 1:
            return "The other party has made multiple concessions. Reciprocate with a small concession to maintain goodwill."
        elif len(buyer_concessions) + len(seller_concessions) == 0:
            return "Make a small initial concession to signal flexibility and encourage reciprocity. Keep larger concessions for later."
        else:
            return "Concessions are balanced. Continue with gradual, decreasing concessions as you approach agreement."


class AnchoringDetector:
    """Detect and analyze anchoring in negotiations.

    Based on: Galinsky & Mussweiler (2001) - First offers strongly influence outcomes
              Mason et al. (2013) - Precise numbers seem more informed
    """

    def __init__(self):
        self.anchors = []

    def detect_anchors(self, conversation: List[Dict]) -> Dict:
        """Detect anchoring attempts in conversation.

        Args:
            conversation: List of message dictionaries

        Returns:
            Analysis of anchoring behavior
        """
        self.anchors = []

        for i, msg in enumerate(conversation):
            # Extract numeric values
            prices = re.findall(r'\$[\d,]+(?:\.\d{2})?', msg['content'])
            percentages = re.findall(r'\d+(?:\.\d+)?%', msg['content'])
            raw_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', msg['content'])

            if prices or percentages or raw_numbers:
                # Check if this is early in conversation (first 3 turns)
                is_early = i < 3

                # Check precision (precise numbers like $2,347 vs. $2,000)
                is_precise = any('.' in p or ',' in p for p in prices + raw_numbers)

                self.anchors.append({
                    'turn': i,
                    'role': msg.get('role', 'user'),
                    'prices': prices,
                    'percentages': percentages,
                    'numbers': raw_numbers,
                    'is_early_anchor': is_early,
                    'is_precise': is_precise,
                    'content_snippet': msg['content'][:100]
                })

        # Analyze
        first_anchor = self.anchors[0] if self.anchors else None

        return {
            'anchor_detected': len(self.anchors) > 0,
            'first_anchor': first_anchor,
            'total_numeric_mentions': len(self.anchors),
            'recommendation': self._get_anchoring_strategy(first_anchor, conversation),
            'anchors': self.anchors
        }

    def _get_anchoring_strategy(self, first_anchor: Optional[Dict], conversation: List[Dict]) -> Dict:
        """Generate strategic recommendations based on anchoring analysis.

        Research: Galinsky & Mussweiler (2001) - Counter-anchoring strategies
        """
        if not first_anchor:
            return {
                'strategy': 'make_first_offer',
                'suggestion': 'No anchor set yet. Consider making a bold first offer to anchor the negotiation in your favor.',
                'tactics': [
                    'Use a precise number (e.g., $12,347) to appear informed',
                    'Provide reasoning for your anchor',
                    'Set an optimistic but defensible starting point'
                ],
                'research': 'Galinsky & Mussweiler (2001): First offers strongly anchor final outcomes'
            }

        # Check if user set the anchor
        user_set_anchor = 'buyer' in first_anchor['role'].lower() or 'user' in first_anchor['role'].lower()

        if user_set_anchor:
            return {
                'strategy': 'defend_anchor',
                'suggestion': 'You set the anchor. Defend it with reasoning and data.',
                'tactics': [
                    'Provide justification for your number',
                    'Show market research or comparables',
                    'Make small, gradual concessions if needed'
                ],
                'research': 'Galinsky & Mussweiler (2001): Anchors are sticky - defend yours'
            }
        else:
            # Other party set anchor
            anchor_values = first_anchor.get('prices', []) or first_anchor.get('numbers', [])

            return {
                'strategy': 'counter_anchor',
                'suggestion': f"They anchored at {anchor_values[0] if anchor_values else 'their number'}. Counter with your own range.",
                'tactics': [
                    'Reject their anchor explicitly: "That\'s outside my range"',
                    'Provide your own counter-anchor immediately',
                    'Use objective criteria to justify your number',
                    'Focus on your interests, not their position'
                ],
                'research': 'Galinsky & Mussweiler (2001): Counter-anchoring reduces anchor effects'
            }


class QuestionStrategyAnalyzer:
    """Analyze and generate strategic questions.

    Based on: Galinsky et al. (2015) - Questions extract information and build rapport
              Thompson (1991) - "Why" questions reveal interests
    """

    def __init__(self):
        self.questions = []
        self.information_gaps = set()

        self.question_types = {
            'diagnostic': {
                'purpose': 'Understand underlying interests',
                'examples': [
                    'Why is {issue} important to you?',
                    'What are you trying to achieve with {issue}?',
                    'What would a successful outcome look like?'
                ]
            },
            'value_creating': {
                'purpose': 'Explore creative solutions',
                'examples': [
                    'What if we could {possibility}?',
                    'How would you feel about {alternative}?',
                    'Could we explore {option} instead?'
                ]
            },
            'priority': {
                'purpose': 'Rank importance of issues',
                'examples': [
                    'Which matters more to you: {issue1} or {issue2}?',
                    'What are your top 3 priorities?',
                    'If you had to choose, what\'s most important?'
                ]
            },
            'constraint': {
                'purpose': 'Identify limitations',
                'examples': [
                    'What constraints are you working with?',
                    'What flexibility do you have on {issue}?',
                    'What are your must-haves vs. nice-to-haves?'
                ]
            },
            'batna': {
                'purpose': 'Assess alternatives',
                'examples': [
                    'What happens if we don\'t reach an agreement?',
                    'Are you considering other options?',
                    'What would make this better than your alternatives?'
                ]
            }
        }

    def analyze_questions(self, conversation: List[Dict]) -> Dict:
        """Analyze question patterns in conversation.

        Args:
            conversation: List of message dictionaries

        Returns:
            Analysis of question usage and suggestions
        """
        self.questions = []

        for i, msg in enumerate(conversation):
            content = msg['content']
            role = msg.get('role', 'user')

            # Detect questions
            if '?' in content:
                # Classify question type
                q_type = self._classify_question(content)

                self.questions.append({
                    'turn': i,
                    'role': role,
                    'type': q_type,
                    'content': content
                })

        # Identify information gaps
        self._identify_information_gaps(conversation)

        return {
            'total_questions': len(self.questions),
            'questions_by_type': self._count_by_type(),
            'information_gaps': list(self.information_gaps),
            'suggested_questions': self._generate_strategic_questions(conversation),
            'questions': self.questions
        }

    def _classify_question(self, question: str) -> str:
        """Classify question type based on content."""
        q_lower = question.lower()

        if any(word in q_lower for word in ['why', 'reason', 'because']):
            return 'diagnostic'
        elif any(word in q_lower for word in ['what if', 'could we', 'would you']):
            return 'value_creating'
        elif any(word in q_lower for word in ['which', 'priority', 'important', 'matters more']):
            return 'priority'
        elif any(word in q_lower for word in ['constraint', 'limitation', 'flexibility']):
            return 'constraint'
        elif any(word in q_lower for word in ['alternative', 'other option', 'if we don\'t']):
            return 'batna'
        else:
            return 'general'

    def _count_by_type(self) -> Dict[str, int]:
        """Count questions by type."""
        counts = defaultdict(int)
        for q in self.questions:
            counts[q['type']] += 1
        return dict(counts)

    def _identify_information_gaps(self, conversation: List[Dict]):
        """Identify what information is still unknown."""
        self.information_gaps = set()

        # Check if we know their priorities
        priority_keywords = ['important', 'priority', 'matters', 'need', 'must have']
        if not any(any(kw in msg['content'].lower() for kw in priority_keywords)
                   for msg in conversation):
            self.information_gaps.add('priorities')

        # Check if we know their constraints
        constraint_keywords = ['budget', 'deadline', 'limit', 'constraint', 'cannot']
        if not any(any(kw in msg['content'].lower() for kw in constraint_keywords)
                   for msg in conversation):
            self.information_gaps.add('constraints')

        # Check if we know their BATNA
        batna_keywords = ['alternative', 'other option', 'else', 'different']
        if not any(any(kw in msg['content'].lower() for kw in batna_keywords)
                   for msg in conversation):
            self.information_gaps.add('alternatives')

    def _generate_strategic_questions(self, conversation: List[Dict]) -> List[Dict]:
        """Generate suggested questions based on information gaps.

        Research: Galinsky et al. (2015) - Strategic question asking
        """
        suggestions = []

        for gap in self.information_gaps:
            if gap == 'priorities':
                suggestions.append({
                    'type': 'priority',
                    'question': 'What are your top priorities in this negotiation?',
                    'why': 'Understanding their priorities helps identify tradeoff opportunities',
                    'research': 'Thompson (1991): Knowing priorities enables logrolling'
                })

            elif gap == 'constraints':
                suggestions.append({
                    'type': 'constraint',
                    'question': 'What constraints or limitations are you working with?',
                    'why': 'Understanding constraints helps find creative solutions within boundaries',
                    'research': 'Galinsky et al. (2015): Questions reveal hidden information'
                })

            elif gap == 'alternatives':
                suggestions.append({
                    'type': 'batna',
                    'question': 'What happens if we don\'t reach an agreement today?',
                    'why': 'Understanding their BATNA helps assess your negotiating power',
                    'research': 'Fisher & Ury (1981): BATNA determines negotiation power'
                })

        return suggestions


class FramingGenerator:
    """Generate alternative framings for suggestions.

    Based on: Kahneman & Tversky (1979) - Prospect theory and framing effects
              Neale & Bazerman (1985) - Framing in negotiation
    """

    def __init__(self):
        self.framing_types = {
            'gain': {
                'description': 'Emphasize what they will gain',
                'template': 'By agreeing to this, you\'ll gain {benefit}.',
                'when_to_use': 'When dealing with risk-averse negotiators'
            },
            'loss': {
                'description': 'Emphasize what they will lose',
                'template': 'Without this agreement, you\'ll miss out on {benefit}.',
                'when_to_use': 'When creating urgency or FOMO'
            },
            'fairness': {
                'description': 'Appeal to fairness and equality',
                'template': 'This is fair because {reasoning}.',
                'when_to_use': 'When both parties value equity'
            },
            'efficiency': {
                'description': 'Highlight time/resource savings',
                'template': 'This approach saves {resource} and gets us to agreement faster.',
                'when_to_use': 'When time pressure exists'
            },
            'relationship': {
                'description': 'Focus on long-term relationship',
                'template': 'This sets us up for a strong partnership going forward.',
                'when_to_use': 'When ongoing relationship matters'
            },
            'scarcity': {
                'description': 'Highlight limited availability',
                'template': 'This opportunity won\'t be available for long.',
                'when_to_use': 'When you have leverage'
            }
        }

    def generate_framed_versions(self, base_suggestion: str, context: Dict) -> List[Dict]:
        """Generate multiple framed versions of a suggestion.

        Args:
            base_suggestion: Original suggestion text
            context: Context about negotiation state

        Returns:
            List of framed suggestions with explanations
        """
        framed_suggestions = []

        for frame_type, frame_info in self.framing_types.items():
            framed_suggestions.append({
                'frame': frame_type,
                'text': self._apply_framing(base_suggestion, frame_type, context),
                'description': frame_info['description'],
                'when_to_use': frame_info['when_to_use'],
                'research': 'Kahneman & Tversky (1979): Framing significantly impacts decision-making'
            })

        return framed_suggestions

    def _apply_framing(self, suggestion: str, frame_type: str, context: Dict) -> str:
        """Apply framing to suggestion text."""

        if frame_type == 'gain':
            # Add gain-focused language
            return f"{suggestion} This approach helps you achieve your goals and secure the value you're looking for."

        elif frame_type == 'loss':
            # Add loss-aversion language
            return f"{suggestion} Without moving forward, we risk missing this opportunity."

        elif frame_type == 'fairness':
            # Add fairness language
            return f"{suggestion} This is a fair outcome that respects both parties' interests."

        elif frame_type == 'efficiency':
            # Add efficiency language
            return f"{suggestion} This streamlines the process and saves us both time."

        elif frame_type == 'relationship':
            # Add relationship language
            return f"{suggestion} This builds a foundation for a strong, lasting partnership."

        elif frame_type == 'scarcity':
            # Add scarcity language
            return f"{suggestion} This is a limited-time opportunity we should act on."

        return suggestion
