"""Chain-of-Thought Consensus fusion strategy - finds agreement among reasoning traces."""

from collections import Counter

from hermes_fusion.strategies.base import (
    FusionResult,
    FusionStrategy,
    ProviderResponse,
    normalize_answer,
)


class CoTConsensusStrategy(FusionStrategy):
    """
    CoT Consensus: extract final answers from chain-of-thought reasoning,
    find majority agreement. Requires responses to include reasoning.
    """
    name = "cot_consensus"

    def __init__(self, min_agreement: float = 0.6, extract_answer_fn=None):
        self.min_agreement = min_agreement
        self.extract_answer_fn = extract_answer_fn or self._default_extract

    def _default_extract(self, content: str) -> str:
        """Extract final answer from CoT response (after 'Answer:' or last paragraph)."""
        import re
        
        # Quick fallback: if content is short (<50 chars) and no colon markers, use as-is
        if len(content) < 50 and not any(kw in content.lower() for kw in ['answer:', 'conclusion:', 'result:']):
            return content.strip()
        
        # Look for explicit answer markers with colon - "Answer: Answer A"
        patterns = [
            r'(?:Answer|answer)[\s:]+([A-Za-z0-9\s\-_]+)',  # Answer: Answer A
            r'(?:Conclusion|conclusion|Result|result)[\s:]+([^\n]+)',
            r'(?:Final answer|final answer)[\s:]+([^\n]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            if matches:
                return matches[-1].strip()
        
        # Fallback: last non-empty paragraph
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if paragraphs:
            return paragraphs[-1]
        
        # Last resort: full content up to 200 chars
        return content[:200]

    async def fuse(self, question: str, responses: list[ProviderResponse]) -> FusionResult:
        if not responses:
            return FusionResult(
                final_answer="",
                confidence=0.0,
                method=self.name,
                participating_providers=[],
            )

        # Extract answers from each response
        extracted = []
        for resp in responses:
            if not resp.content.strip():
                continue
            answer = self.extract_answer_fn(resp.content)
            norm = normalize_answer(answer)
            if norm:
                extracted.append((norm, answer, resp.provider))

        if not extracted:
            return FusionResult(
                final_answer="",
                confidence=0.0,
                method=self.name,
                participating_providers=[r.provider for r in responses],
                raw_responses=responses,
            )

        # Count agreement
        counter = Counter(norm for norm, _, _ in extracted)
        most_common = counter.most_common(1)[0] if counter else ("", 0)
        winning_norm, count = most_common
        total = len(extracted)
        agreement = count / total if total > 0 else 0.0

        if agreement >= self.min_agreement:
            # Find original answer for winning norm
            winning_answer = next(orig for norm, orig, _ in extracted if norm == winning_norm)
            providers = [prov for norm, _, prov in extracted if norm == winning_norm]
            return FusionResult(
                final_answer=winning_answer,
                confidence=agreement,
                method=self.name,
                participating_providers=list(set(providers)),
                raw_responses=responses,
                metadata={
                    "agreement_ratio": agreement,
                    "total_responses": total,
                    "answer_counts": dict(counter),
                },
            )
        else:
            # No consensus - return best with low confidence
            winning_answer = extracted[0][1] if extracted else ""
            return FusionResult(
                final_answer=winning_answer,
                confidence=agreement,
                method=self.name,
                participating_providers=[extracted[0][2]] if extracted else [],
                raw_responses=responses,
                metadata={
                    "agreement_ratio": agreement,
                    "total_responses": total,
                    "answer_counts": dict(counter),
                    "note": "below consensus threshold",
                },
            )