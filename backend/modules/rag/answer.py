"""
RAG Answer dataclass for structured responses.
"""

from dataclasses import dataclass, field
import adalflow as adal


@dataclass
class RAGAnswer(adal.DataClass):
    """Structured answer from RAG with rationale and formatted response."""
    
    rationale: str = field(
        default="",
        metadata={"desc": "Chain of thoughts for the answer."}
    )
    answer: str = field(
        default="",
        metadata={
            "desc": "Answer to the user query, formatted in markdown for beautiful "
                    "rendering with react-markdown. DO NOT include ``` triple backticks "
                    "fences at the beginning or end of your answer."
        }
    )

    __output_fields__ = ["rationale", "answer"]
