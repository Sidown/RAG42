from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """A source location in the indexed corpus."""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without a ground truth answer."""
    question_id: str = Field(default_factory=lambda:
                             str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its ground truth sources and answer."""
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions, answered or unanswered."""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Search results for a single question."""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results for a single question with a generated answer."""
    answer: str


class StudentSearchResults(BaseModel):
    """The full output of a search_dataset run."""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """The full output of an answer_dataset run."""
    search_results: List[MinimalAnswer]
    k: int
