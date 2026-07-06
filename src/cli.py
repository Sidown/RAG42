from src.chunker import get_all_chunk
from src.index_manager import build_bm25_index
from src.index_manager import save_index, load_index
from src.index_manager import bm25_search, rrf_search
import os
import json
from src.data_models import MinimalAnswer, MinimalSource
from src.data_models import StudentSearchResults, MinimalSearchResults
from src.data_models import StudentSearchResultsAndAnswer
from src.qwen import QwenChatbot
from tqdm import tqdm
from typing import Optional
from src.semantic_embeddings import SemanticIndexing


class RAG:
    """
    CLI class exposing the RAG pipeline commands via Python Fire.

    Commands: index, search, search_dataset, answer,
    answer_dataset, evaluate.
    """
    INDEX_PATH = "data/processed/bm25_index"
    CACHE_PATH = "data/output/cache.json"

    def __init__(self) -> None:
        """
        Init for the chatbot
        """
        self._chatbot: Optional[QwenChatbot] = None
        self._cache: dict[str, dict] = {}

    def _save_cache(self) -> None:
        """
        Save the cache in a json file.
        """
        os.makedirs(os.path.dirname(self.CACHE_PATH), exist_ok=True)
        with open(self.CACHE_PATH, 'w') as f:
            json.dump(self._cache, f, indent=2)

    def _check_cache(self, query: str) -> None | str:
        """
        Check if a query is in the cache and return its answer if its in
        the cache of None if not.
        """
        result = self._cache.get(query)
        if result is None:
            return None
        answer = result.get("answer")
        return str(answer) if answer is not None else None

    def _load_cache(self) -> None:
        """
        Load the cache from disk into memory.
        Silently ignores missing file.
        """
        try:
            with open(self.CACHE_PATH) as f:
                self._cache = json.load(f)
        except Exception:
            pass

    def _get_chatbot(self) -> QwenChatbot:
        """
        Change the chatbot from none to QwenChatBot and return it.
        """
        if self._chatbot is None:
            print("Loading LLM")
            self._chatbot = QwenChatbot()
        return self._chatbot

    def _has_overlap(self, retrieved: dict, truth: dict) -> bool:
        """
        Check if a retrieved source overlaps sufficiently with a
        ground truth source.

        A source is considered found if the overlap covers at
        least 5% of
        the ground truth character range (IoU >= 0.05).

        Args:
            retrieved: Dict with file_path, first_character_index,
                    last_character_index from retrieval results.
            truth: Dict with file_path, first_character_index,
                last_character_index from ground truth.

        Returns:
            True if overlap ratio >= 0.05 and same file, False otherwise.
        """
        if retrieved["file_path"] != truth["file_path"]:
            return False

        overlap_start = max(retrieved["first_character_index"],
                            truth["first_character_index"])
        overlap_end = min(retrieved["last_character_index"],
                          truth["last_character_index"])
        overlap_size = max(0, overlap_end - overlap_start)
        overlap_ratio = overlap_size / (truth["last_character_index"]
                                          - truth["first_character_index"])

        if overlap_ratio >= 0.05:
            return True
        return False

    def index(self, max_chunk_size: int = 2000, hybrid: bool = False) -> None:
        """
        Ingest the vLLM repository and build the BM25
        (and optionally semantic) index.

        Args:
            max_chunk_size: Maximum characters per chunk.
            Must be between 1 and 2000.
            hybrid: If True, also builds the semantic embedding index.
        """

        if max_chunk_size <= 0 or max_chunk_size > 2000:
            print("max chunk size must be superior to 0 and maximum 2000.")
            return

        try:
            print("(BM25) Indexing in progress...")
            chunks = get_all_chunk(max_chunk_size)
            retriever = build_bm25_index(chunks)
            save_index(self.INDEX_PATH, retriever, chunks)
            print("(BM25)Indexing done !")

            if hybrid:
                print("(Semantic) Indexing in progress...")
                semantic = SemanticIndexing()
                semantic.build(chunks)
                print("(Semantic) Indexing done !")

        except Exception as e:
            print(f"Error: {e}")

    def search(self, query: str, k: int = 5, hybrid: bool = False) -> None:
        """
        Search for the top-k most relevant chunks for a single query.

        Args:
            query: The user query string.
            k: Number of results to retrieve.
            hybrid: If True, uses RRF hybrid retrieval (BM25 + semantic).
        """
        if query == '':
            print("Please give a query.")
            return
        if k <= 0:
            print("k must be greater than 0.")
            return

        try:
            retriever, chunks = load_index(self.INDEX_PATH)
            if hybrid:
                semantic = SemanticIndexing()
                chunks_found = rrf_search(
                    query, retriever, semantic, chunks, k)
            else:
                chunks_found = bm25_search(query, retriever, chunks, k)

            for i, chunk in enumerate(chunks_found):
                print(f"Result {i}:")
                print(f"File path: {chunk['file']}")
                print(f"Content:\n{chunk['text']}")

        except Exception as e:
            print(f"Error: {e}")

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str, hybrid: bool = False) -> None:
        """
        Run BM25 search over a full dataset and save results as JSON.

        Args:
            dataset_path: Path to the JSON dataset file
            (UnansweredQuestions format).
            k: Number of chunks to retrieve per question.
            save_directory: Directory where the output JSON file is saved.
            hybrid: If True, uses RRF hybrid retrieval (BM25 + semantic).
        """
        os.makedirs(save_directory, exist_ok=True)
        retriever, chunks = load_index(self.INDEX_PATH)
        mini_search_list = []

        if k <= 0:
            print("k must be greater than 0.")
            return

        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            semantic = SemanticIndexing() if hybrid else NotImplemented

            for d in data["rag_questions"]:
                mini_source = []
                if hybrid:
                    m = rrf_search(d["question"], retriever, semantic,
                                   chunks, k)
                else:
                    m = bm25_search(d["question"], retriever, chunks, k)

                for ans in m:
                    mini_source.append(MinimalSource(
                        file_path=ans["file"],
                        first_character_index=ans["first_char_index"],
                        last_character_index=ans["last_char_index"]))

                mini_search = MinimalSearchResults(
                    question_id=d["question_id"],
                    question=d["question"],
                    retrieved_sources=mini_source)
                mini_search_list.append(mini_search)

            stud_search_res = StudentSearchResults(
                search_results=mini_search_list, k=k)

            a = stud_search_res.model_dump(mode='json')
            filename = os.path.basename(dataset_path)
            output_path = os.path.join(save_directory, filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(a, f, indent=2)

            print(f"Saved to {output_path}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def answer(self, query: str, k: int = 5) -> None:
        """
        Answer a single query using retrieved context and Qwen3-0.6B.

        Checks the cache first. If not cached, retrieves relevant chunks,
        builds a prompt, generates an answer, and caches the result.

        Args:
            query: The user query string.
            k: Number of chunks to retrieve for context.
        """
        if query == '':
            print("Please give a query.")
            return
        if k <= 0:
            print("k must be greater than 0.")
            return

        try:
            self._load_cache()
            answer_in_cache = self._check_cache(query)
            if answer_in_cache:
                print(answer_in_cache)
                return
            to_cache = MinimalAnswer(
                question_id="",
                question=query,
                retrieved_sources=[],
                answer=""
            )
            chatbot = self._get_chatbot()
            documentation = ""
            retriever, chunks = load_index(self.INDEX_PATH)
            chunks_found = bm25_search(query, retriever, chunks, k)

            for chunk in chunks_found:
                mini_source = MinimalSource(
                    file_path=chunk['file'],
                    first_character_index=chunk['first_char_index'],
                    last_character_index=chunk['last_char_index']
                )
                to_cache.retrieved_sources.append(mini_source)
                documentation += chunk['text']

            llm_query = ("Your role: you are an assistant responsible"
                         "for helping the user answer questions. "
                         "To help you, you will be provided with "
                         "information. Use these informations to"
                         " formulate a comprehensible answer. "
                         f"QUERY: {query} INFORMATION: {documentation}")

            response = chatbot.generate_response(llm_query)
            print(response)
            to_cache.answer = response
            jsoned = to_cache.model_dump(mode='json')
            self._cache.update({query: jsoned})
            self._save_cache()

        except Exception as e:
            print(f"Error: {e}")

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        """
        Generate answers for all questions in a search results file.

        Reads a StudentSearchResults JSON file, generates an answer for each
        question using the retrieved sources as context, and saves the output
        as a StudentSearchResultsAndAnswer JSON file.

        Args:
            student_search_results_path: Path to the search results JSON file.
            save_directory: Directory where the output JSON file is saved.
        """
        try:
            self._load_cache()

            os.makedirs(save_directory, exist_ok=True)
            chatbot = self._get_chatbot()

            with open(student_search_results_path, 'r',
                      encoding='utf-8') as f:
                data = json.load(f)
            search_results_and_answer = StudentSearchResultsAndAnswer(
                search_results=[],
                k=data["k"]
            )

            for d in tqdm(data["search_results"],
                          desc="Loading answers for queries..."):
                answer_in_cache = self._check_cache(d["question"])

                mini_answer = MinimalAnswer(
                    question_id=d["question_id"],
                    question=d["question"],
                    retrieved_sources=[],
                    answer=""
                )
                informations = ""

                for source in d["retrieved_sources"]:
                    with open(source["file_path"], encoding='utf-8') as f:
                        info_read = f.read()
                    informations += info_read[
                        source["first_character_index"]:
                        source["last_character_index"]]

                    mini_source = MinimalSource(
                        file_path=source["file_path"],
                        first_character_index=source["first_character_index"],
                        last_character_index=source["last_character_index"]
                    )
                    mini_answer.retrieved_sources.append(mini_source)

                llm_query = ("Your role: you are an assistant"
                             "responsible for helping"
                             " the user answer questions. To help you,"
                             " you will be provided with information."
                             "Use these informations to formulate a "
                             "comprehensible answer. "
                             f"Query: {d['question']} "
                             f"Information: {informations}")

                if answer_in_cache:
                    response = answer_in_cache
                    mini_answer.answer = response
                else:
                    response = chatbot.generate_response(llm_query)
                    mini_answer.answer = response
                    jsoned_cache = mini_answer.model_dump(mode='json')
                    self._cache.update({d["question"]: jsoned_cache})
                search_results_and_answer.search_results.append(mini_answer)

                self._save_cache()

            to_dump = search_results_and_answer.model_dump(mode='json')
            filename = os.path.basename(student_search_results_path)
            output_path = os.path.join(save_directory, filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(to_dump, f, indent=2)
            print(f"Saved to {output_path}")

        except Exception as e:
            print(f"Error: {e}")

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """
        Evaluate retrieval quality by computing recall@k against ground truth.

        Reports recall at k=1, 3, 5, and 10. A source is considered found
        if the retrieved chunk overlaps the correct source by at least 5%.

        Args:
            student_search_results_path: Path to the student
                search results JSON.
            dataset_path: Path to the ground truth AnsweredQuestions JSON.
        """
        try:
            with open(student_search_results_path, encoding='utf-8') as f:
                stud_answers = json.load(f)
            with open(dataset_path, encoding='utf-8') as f:
                true_answers = json.load(f)

            for k in [1, 3, 5, 10]:
                found = 0
                total_sources = 0

                for truth in true_answers["rag_questions"]:
                    stud_result = next(
                        (r for r in stud_answers["search_results"]
                         if r["question_id"] == truth["question_id"]),
                        None
                    )
                    if stud_result is None:
                        continue

                    retrieved_k = stud_result["retrieved_sources"][:k]

                    for correct_source in truth["sources"]:
                        total_sources += 1
                        for retrieved_source in retrieved_k:
                            if self._has_overlap(retrieved_source,
                                                 correct_source):
                                found += 1
                                break

                recall = found / total_sources if total_sources else 0
                print(f"Recall@{k}: {int(recall * 100)}%")

        except Exception as e:
            print(f"Error: {e}")
