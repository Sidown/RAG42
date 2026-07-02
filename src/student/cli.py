from student.chunker import get_all_chunk
from student.index_manager import index_chunks
from student.index_manager import save_index, load_index
from student.index_manager import search_match
import os
import json
from student.data_models import MinimalAnswer, MinimalSource
from student.data_models import StudentSearchResults, MinimalSearchResults
from student.data_models import StudentSearchResultsAndAnswer
from student.qwen import QwenChatbot
from tqdm import tqdm
from typing import Optional


class RAG:
    """
    CLI class with index, search, search_dataset, answer, answer_dataset
    and evaluate as command.
    """
    index_path = "data/processed/bm25_index"
    cache_path = "data/output/cache.json"

    def __init__(self) -> None:
        """
        Init for the chatbot
        """
        self._chatbot: Optional[QwenChatbot] = None
        self._cache: dict[str, list[dict[str, str | int]]] = {}

    def _save_cache(self) -> None:
        with open(self.cache_path, 'w') as f:
            json.dump(self._cache, f, indent=2)


    def _check_cache(self, query: str) -> None | dict[str, str | int]:
        for key in self._cache:
            if key == query:
                return self._cache[key]
        return None
    
    def _load_cache(self) -> None:
        try:
            with open(self.cache_path) as f:
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
        Check if the text of the retrieved information and the true information
        overlap
        """
        if retrieved["file_path"] != truth["file_path"]:
            return False

        overlap_start = max(retrieved["first_character_index"],
                            truth["first_character_index"])
        overlap_end = min(retrieved["last_character_index"],
                          truth["last_character_index"])
        overlap_size = max(0, overlap_end - overlap_start)
        overlap_percent = overlap_size / (truth["last_character_index"]
                                          - truth["first_character_index"])

        if overlap_percent >= 0.05:
            return True
        return False

    def index(self, max_chunk_size: int = 2000) -> None:
        """
        Index files
        max_chunk_size: number of char in a chunk
        """
        print("Indexing in progress...")
        chunks = get_all_chunk(max_chunk_size)
        retriever = index_chunks(chunks)
        save_index(self.index_path, retriever, chunks)
        print("Indexing done !")

    def search(self, query: str, k: int = 5) -> None:
        """
        Search chunk match for a query
        query: user query
        k: number max of matching chunks to retrieve
        """
        if query == '':
            print("Please give a query.")
            return

        retriever, chunks = load_index(self.index_path)
        chunks_found = search_match(query, retriever, chunks, k)

        for i, chunk in enumerate(chunks_found):
            print(f"Result {i}:")
            print(f"File path: {chunk['file']}")
            print(f"Content:\n{chunk['text']}")

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str) -> None:
        """
        Search matching chunks for a dataset and save it in a file
        dataset_path: path to the dataset containing queries
        k: number of chunks to retrieve for each query
        save_directory: path where to save the result
        """
        os.makedirs(save_directory, exist_ok=True)
        retriever, chunks = load_index(self.index_path)
        mini_search_list = []

        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for d in data["rag_questions"]:
                mini_source = []
                m = search_match(d["question"], retriever, chunks, k)

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

        except Exception:
            print("Please check your arguments:")
            print("-> uv run python -m student"
                  " search_dataset {dataset_path} {k} {save_directory}")

    def answer(self, query: str, k: int = 5) -> None:
        """
        Use QWEN and chunks to answer a single query of the user
        query: user query
        k: number of chunks to retrieve for the query
        """
        if query == '':
            print("Please give a query.")
            return

        self._load_cache()
        answer_in_cache = self._check_cache(query)
        if answer_in_cache:
            print(answer_in_cache)
            return
        
        chatbot = self._get_chatbot()
        documentation = ""
        retriever, chunks = load_index(self.index_path)
        chunks_found = search_match(query, retriever, chunks, k)

        for chunk in chunks_found:
            documentation += chunk['text']

        llm_query = ("Your role: you are an assistant responsible for helping"
                     " the user answer questions. To help you, you will be"
                     " provided with information. Use these informations to"
                     " formulate a comprehensible answer. "
                     f"QUERY: {query} INFORMATION: {documentation}")

        response = chatbot.generate_response(llm_query)
        print(response)
        self._cache.update({query: response})
        self._save_cache()

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        """
        Use QWEN and chunks to answer questions from a dataset
        student_search_results_path: Path to the search results
        save_directory: path where to save the answers
        """
        try:
            self._load_cache()
            # answer_in_cache = self._check_cache(query)
            # if answer_in_cache:
            #     print(answer_in_cache)
            #     return

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
                else:
                    response = chatbot.generate_response(llm_query)
                    self._cache.update({d["question"]: response})
                mini_answer.answer = response
                search_results_and_answer.search_results.append(mini_answer)

            dumped = search_results_and_answer.model_dump(mode='json')
            filename = os.path.basename(student_search_results_path)
            output_path = os.path.join(save_directory, filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dumped, f, indent=2)
            print(f"Saved to {output_path}")
            with open(self.cache_path, 'w') as f:
                json.dump(self._cache, f, indent=2)


        except Exception:
            print("Answer dataset failed, please check your arguments:")
            print("-> uv run python -m student answer_dataset"
                  " {student_search_results_path}"
                  " {save_directory}")

    def evaluate(self, student_answer_path: str, dataset_path: str) -> None:
        """
        Evaluates the student's sources by comparing them with the real
        sources and returns a score for the first 1, 3, 5 and 10
        sources found.
        student_answer_path: path to the student file
        dataset_path: path to the comparison file
        """
        try:
            with open(student_answer_path, encoding='utf-8') as f:
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

        except Exception:
            print("Evaluation failed, please check your arguments:")
            print("-> uv run python -m student"
                  " evaluate {student_answer_path} {dataset_path}")
