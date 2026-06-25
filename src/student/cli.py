from student.files_loader import get_all_chunk
from student.indexation import index_chunks
from student.index_manager import save_index, load_index
from student.search_files import search_match
import os
import json
from student.data_models import MinimalAnswer, MinimalSource, StudentSearchResults, RagDataset, MinimalSearchResults, StudentSearchResultsAndAnswer
from student.qwen import QwenChatbot
from tqdm import tqdm


class RAG:
    index_path = "data/processed/bm25_index"

    def __init__(self):
        self._chatbot = None

    def _get_chatbot(self):
        if self._chatbot is None:
            print("Loading LLM")
            self._chatbot = QwenChatbot()
        return self._chatbot
    
    def _has_overlap(self, retrieved: dict, truth: dict) -> bool:
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
        retriever, chunks = load_index(self.index_path)
        chunks_found = search_match(query, retriever, chunks, k)
        for i, chunk in enumerate(chunks_found):
            print(f"Result {i}:")
            print(f"File path: {chunk['file']}")
            print(f"Content:\n{chunk['text']}")

    def search_dataset(self, dataset_path: str, k: int, save_directory: str) -> None:
        """
        Search matching chunks for a dataset and save it in a file
        dataset_path: path to the dataset containing queries
        k: number of chunks to retrieve for each query
        save_directory: path where to save the result
        """
        os.makedirs(save_directory, exist_ok=True)
        retriever, chunks = load_index(self.index_path)
        mini_search_list = []
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        for d in data["rag_questions"]:
            mini_source = []
            m = search_match(d["question"], retriever, chunks, k)
            for ans in m:
                mini_source.append(MinimalSource(file_path=ans["file"], first_character_index=ans["first_char_index"],
                                                 last_character_index=ans["last_char_index"]))
                
            mini_search = MinimalSearchResults(question_id=d["question_id"], question=d["question"],
                                                retrieved_sources=mini_source)
            mini_search_list.append(mini_search)
        stud_search_res = StudentSearchResults(search_results=mini_search_list, k=k)
        a = stud_search_res.model_dump(mode='json')
        filename = os.path.basename(dataset_path)
        output_path = os.path.join(save_directory, filename)
        with open(output_path, 'w') as f:
            json.dump(a, f, indent=2)
        print(f"Saved to {output_path}")

    def answer(self, query: str, k: int = 5) -> None:
        """
        Use QWEN and chunks to answer a single query of the user
        query: user query
        k: number of chunks to retrieve for the query
        """
        # lecture index -> cherche match -> construction query pour llm -> envoie llm
        chatbot = self._get_chatbot()
        documentation = ""
        retriever, chunks = load_index(self.index_path)
        chunks_found = search_match(query, retriever, chunks, k)
        for chunk in chunks_found:
            documentation += chunk['text']
        print(documentation)
        llm_query = ("Your role: you are an assistant responsible for helping"
                     " the user answer questions. To help you, you will be"
                     " provided with information. Use these informations to"
                     " formulate a comprehensible answer. "
                     f"QUERY: {query} INFORMATION: {documentation}")
        
        response = chatbot.generate_response(llm_query)
        print(response)

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        """
        Use QWEN and chunks to answer questions from a dataset
        student_search_results_path: Path to the search results
        save_directory: path where to save the answers
        """
        os.makedirs(save_directory, exist_ok=True)
        chatbot = self._get_chatbot()

        with open(student_search_results_path, 'r') as f:
            data = json.load(f)
        search_results_and_answer = StudentSearchResultsAndAnswer(
            search_results = [],
            k = data["k"]
        )

        for d in tqdm(data["search_results"], desc="Loading answers for queries..."):
            mini_answer = MinimalAnswer(
                question_id = d["question_id"],
                question = d["question"],
                retrieved_sources = [],
                answer = ""
            )
            informations = ""

            for source in d["retrieved_sources"]:
                with open(source["file_path"]) as f:
                    info_read = f.read()
                informations += info_read[
                    source["first_character_index"]
                    :source["last_character_index"]]
                
                mini_source = MinimalSource(
                    file_path = source["file_path"],
                    first_character_index = source["first_character_index"],
                    last_character_index = source["last_character_index"]
                )
                mini_answer.retrieved_sources.append(mini_source)

            llm_query = ("Your role: you are an assistant responsible for helping"
                        " the user answer questions. To help you,"
                        " you will be provided with information. Use these"
                        " informations to formulate a comprehensible answer. "
                        f"Query: {d['question']}  Information: {informations}")
            
            response = chatbot.generate_response(llm_query)
            mini_answer.answer = response
            search_results_and_answer.search_results.append(mini_answer)

        dumped = search_results_and_answer.model_dump(mode='json')
        filename = os.path.basename(student_search_results_path)
        output_path = os.path.join(save_directory, filename)
        with open(output_path, 'w') as f:
            json.dump(dumped, f, indent=2)
        print(f"Saved to {output_path}")

    def evaluate(self, student_answer_path: str, dataset_path: str) -> None:
        count = 0
        with open(student_answer_path) as f:
            stud_answers = json.load(f)
        with open(dataset_path) as f:
            true_answers = json.load(f)
        for truth in true_answers["rag_questions"]:
            for file in truth["sources"]:
                for retrieved in stud_answers["search_results"]:
                    for source in retrieved["retrieved_sources"]:
                        if self._has_overlap(source, file):
                            count += 1
                            break
        
        print(count)
        
