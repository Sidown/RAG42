from student.files_loader import get_all_chunk
from student.indexation import index_chunks
from student.index_manager import save_index, load_index
from student.search_files import search_match
import os
import json
from student.data_models import MinimalSource, StudentSearchResults, RagDataset, MinimalSearchResults
from student.qwen import QwenChatbot


class RAG:
    index_path = "data/processed/bm25_index"

    def __init__(self):
        self._chatbot = None

    def _get_chatbot(self):
        if self._chatbot is None:
            print("Loading LLM")
            self._chatbot = QwenChatbot()
        return self._chatbot

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
        doc_path = "data/processed/chunks.json"
        retriever, chunks = load_index(self.index_path)
        chunks_found = search_match(query, retriever, chunks, k)
        for chunk in chunks_found:
            print(chunk)
        llm_query = ("Your role: you are an assistant responsible for helping"
                     " the user answer questions. To help you, you will be"
                     " provided with information. Use these informations to"
                     " formulate a comprehensible answer. "
                     f"QUERY: {query} INFORMATION: {chunks_found}")
        
        
        response = chatbot.generate_response(llm_query)
        print(response)

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        """
        Use QWEN and chunks to answer questions from a dataset
        student_search_results_path: Path to the search results
        save_directory: path where to save the answers
        """
        with open(student_search_results_path) as f:
            data = f.read()
        print(data)
        
        llm_query = [
            "Your role: you are an assistant responsible for helping the user answer questions. To help you,"
            " you will be provided with information. Use these informations to formulate a comprehensible answer."
        ]

    def evaluate(self):
        pass
