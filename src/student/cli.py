from student.files_loader import get_all_chunk
from student.indexation import index_chunks
from student.index_manager import save_index, load_index
from student.search_files import search_match
import os
import json
from student.data_models import MinimalSource, StudentSearchResults, RagDataset, MinimalSearchResults


class RAG:
    path = "data/processed/bm25_index"

    def index(self, max_chunk_size: int = 2000):
        print("Indexing in progress...")
        chunks = get_all_chunk(max_chunk_size)
        retriever = index_chunks(chunks)
        save_index(self.path, retriever, chunks)
        print("Indexing done !")

    def search(self, query: str, k: int = 5):
        retriever, chunks = load_index(self.path)
        chunks_found = search_match(query, retriever, chunks, k)
        for i, chunk in enumerate(chunks_found):
            print(f"Result {i}:")
            print(f"File path: {chunk['file']}")
            print(f"Content:\n{chunk['text']}")

    def search_dataset(self, dataset_path: str, k: int, save_directory: str):
        os.makedirs(os.path.dirname(save_directory), exist_ok=True)
        retriever, chunks = load_index(self.path)
        mini_search_list = []
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        for d in data["rag_questions"]:
            # print(d["question"])
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
        print(a)

    def answer(self):
        pass

    def answer_dataset(self):
        pass

    def evaluate(self):
        pass
