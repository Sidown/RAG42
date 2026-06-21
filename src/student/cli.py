from student.files_loader import get_all_chunk
from student.indexation import index_chunks
from student.index_manager import save_index, load_index
from student.search_files import search_match


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

    def search_dataset(self):
        pass

    def answer(self):
        pass

    def answer_dataset(self):
        pass

    def evaluate(self):
        pass
