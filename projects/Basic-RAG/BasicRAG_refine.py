from src.grag.rag import BasicRAG

rag = BasicRAG(doc_chain="refine")

if __name__ == "__main__":
    while True:
        query = input("Query:")
        rag(query)
