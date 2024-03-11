from src.rag.basic_rag import BasicRAG

rag = BasicRAG(doc_chain="stuff")

if __name__ == "__main__":
    while True:
        query = input("Query:")
        rag(query)
