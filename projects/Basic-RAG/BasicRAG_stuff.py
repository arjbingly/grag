from grag.grag.rag import BasicRAG

rag = BasicRAG(doc_chain="stuff")

if __name__ == "__main__":
    while True:
        query = input("Query:")
        rag(query)
