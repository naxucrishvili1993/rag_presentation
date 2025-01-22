from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from embedding_function import embedding_function

CHROMA_PATH = '../../../rag_presentation/chroma'
DATA_PATH = 'pdfs'


def main():
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function())
    print("Adding documents to chroma")
    db.add_documents(chunks)


if __name__ == '__main__':
    main()
