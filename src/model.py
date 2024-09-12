import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp
import re


print("Llamacpp================", LlamaCpp)
def clean_content(content: str) -> str:
    # Replace one or more tab characters with a single space
    return re.sub(r"\t+", " ", content)


from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor


# Define Chunk class
class Chunk:
    def __init__(self, page_content, start_index=None):
        self.page_content = page_content
        self.start_index = start_index

    def to_dict(self):
        return {"page_content": self.page_content, "start_index": self.start_index}

    @staticmethod
    def from_dict(data):
        return Chunk(
            page_content=data.get("page_content", ""),
            start_index=data.get("start_index"),
        )
       

# Define prompt template
PROMPT_TEMPLATE = """
Given the following document context in JSON format, answer the question. Do not mention chunk details and answer only from the provided context. Try to answer in a maximum word limit of 300 words. If the question is out of context or cannot be answered with the provided information, simply return: "The question you asked is out of context."

Context: {context}
Question: {question}

Answer:

"""


# Generate sample text for LLM check
def generate_sample_prompt(question, context_json):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format_messages(context=context_json, question=question)
    return prompt


# Function to perform similarity search
def similarity_search(index, query_embedding, k=7):
    query_embedding = np.array(query_embedding, dtype=np.float32)
    D, I = index.search(query_embedding[np.newaxis], k)
    return I[0], D[0]


def initBookProcess():
    # try:
    # Constants
    PDF_FILE_NAME = "budget_speech.pdf"
    DB_FILE_NAME = os.path.splitext(os.path.basename(PDF_FILE_NAME))[0]
    INDEX_FILE = f"{DB_FILE_NAME}.index"
    EMBEDDINGS_FILE = "embeddings.npy"
    CHUNKS_FILE = "chunks.json"
    MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    LLM_MODEL_NAME = "zhentaoyu/Llama-2-7b-chat-hf-Q4_K_S-GGUF"
    LLM_MODEL_FILE = "llama-2-7b-chat-hf-q4_k_s.gguf"
    LLM_MODEL_LOCAL_PATH = "./llama-2-7b-chat-hf-q4_k_s.gguf"

    # Initialize model
    model = SentenceTransformer(MODEL_NAME)

    # Function to extract text and split into chunks
    def extract_and_split_text(pdf_file, chunk_size=400, chunk_overlap=200):
        # print("kkkkkkkk======================")
        loader = PyPDFLoader(pdf_file)
        doc = loader.load_and_split()
        # print("doc============",doc)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(doc)
        chunks = [
            Chunk(page_content=clean_content(chunk.page_content))
            for chunk in chunks
            if 0 < len(chunk.page_content) < chunk_size
        ]
        print(f"Extracted {len(chunks)} chunks.")
        return chunks

    # Function to save chunks
    def save_chunks(chunks, file_name):
        with open(file_name, "w") as f:
            json.dump([chunk.to_dict() for chunk in chunks], f, indent=2)
        print(f"Saved {len(chunks)} chunks to {file_name}.")

    # Function to load chunks
    def load_chunks(file_name):
        if not os.path.exists(file_name) or os.path.getsize(file_name) == 0:
            return []
        with open(file_name, "r") as f:
            chunk_dicts = json.load(f)
        chunks = [Chunk.from_dict(d) for d in chunk_dicts]
        print(f"Loaded {len(chunks)} chunks from {file_name}.")
        return chunks

    # Function to create embeddings
    def create_embeddings(chunks, model):
        print("Creating embeddings...")

        def encode_chunk(chunk):
            return (
                model.encode(chunk.page_content, convert_to_tensor=True).cpu().numpy()
            )
            print()

        with ThreadPoolExecutor() as executor:
            embeddings_list = list(executor.map(encode_chunk, chunks))

        embeddings = np.vstack(embeddings_list)
        print("Embeddings created.")
        return embeddings

    # Function to save embeddings
    def save_embeddings(embeddings, file_name):
        np.save(file_name, embeddings)
        print(f"Saved embeddings to {file_name}.")

    # Function to load embeddings
    def load_embeddings(file_name):
        return np.load(file_name)

    # Function to create FAISS index
    def create_faiss_index(embeddings, index_file):
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, index_file)
        print(f"Created FAISS index and saved to {index_file}.")
        return index

    # Function to load FAISS index
    def load_faiss_index(index_file):
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file does not exist: {index_file}")
        index = faiss.read_index(index_file)
        print(f"Loaded FAISS index from {index_file}.")
        return index

    # Check if the LLM model file exists locally
    if not os.path.exists(LLM_MODEL_LOCAL_PATH):
        model_file = hf_hub_download(
            repo_id=LLM_MODEL_NAME, filename=LLM_MODEL_FILE, local_dir="./"
        )
    else:
        model_file = LLM_MODEL_LOCAL_PATH
    print("llmmm==================")
    llm:any
    try:
    # Initialize LLM model
        llm = LlamaCpp(
            model_path=model_file,
            n_gpu_layers=-1,
            n_batch=512,
            temperature=0.5,
            n_ctx=10000,
        )
    except Exception as e:
        print("error====--------------------------------=",e)

    # Load or extract chunks
    if os.path.exists(CHUNKS_FILE):
        chunks = load_chunks(CHUNKS_FILE)
    else:
        chunks = extract_and_split_text(PDF_FILE_NAME)
        save_chunks(chunks, CHUNKS_FILE)

    # Load or create embeddings
    if os.path.exists(EMBEDDINGS_FILE):
        embeddings = load_embeddings(EMBEDDINGS_FILE)
    else:
        embeddings = create_embeddings(chunks, model)
        save_embeddings(embeddings, EMBEDDINGS_FILE)
    
    # Create or load FAISS index
    if not os.path.exists(INDEX_FILE):
        index = create_faiss_index(embeddings, INDEX_FILE)         
    else:
        index = load_faiss_index(INDEX_FILE)
    return {"model": model, "index": index, "chunks": chunks, "llm": llm}
