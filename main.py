from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)

vector_store_initialized = False
initialization_error = None


def extract_text_with_pages(pdf_path):
    documents = []
    try:
        with open(pdf_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"page_number": i + 1}
                    ))
    except Exception:
        pass
    return documents


def initialize_vector_store():
    global vector_store_initialized, initialization_error
    try:
        pdf_path = "example.pdf"
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")

        docs = extract_text_with_pages(pdf_path)
        if not docs:
            raise ValueError("No extractable text in PDF")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
        if not split_docs:
            raise ValueError("No valid text chunks created")

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(split_docs, embedding=embeddings)
        vector_store.save_local("faiss_index")
        vector_store_initialized = True
    except Exception as e:
        initialization_error = str(e)
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        initialize_vector_store()
        yield
    except Exception:
        yield


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
async def serve_frontend():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
async def health_check():
    if initialization_error:
        return {
            "status": "error",
            "initialized": False,
            "error": initialization_error
        }
    return {
        "status": "ok",
        "initialized": vector_store_initialized
    }


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        if not vector_store_initialized:
            if initialization_error:
                return {
                    "status": "error",
                    "answer": f"Cannot process questions: {initialization_error}"
                }
            raise HTTPException(status_code=400, detail="PDF processing not completed yet")

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(request.question)

        chain = get_conversational_chain()
        response = chain.invoke({
            "input_documents": docs,
            "question": request.question
        })

        pages = sorted({doc.metadata.get("page_number") for doc in docs if "page_number" in doc.metadata})
        page_info = f"\n\n\ud83d\udcc4 Referenced page(s): {', '.join(map(str, pages))}" if pages else ""

        return {
            "status": "success",
            "answer": response["output_text"] + page_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


def get_conversational_chain():
    prompt_template = """
    Answer the question using the context below. If unsure, say you don't know.
    Context: {context}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        temperature=0.3
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


if __name__ == "__main__":
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)

    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
