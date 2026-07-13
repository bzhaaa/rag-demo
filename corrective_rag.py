from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from typing import Any, Dict, TypedDict
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import pprint
import nest_asyncio
import tempfile
import os
import time


nest_asyncio.apply()
load_dotenv(override=True)

retriever = None


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_langsmith() -> tuple[bool, str]:
    enabled = env_flag("LANGSMITH_TRACING")
    project = os.getenv("LANGSMITH_PROJECT", "corrective-rag").strip()

    if not enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ.pop("LANGCHAIN_API_KEY", None)
        os.environ.pop("LANGCHAIN_PROJECT", None)
        os.environ.pop("LANGCHAIN_ENDPOINT", None)
        return False, project

    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        st.error("LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is missing")
        st.stop()

    # LangChain 0.3 uses the legacy LANGCHAIN_* names for callback tracing.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project

    endpoint = os.getenv("LANGSMITH_ENDPOINT", "").strip()
    if endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint

    return True, project


LANGSMITH_ENABLED, LANGSMITH_PROJECT = configure_langsmith()
LANGSMITH_HIDE_INPUTS = env_flag("LANGSMITH_HIDE_INPUTS")
LANGSMITH_HIDE_OUTPUTS = env_flag("LANGSMITH_HIDE_OUTPUTS")


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        st.error(f"Missing required environment variable: {name}")
        st.stop()
    return value


def normalize_openai_base_url(url: str) -> str:
    normalized = url.rstrip("/")
    for suffix in ("/chat/completions", "/embeddings"):
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
    return normalized


LLM_API_KEY = require_env("LLM_API_KEY")
LLM_BASE_URL = normalize_openai_base_url(require_env("LLM_BASE_URL"))
LLM_MODEL = require_env("LLM_MODEL")
EMBEDDING_API_KEY = require_env("EMBEDDING_API_KEY")
EMBEDDING_BASE_URL = normalize_openai_base_url(require_env("EMBEDDING_BASE_URL"))
EMBEDDING_MODEL = require_env("EMBEDDING_MODEL")

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost").strip()
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530").strip()
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "corrective_rag").strip()

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=EMBEDDING_API_KEY,
    base_url=EMBEDDING_BASE_URL,
    # This compatible endpoint accepts strings, not pre-tokenized integer arrays.
    check_embedding_ctx_length=False,
    chunk_size=10,
)

def create_chat_model(max_tokens: int = 1000):
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0,
        max_completion_tokens=max_tokens,
    )

def get_milvus_connection_args() -> dict:
    return {
        "host": MILVUS_HOST,
        "port": MILVUS_PORT,
    }


def initialize_app_state() -> None:
    defaults = {
        "retriever": None,
        "knowledge_source": None,
        "chunk_count": 0,
        "ingestion_seconds": None,
        "rag_steps": [],
        "final_generation": None,
        "question_seconds": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_app_state()


def web_search(state):
    """Add deterministic mock search context without making a network request."""
    print("~-mock web search-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    mock_document = Document(
        page_content=(
            "Mock web search result. No external search request was made. "
            f"The simulated search query was: {question}. "
            "This placeholder exists only to exercise the corrective RAG web-search "
            "branch and must not be treated as verified external evidence."
        ),
        metadata={
            "source": "mock_web_search",
            "title": "Mock web search result",
            "query": question,
            "is_mock": True,
        },
    )
    documents.append(mock_document)
    st.info("Mock web search added one local placeholder result.")

    return {"keys": {"documents": documents, "question": question}}


def load_documents(file_path: str) -> list:
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension in [".txt", ".md"]:
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        return loader.load()
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return []

def build_vector_store(docs: list, source_name: str) -> None:
    global retriever
    started_at = time.perf_counter()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    all_splits = text_splitter.split_documents(docs)
    if not all_splits:
        raise ValueError("No text chunks were extracted from the document")

    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64},
    }
    search_params = {"metric_type": "COSINE", "params": {"ef": 16}}

    try:
        vectorstore = Milvus.from_documents(
            documents=all_splits,
            embedding=embeddings,
            collection_name=MILVUS_COLLECTION,
            connection_args=get_milvus_connection_args(),
            index_params=index_params,
            search_params=search_params,
            drop_old=True,
            metadata_field="metadata",
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        st.session_state.retriever = retriever
        st.session_state.knowledge_source = source_name
        st.session_state.chunk_count = len(all_splits)
        st.session_state.ingestion_seconds = time.perf_counter() - started_at
        st.session_state.rag_steps = []
        st.session_state.final_generation = None
        st.session_state.question_seconds = None
    except Exception:
        retriever = None
        st.session_state.retriever = None
        raise


st.subheader("Document Input")

with st.form("document_ingestion_form"):
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "md"],
    )
    load_document = st.form_submit_button(
        "Load into knowledge base",
        type="primary",
        use_container_width=True,
    )

if load_document:
    with st.status("Preparing knowledge base...", expanded=True) as status:
        try:
            if uploaded_file is None:
                raise ValueError("Please upload a document")

            status.write("Reading the uploaded document...")
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            try:
                docs = load_documents(temp_path)
            finally:
                os.unlink(temp_path)
            source_name = uploaded_file.name

            if not docs:
                raise ValueError("The document could not be loaded or contained no text")

            status.write("Creating embeddings and rebuilding the Milvus collection...")
            build_vector_store(docs, source_name)
            status.update(label="Knowledge base ready", state="complete")
        except Exception as e:
            status.update(label="Knowledge base loading failed", state="error")
            st.error(f"Error initializing local Milvus vector store: {str(e)}")

if st.session_state.retriever is not None:
    retriever = st.session_state.retriever
    st.success(
        f"Knowledge base ready: {st.session_state.chunk_count} chunks from "
        f"'{st.session_state.knowledge_source}' in "
        f"{st.session_state.ingestion_seconds:.1f}s"
    )
else:
    st.info("Load a document into the knowledge base before asking questions.")


class GraphState(TypedDict):
    keys: Dict[str, Any]


def retrieve(state):
    print("~-retrieve-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    
    if retriever is None:
        return {"keys": {"documents": [], "question": question}}
        
    documents = retriever.get_relevant_documents(question)
    return {"keys": {"documents": documents, "question": question}}


def generate(state):
    """Generate an answer with the configured OpenAI-compatible chat model."""
    print("~-generate-~")
    state_dict = state["keys"]
    question, documents = state_dict["question"], state_dict["documents"]
    try:
        prompt = PromptTemplate(template="""Based on the following context, please answer the question.
            Context: {context}
            Question: {question}
            Answer:""", input_variables=["context", "question"])
        llm = create_chat_model(max_tokens=1000)
        context = "\n\n".join(doc.page_content for doc in documents)

        # Create and run chain
        rag_chain = (
            {"context": lambda x: context, "question": lambda x: question} 
            | prompt 
            | llm 
            | StrOutputParser()
        )

        generation = rag_chain.invoke({})

        return {
            "keys": {
                "documents": documents,
                "question": question,
                "generation": generation
            }
        }

    except Exception as e:
        error_msg = f"Error in generate function: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return {"keys": {"documents": documents, "question": question, 
                "generation": "Sorry, I encountered an error while generating the response."}}

def grade_documents(state):
    """Determines whether the retrieved documents are relevant."""
    print("~-check relevance-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    llm = create_chat_model(max_tokens=1000)

    prompt = PromptTemplate(template="""You are grading the relevance of a retrieved document to a user question.
        Return ONLY a JSON object with a "score" field that is either "yes" or "no".
        Do not include any other text or explanation.
        
        Document: {context}
        Question: {question}
        
        Rules:
        - Check for related keywords or semantic meaning
        - Use lenient grading to only filter clear mismatches
        - Return exactly like this example: {{"score": "yes"}} or {{"score": "no"}}""",
        input_variables=["context", "question"])

    chain = (
        prompt 
        | llm 
        | StrOutputParser()
    )

    filtered_docs = []
    search = "Yes" if not documents else "No"
    
    for d in documents:
        try:
            response = chain.invoke({"question": question, "context": d.page_content})
            import re
            json_match = re.search(r'\{.*\}', response)
            if json_match:
                response = json_match.group()
            
            import json
            score = json.loads(response)
            
            if score.get("score") == "yes":
                print("~-grade: document relevant-~")
                filtered_docs.append(d)
            else:
                print("~-grade: document not relevant-~")
                search = "Yes"
                
        except Exception as e:
            print(f"Error grading document: {str(e)}")
            # On error, keep the document to be safe
            filtered_docs.append(d)
            continue

    return {"keys": {"documents": filtered_docs, "question": question, "run_web_search": search}}


def transform_query(state):
    """Transform the query to produce a better question."""
    print("~-transform query-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Create a prompt template
    prompt = PromptTemplate(
        template="""Generate a search-optimized version of this question by 
        analyzing its core semantic meaning and intent.
        \n ------- \n
        {question}
        \n ------- \n
        Return only the improved question with no additional text:""",
        input_variables=["question"],
    )

    llm = create_chat_model(max_tokens=1000)

    # Prompt
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})

    return {
        "keys": {"documents": documents, "question": better_question}
    }


def decide_to_generate(state):
    print("~-decide to generate-~")
    state_dict = state["keys"]
    search = state_dict["run_web_search"]

    if search == "Yes":
     
        print("~-decision: transform query and run web search-~")
        return "transform_query"
    else:
        print("~-decision: generate-~")
        return "generate"
    
def format_document(doc: Document) -> str:
    return f"""
    Source: {doc.metadata.get('source', 'Unknown')}
    Title: {doc.metadata.get('title', 'No title')}
    Content: {doc.page_content[:200]}...
    """

def format_state(state: dict) -> str:
    formatted = {}
    
    for key, value in state.items():
        if key == "documents":
            formatted[key] = [format_document(doc) for doc in value]
        else:
            formatted[key] = value
            
    return formatted


workflow = StateGraph(GraphState)

# Define the nodes by langgraph
workflow.add_node("retrieve", retrieve) 
workflow.add_node("grade_documents", grade_documents)  
workflow.add_node("generate", generate) 
workflow.add_node("transform_query", transform_query)  
workflow.add_node("web_search", web_search) 

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

st.title("Corrective RAG Agent - .env + Milvus")

with st.sidebar:
    st.subheader("Runtime Configuration")
    st.text(f"LLM model: {LLM_MODEL}")
    st.text(f"Embedding model: {EMBEDDING_MODEL}")
    st.text(f"Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    st.text(f"Collection: {MILVUS_COLLECTION}")
    st.text(
        f"LangSmith: {'Enabled' if LANGSMITH_ENABLED else 'Disabled'}"
    )
    if LANGSMITH_ENABLED:
        st.text(f"LangSmith project: {LANGSMITH_PROJECT}")
        st.text(
            "Trace privacy: "
            f"inputs {'hidden' if LANGSMITH_HIDE_INPUTS else 'visible'}, "
            f"outputs {'hidden' if LANGSMITH_HIDE_OUTPUTS else 'visible'}"
        )
    st.caption("API keys and base URLs are loaded from .env.")
    st.caption("Web search is running in deterministic mock mode.")

st.text("A possible query: What are the experiment results and ablation studies in this research paper?")

with st.form("question_form"):
    user_question = st.text_input("Please enter your question:")
    ask_question = st.form_submit_button(
        "Ask",
        type="primary",
        disabled=st.session_state.retriever is None,
        use_container_width=True,
    )

if ask_question:
    inputs = {
        "keys": {
            "question": user_question,
        }
    }
    run_config = {
        "run_name": "corrective-rag-workflow",
        "tags": ["corrective-rag", "streamlit", "mock-web-search"],
        "metadata": {
            "llm_model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "milvus_collection": MILVUS_COLLECTION,
            "web_search_mode": "mock",
        },
    }

    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        started_at = time.perf_counter()
        steps = []
        with st.status("Running corrective RAG...", expanded=True) as status:
            try:
                for output in app.stream(inputs, config=run_config):
                    for key, value in output.items():
                        status.write(f"Completed step: {key}")
                        steps.append((key, value["keys"]))

                st.session_state.rag_steps = steps
                st.session_state.final_generation = steps[-1][1].get(
                    "generation",
                    "No final generation produced.",
                )
                st.session_state.question_seconds = time.perf_counter() - started_at
                status.update(label="Corrective RAG complete", state="complete")
            except Exception as e:
                status.update(label="Corrective RAG failed", state="error")
                st.error(f"Error running corrective RAG: {str(e)}")

for key, step_state in st.session_state.rag_steps:
    with st.expander(f"Step '{key}':"):
        st.text(pprint.pformat(format_state(step_state), indent=2, width=80))

if st.session_state.final_generation is not None:
    st.subheader("Final Generation:")
    st.write(st.session_state.final_generation)
    st.caption(f"Question completed in {st.session_state.question_seconds:.1f}s")
