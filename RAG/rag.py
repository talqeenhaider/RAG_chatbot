import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)




persist_directory = "docs/chroma"

# vectordb = Chroma(
#     persist_directory=persist_directory,
#     embedding_function=embeddings
# )


# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("SECRET_KEY"),
    temperature=0.2
)


# ============================================================
# RETRIEVER
# ============================================================

def create_retriever(vectordb):

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(
            search_kwargs={"k": 5}
        ),
        llm=llm
    )

    return retriever



# ============================================================
# FORMAT DOCUMENTS
# ============================================================

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )



# ============================================================
# RAG CHAIN
# ============================================================

def create_rag_chain(retriever, llm):

    prompt = PromptTemplate(
        template="""
You are a knowledgeable AI assistant that answers questions strictly based on the uploaded document.

Rules:

1. Carefully read the entire retrieved context below before answering. The answer may be phrased differently in the document than in the user's question — look for the underlying meaning, related terms, synonyms, or indirect mentions, not just exact keyword matches.
2. If the context contains information that answers the question — even partially or indirectly — synthesize it into a clear, complete answer using only that information.
3. Do not use your own knowledge, assumptions, or any outside information beyond what is in the context.
4. Only respond with "I couldn't find this information in the document." if, after careful reading, the context truly contains nothing relevant to the question.
5. Do not invent facts, numbers, names, or details that are not present in the context.
6. Do not mention RAG, retrieval, chunks, embeddings, or vector databases.
7. Write the final answer in one clear paragraph.
# 7. Write the final answer as a well-structured paragraph (or a short list if that improves clarity).
# 8. If the user's question is a general greeting or unrelated to any document (e.g. "hello", "who are you"), respond naturally and helpfully without referencing "the document."


Retrieved Context:
{context}

User Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    answer_chain = prompt | llm | StrOutputParser()
        # ========================================================
    # RUN FUNCTION — returns answer + source page numbers
    # ========================================================

    def run(question):

        docs = retriever.invoke(question)

        context = format_docs(docs)

        # Extract unique page numbers (PyPDFLoader pages are 0-indexed)
        pages = sorted({
            doc.metadata.get("page") + 1
            for doc in docs
            if doc.metadata.get("page") is not None
        })

        answer = answer_chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "answer": answer,
            "pages": pages
        }

    return run




    # parallel_chain = RunnableParallel({
    #     "context": retriever | RunnableLambda(format_docs),
    #     "question": RunnablePassthrough()
    # })

    # main_chain = (
    #     parallel_chain
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    # return main_chain

