# app/chain.py

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.vectorstore import get_retriever
from app.config import GROQ_API_KEY

# 1️⃣ Load the retriever
retriever = get_retriever()

# 2️⃣ Define the prompt template
prompt_template = """
You are a helpful AI assistant. Answer the user's question based only on the provided context.

Context:
{context}

Question:
{question}

If the answer is not in the context, say you don't know.
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# 3️⃣ Initialize the Groq LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="mixtral-8x7b-32768",  # You can change to other Groq-supported models
    temperature=0
)

# 4️⃣ Build the RAG chain
rag_chain = RunnableMap({
    "context": lambda x: "\n\n".join(
        [doc.page_content for doc in retriever.get_relevant_documents(x["question"])]
    ),
    "question": RunnablePassthrough()
}) | prompt | llm | StrOutputParser()

# 5️⃣ Function to run the chain
def get_answer(question: str) -> str:
    """
    Given a user question, retrieve relevant context and return the model's answer.
    """
    return rag_chain.invoke({"question": question})
