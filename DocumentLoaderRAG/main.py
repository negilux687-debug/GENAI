from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5,
    },
)


llm = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0,
)


prompt = ChatPromptTemplate.from_messages([
    (
        "system","""you are a helpfull ai assistant. use only the provided context to answer the question.
        if the answer is not present in the context ,say : I could not find the answer in the document.

        """
    ),
    (
        "human","""context:
        {context} 
        
        question:{question}"""
    )
])


print("RAG system created.")
print("Press 0 to exit.")

while True:
    query = input("\nYou: ")

    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query,
        }
    )

    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}")