from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from embedding_function import embedding_function

CHROMA_PATH = "../../../rag_presentation/chroma"

PROMPT_TEMPLATE = """
    Answer the question based only on the following
    context: {context}

    ---
    
    Answer the question based on the above context: {question}

"""


def answerToQuestion(question: str):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function())
    result = db.similarity_search_with_score(question, k=3)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in result])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(question=question, context=context_text)

    model = Ollama(model='llama3.1:8b')

    response_text = model.invoke(prompt)

    return response_text


print("\n\nWhat are the main applications of AI in healthcare diagnostics?\n")
print(answerToQuestion("What are the main applications of AI in healthcare diagnostics?"))

print("\n\nHow does AI enhance the personalization of learning in education?\n")
print(answerToQuestion("How does AI enhance the personalization of learning in education?"))

print("\n\nWhat are the key challenges associated with implementing AI in healthcare?\n")
print(answerToQuestion("What are the key challenges associated with implementing AI in healthcare?"))

print("\n\nHow can AI be used to predict and mitigate the impacts of climate change?\n")
print(answerToQuestion("How can AI be used to predict and mitigate the impacts of climate change?"))

print("\n\nWhat role does AI play in medical imaging, and how does it improve diagnostic accuracy?\n")
print(answerToQuestion("What role does AI play in medical imaging, and how does it improve diagnostic accuracy?"))

print("\n\nHow can AI-powered tutoring systems support students' learning outside the classroom?\n")
print(answerToQuestion("How can AI-powered tutoring systems support students' learning outside the classroom?"))

print("\n\nWhat are some ethical concerns associated with AI in education?\n")
print(answerToQuestion("What are some ethical concerns associated with AI in education?"))

print("\n\nHow is AI helping reduce carbon footprints across various sectors?\n")
print(answerToQuestion("How is AI helping reduce carbon footprints across various sectors?"))

print("\n\nWhat are some of the challenges AI faces in climate change predictions?\n")
print(answerToQuestion("What are some of the challenges AI faces in climate change predictions?"))

print("\n\nHow does adaptive learning technology work in AI-powered education platforms?\n")
print(answerToQuestion("How does adaptive learning technology work in AI-powered education platforms?"))


