from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model=OllamaLLM(model="gemma3:4b")

template=''' you are an expert in answering questions about Pizza restraunts
Here are some relevant reviews : {reviews}
Here is  the question to  answer : {question} '''

prompt=ChatPromptTemplate.from_template(template)
chain = prompt | model
while True:
    print("\n\n-----------------------------------------------")
    question=input(" Ask your question (q to quit : )")
    if question =="q":
        break
    result=chain.invoke({"reviews":[],"question": question})
    print(result)
