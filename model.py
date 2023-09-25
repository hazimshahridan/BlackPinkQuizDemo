import re
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import spacy
from tqdm import tqdm
import pandas as pd

def shall_extract(s,word="shall"):
    match = re.search(r"\b{}\b".format(word),s.lower())
    return match

def genQnA(shall_statement,qa):
    chat_history=[]
    test_query = "Generate a question from the following sentence: " + shall_statement
    test_result=qa({"question":test_query,"chat_history":chat_history})
    shall_context = test_result["answer"]

    context_query = shall_context + "Let's explain this step by step."
    context_result = qa({"question":context_query,"chat_history":chat_history})
    context_data = context_result["answer"]

    qna_query = "Question: " + shall_context + "\nAnswer: " + context_data + "Based on the question and answer above, generate 1 multiple choice quiz question. Do not note down the correct answer at the end of the quiz."
    qna_result = qa({"question":qna_query,"chat_history":chat_history})
    qna_data = qna_result["answer"]

    answer_query = qna_data + "Note down the correct answer at the end in the following format: 'Correct answer: <Answer Choice>). <Explanation>'. Let's explain step by step."
    answer_result = qa({"question":answer_query,"chat_history":chat_history})
    answer = answer_result["answer"]

    return {"Sentence":shall_statement,"Query":shall_context,"Context":context_data,"Quiz":qna_data,"Answer":answer_extract(answer)}

def answer_extract(answer):
    matches = re.findall(r"\b[A-Za-z]\)",answer)[0]
    ab=re.findall(r"[A-Za-z]",matches)[0]
    return ab

def init_models(key):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",openai_api_key=key,max_retries=3,request_timeout=60,temperature=0)
    embedder = OpenAIEmbeddings(openai_api_key=key,chunk_size=1)
    
    print("Models prepped.\n")
    return llm, embedder

def embed(data,embedder):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents(data)
    vector_db = FAISS.from_documents(documents=docs,embedding=embedder)
    return vector_db

def qaSetup(llm,db):
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""You are a subject matter on BlackPink. You must answer commands or questions based on your knowledge of BlackPink.
    Chat History:
    {chat_history}
    Question: {question}
    BlackPink SME Response:""")
    qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                            retriever=db.as_retriever(),
                                            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                                            return_source_documents=True,
                                            verbose=False)
    print("QA prepped.\n")
    return qa

def run_pipeline(data,qa):
    print("Running pipeline...\n")
    qnas = []
    shalls=[]
    for d in data:
        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe('sentencizer')
        doc = nlp(d)
        for sent in doc.sents:
            if shall_extract(sent.text):
                shalls.append(sent.text)
        print(len(shalls))
        for s in tqdm(shalls):
            qnas.append(genQnA(s,qa))
        df = pd.DataFrame(qnas)
        print("Done!\n")
    return df


