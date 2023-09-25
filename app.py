import streamlit as st
from model import * 

def main():
    with st.sidebar:
        openai_api_key = st.text_input("API Key", key="api_key", type="password")
    #api_key = os.environ("OPENAI_API_KEY")
    st.write('Are your friends true BlackPink fans?')
    st.write("Upload your factsheet to generate a quiz and test if they are! :D")
    st.write('Get Started Here:')
    factsheet = st.file_uploader("Upload your factsheet as a .txt file here!")
    if factsheet is None:
        st.session_state["upload_state"]="Please upload a file first!"
        st.stop()
    def showQuiz(data):
        llm, embedder = init_models(openai_api_key)
        db = embed(data,embedder)
        qa = qaSetup(llm,db)
        quizzes = run_pipeline(data,qa)
        def question_index(index):
            return "Question " + str(index)
        if "quiz_df" not in st.session_state:
            st.session_state["quiz_df"] = pd.DataFrame(list(zip(map(question_index,quizzes.index.values.tolist()),quizzes["Quiz"])),columns=["Question #","Question"])
        if "ans_df" not in st.session_state:
             st.session_state["ans_df"] = pd.DataFrame(list(zip(quizzes["Quiz"],quizzes["Answer"])),columns=["Question","Answer"])
        return
    def upload():
        data=[factsheet.read().decode('utf-8')]
        with st.spinner('This may take some time... Why don\'t you get a coffee while waiting?'):
            showQuiz(data)
        st.table(st.session_state["quiz_df"])
        return
    
    def show_ans():
        st.table(st.session_state["ans_df"])
        return
    
    if "Generate Quiz!" not in st.session_state:
            st.session_state["Generate Quiz!"] = False
    if "Show Answers" not in st.session_state:
            st.session_state["Show Answers"] = False
            
    if openai_api_key and st.button("Generate Quiz!"):
        st.session_state["Generate Quiz!"] = not st.session_state["Generate Quiz!"]
        upload()
    if openai_api_key and st.session_state["Generate Quiz!"]:
        if st.button("Show Answers"):
            st.session_state["Show Answers"] = not st.session_state["Show Answers"]
            show_ans()
if __name__ == '__main__':
    main()
