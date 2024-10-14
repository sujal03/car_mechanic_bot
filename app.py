from langchain_community.llms import Cohere
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os
import warnings
import streamlit as st
from streamlit_chat import message
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

if 'conversation' not in st.session_state:
    st.session_state['conversation'] =None
if 'messages' not in st.session_state:
    st.session_state['messages'] =[]


def get_response(user_input):
    if st.session_state['conversation'] is None:
        llm=Cohere(temperature=0.1)

        
        prompt=PromptTemplate(
            input_variables=["input","history"],
            template="""
            Task: you are a vehicle repairing solution providing chatbot.
            Instructions: 1. If user have any problem related to one's vehicle then you have to ask him the model name, model year and the specific problem of vehicle.
                   2. if user provides all the details only then you have to provide them solution.
                   3. if user do not give all the details then you keep asking him model name, model year and the specific problem.
                   4. your tone is polite.
                   5. try to use maximum 20 words in a response and explain step by step with using bullet points.
                   6. if user is just greeting then only respond "Hello,I am a car mechanic assistant.How can I assist you today? Please provide me the model name, model year and the problem of your vehicle to assist you further."
                   7. you have previous history also in {history}
            user quesion: {input}
            """
        )
        memory=ConversationBufferMemory()
        st.session_state['conversation']=ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=True
        )    
    result=st.session_state['conversation'].predict(input=user_input)
    print(result)
    return result



# UI 
    
st.set_page_config(page_title='Chatbot',page_icon=':robot_face:')
st.markdown("<h1 style='text-align=center;'>How can I assist you today?</h1>",unsafe_allow_html=True)
response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("Your question goes here:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')
        if submit_button:
            st.session_state['messages'].append(user_input)
            model_response=get_response(user_input)
            st.session_state['messages'].append(model_response)
            with response_container:
                for i in range(len(st.session_state['messages'])):
                        if (i % 2) == 0:
                            message(st.session_state['messages'][i], is_user=True, key=str(i) + '_user')
                        else:
                            message(st.session_state['messages'][i], key=str(i) + '_AI')