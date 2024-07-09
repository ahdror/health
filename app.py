import streamlit as st
import requests
import json
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.language_models.llms import LLM
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Optional

st.set_page_config(page_title="헬스 트레이너", page_icon="🧊")
st.title("🧊 헬스 트레이너")

st.session_state["messages"] = []
st.session_state["store"] = dict()


# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Store chat history in session state
if "store" not in st.session_state:
    st.session_state["store"] = dict()

# Sidebar for session ID, API keys and reset buttons
with st.sidebar:
    session_id = st.text_input("접속자명", value="손님")

    
    if st.button("대화 화면 초기화"):
        st.session_state["messages"] = []
        st.rerun()
    if st.button("대화 저장 기록 초기화"):
        st.session_state["store"] = dict()
        st.rerun()

#출력하는 함수
def print_messages():
    ## 이전 대화기록을 출력해주는 코드    
    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        for chat_message in st.session_state["messages"]:
            st.chat_message(chat_message.role).write(chat_message.content)

# Print previous chat messages
print_messages()

# Function to retrieve or create a chat history object associated with the session id
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]

# Define the custom LLM class
class LlmClovaStudio(LLM):
    host: str
    api_key: str
    api_key_primary_val: str
    request_id: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.host = kwargs.get('host')
        self.api_key = kwargs.get('api_key')
        self.api_key_primary_val = kwargs.get('api_key_primary_val')
        self.request_id = kwargs.get('request_id')

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        headers = {
            "X-NCP-CLOVASTUDIO-API-KEY": self.api_key,
            "X-NCP-APIGW-API-KEY": self.api_key_primary_val,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self.request_id,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "text/event-stream"
        }

        system_prompt = '''\
        # 지시사항 
        - 당신은 헬스 트레이너 입니다.
        - 성별,몸무게,키,운동방법의 정보가 없으면 입력하라고 답변합니다.
        - 운동 및 식단에 관한 질문이 아니라면, 정보를 무시하고 답변하세요
        - 검색된 다음 문맥을 사용하여 질문에 답하세요. 답을 모른다면 모른다고 답변하세요.
        - 정보와 관계 없는 질문이라면, 당신이 아는 답변을 제공하세요.
        '''
        preset_text = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

        request_data = {
            "messages": preset_text,
            "topP": 0.9,
            "topK": 0,
            "maxTokens": 1024,
            "temperature": 0.1,
            "repeatPenalty": 1.2,
            "stopBefore": [],
            "includeAiFilters": False
        }

        response = requests.post(
            self.host + "/testapp/v1/chat-completions/HCX-DASH-001",
            headers=headers,
            json=request_data,
            stream=True
        )

        last_data_content = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if '"data":"[DONE]"' in decoded_line:
                    break
                if decoded_line.startswith("data:"):
                    last_data_content = json.loads(decoded_line[5:])["message"]["content"]

        return last_data_content

# Process user input and make a call to the LLM if input is provided
if user_input := st.chat_input("메세지를 입력해주세요."):
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))


    # Instantiate custom LLM with user-provided API keys
    llm = LlmClovaStudio(
        host='https://clovastudio.stream.ntruss.com',
        api_key='NTA0MjU2MWZlZTcxNDJiYxXVubGNTI5Vvj3+7OKX/EUwHeJtv0sO1VC74rpQaK2O',
        api_key_primary_val='JaWxqB0Ub0fKPzDXwPhDXWlLv4OfHRG3UTmp8NRr',
        request_id='37d72c32-2228-47b4-86c4-56ae0168c473'
    )

    # Create prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "설명을 자세하게 답변하세요."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm

    chain_with_memory = (
        RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history"
        )
    )

    response_content = chain_with_memory.invoke(
        {"question": user_input},
        config={"configurable": {"session_id": session_id}}
    )

    #assistant 응답 필요없는 문자 제거
    cleaned_response = response_content.replace('Human: ', '').replace('AI: ', '').strip()
    cleaned_response = cleaned_response.replace('System: ', '').strip()

    # Add the assistant's response to session_state
    st.session_state["messages"].append(
        ChatMessage(role="assistant", content=cleaned_response)
    )

    # assistant 응답 출력
    st.chat_message("assistant").write(cleaned_response)
