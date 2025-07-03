# importing required modules
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.prompts import PromptTemplate
import re
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


st.title("YouTube Q&A Bot")

def get_video_id(url):
    match = re.search(r"(?:v=|youtu.be/)([\w-]{11})", url)
    return match.group(1) if match else None

video_url = st.text_input("Enter YouTube Video URL:")

if video_url:
    video_id = get_video_id(video_url)

    if video_id:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            output = ""
            for item in transcript:
                sentence = item["text"]
                output += f"{sentence}\n\n"
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

            chunks = text_splitter.split_text(output)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            # embedding_text = embeddings.embed_documents(chunks)
            document = [Document(page_content=chunk) for chunk in chunks]
            vector_store = FAISS.from_documents(document, embeddings)
            retriever = vector_store.as_retriever()

            prompt_template = """
                Answer the question "{input}" based on the context below:

                <context>
                {context}
                </context>
                """
            
            prompt = PromptTemplate.from_template(prompt_template)

            llm = ChatOpenAI(model="gpt-3.5-turbo")

            combine_docs_chain = create_stuff_documents_chain(llm=llm,prompt=prompt)
            chain = create_retrieval_chain(retriever, combine_docs_chain)

            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # # React to user input
            if prompt := st.chat_input("What is up?"):
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

                response = chain.invoke({"input": prompt})
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response["answer"])
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            # st.text_area("Video Transcript", output, height=400)
        except Exception as e:
            st.error(f"Error fetching transcript: {e}")
    else:
        st.warning("Invalid YouTube URL. Please check and try again.")








