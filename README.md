# YouTube Q&A Bot 🤖📺

This is a Streamlit app that allows users to input a YouTube video URL and ask questions based on the video's transcript. The app uses LangChain, FAISS vector store, and OpenAI embeddings/LLM to retrieve relevant answers.

## 🚀 Features

- Extracts transcript from any YouTube video (if available)
- Splits and embeds the transcript using `OpenAIEmbeddings`
- Stores and retrieves information using `FAISS`
- Supports natural language Q&A with `ChatOpenAI`
- Interactive Streamlit chatbot interface

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **Transcript Extraction**: `youtube_transcript_api`
- **Embeddings & LLM**: OpenAI (`text-embedding-3-large`, `gpt-3.5-turbo`)
- **Text Splitting**: LangChain's `RecursiveCharacterTextSplitter`
- **Vector Store**: FAISS
- **Retriever + QA Chain**: LangChain's retrieval chain and `create_stuff_documents_chain`

## 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
