# 💼 Chat with Hsien-Pang's CV

This is an AI-powered chatbot that allows you to interactively ask questions about **Hsien-Pang Hsieh’s CV**. It is built with [Streamlit](https://streamlit.io), [LangChain](https://www.langchain.com/), [FAISS](https://github.com/facebookresearch/faiss), and OpenAI's GPT API.

The chatbot uses semantic search and large language models to answer your questions **based solely on the content of Hsien-Pang's CV**.

---

## 🌐 Try it Live

👉 [**Launch the App on Streamlit Cloud**](https://chatbotrogerhsieh.streamlit.app/)

---

## 🎥 Demo Video

[![Watch the demo](https://via.placeholder.com/800x450.png?text=Click+to+Watch+Demo+Video)](https://drive.google.com/file/d/1_2r9X73ynALXjO5TmgQqGxwMa0HSnJ4U/view?usp=sharing)

👉 [Click here to watch the demo video on Google Drive](https://drive.google.com/file/d/1_2r9X73ynALXjO5TmgQqGxwMa0HSnJ4U/view?usp=sharing)

---

## 🔍 What Does This App Do?

1. **Loads Hsien-Pang’s CV from PDF** and splits it into small, overlapping text chunks.
2. **Embeds** those chunks using OpenAI Embeddings (`text-embedding-ada-002`) and stores them in a **FAISS vector database**.
3. **When you ask a question**, the app performs semantic similarity search to find the most relevant text chunks.
4. The top 3 chunks are used to construct a prompt, and **OpenAI’s GPT model** (gpt-3.5-turbo) generates a concise and accurate answer.
5. The app limits each session to **3 questions** to encourage thoughtful use, with an option to reset.
