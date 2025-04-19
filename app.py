import streamlit as st
from utils import extract_text_from_pdf
from chatbot import create_chatbot
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Chatbot hỏi đáp tài liệu", layout="centered")
st.title("🤖 Chatbot hỏi đáp tài liệu tiếng Việt")

pdf_file = st.file_uploader("📄 Tải lên file PDF", type="pdf")

if pdf_file:
    text = extract_text_from_pdf(pdf_file)
    chatbot = create_chatbot(text)

    st.success("✅ Tài liệu đã tải thành công. Nhập câu hỏi bên dưới.")
    question = st.text_input("❓ Nhập câu hỏi tiếng Việt:")

    if question:
        with st.spinner("🔍 Đang tìm câu trả lời..."):
            answer = chatbot.run(question)
            st.markdown("### 💬 Trả lời:")
            st.write(answer)
