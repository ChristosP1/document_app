import streamlit as st
from langchain_core.output_parsers import StrOutputParser
import re
from langchain.schema import BaseOutputParser
from prompts import INITIAL_PROMPT, REFINE_PROMPT, SUMMARIZATION_PROMPT
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS

#----------------------------------------------
import openai
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer 

from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

#----------------------------------------------
load_dotenv()  # Load environment variables from .env file
openai_api_key = os.getenv("OPENAI_API_TOKEN")

if openai_api_key is None:
    st.error("API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()
else:
    st.write("API key loaded successfully")

def generate_quiz_chain(prompt_template, llm, output_parser):
    chain = prompt_template | llm | output_parser
    return chain


def generate_summarization_chain(prompt_template, llm, output_parser):
    chain = prompt_template | llm | output_parser
    return chain


def extract_text_from_pdf(file):
    document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text


def extract_text_from_pdf_range(file, start_page=None, end_page=None):
    document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    if start_page is None or end_page is None:
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
    else:
        for page_num in range(start_page - 1, end_page):
            page = document.load_page(page_num)
            text += page.get_text()
    return text


def split_into_chunks(text, chunk_size=5000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks


def process_files_for_quiz(uploader_files, context, num_questions, quiz_type, llm, output_parser):
    all_chunks = []
    for file in uploader_files:
        text = extract_text_from_pdf(file)
        chunks = split_into_chunks(text)
        all_chunks.extend(chunks)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(all_chunks, embeddings)

    # Adjust the retriever to request 1 result at a time
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})

    # Extract only relevant context
    relevant_paragraphs = retriever.get_relevant_documents(context)  # Retrieve relevant paragraphs
    extracted_content = "\n\n".join([para.page_content for para in relevant_paragraphs])

    # Display extracted content for debugging
    st.write("Extracted Content:")
    st.write(extracted_content)

    # Generate the initial quiz chain
    initial_chain = generate_quiz_chain(INITIAL_PROMPT, llm, output_parser)

    initial_quiz_response = initial_chain.invoke({
        "num_questions": num_questions,
        "quiz_type": quiz_type,
        "quiz_context": extracted_content
    })

    # Generate the refining quiz chain
    validation_chain = generate_quiz_chain(REFINE_PROMPT, llm, output_parser)

    refined_quiz_response = validation_chain.invoke({
        "generated_quiz": initial_quiz_response,
        "num_questions": num_questions,
        "quiz_type": quiz_type
    })

    st.write("Quiz Generated!")
    st.write(refined_quiz_response)


def combine_summaries(chunk_summaries, llm, language, length):
    # Combine the summaries using AI for coherence
    combination_prompt_template = f"Combine the following {length} summaries into one coherent summary in {language}:\n\n{{chunk_summaries}}"
    combination_prompt = PromptTemplate(template=combination_prompt_template, input_variables=["chunk_summaries"])
    
    combination_chain = LLMChain(llm=llm, prompt=combination_prompt)
    combined_summary = combination_chain.run(chunk_summaries="\n\n".join(chunk_summaries))
    
    return combined_summary


def process_files_for_summarization(uploader_files, llm, output_parser):

    # Choose summary language
    summary_language = st.selectbox("Summary language:", ["English", "Greek", "Ukranian", "Dutch", "Spanish"])

    # Choose summary length
    summary_length = st.selectbox("Summary length:", ["Brief", "Medium", "Long"])

    # To store user inputs for each file
    file_options = {}

    for file in uploader_files:
        st.write(f"Options for: {file.name}")
        whole_doc = st.checkbox(f"Whole document ({file.name})", key=file.name)

        if not whole_doc:
            start_page = st.number_input(f"Start page for {file.name}", min_value=1, key=f"start_{file.name}")
            end_page = st.number_input(f"End page for {file.name}", min_value=start_page, key=f"end_{file.name}")
        else:
            start_page = None
            end_page = None


        file_options[file.name] = {"file": file, "whole_doc": whole_doc, "start_page": start_page, "end_page": end_page}


    if st.button(f"Summarize {file.name}"):
        for file_name, options in file_options.items():
            file = options["file"]
            whole_doc = options["whole_doc"]
            start_page = options["start_page"]
            end_page = options["end_page"]

            text = extract_text_from_pdf_range(file, start_page, end_page)
            st.write(f"Summary for {file.name}:")

            # Split text into manageable chunks
            chunks = split_into_chunks(text)

            # Summarize each chunk individually
            chunk_summaries = []
            summarization_chain = generate_summarization_chain(SUMMARIZATION_PROMPT, llm, output_parser)
            for chunk in chunks:
                summarized_chunk = summarization_chain.invoke({
                    "text": chunk,
                    "language": summary_language,
                    "length": summary_length,
                })
                chunk_summaries.append(summarized_chunk)

            # Combine the chunk summaries into a coherent summary
            combined_summary = combine_summaries(chunk_summaries, llm, summary_language, summary_length)

            st.write(combined_summary)


def main():
    load_dotenv()  # Load environment variables from .env file
    openai_api_key = os.getenv("OPENAI_API_TOKEN")

    st.title("Document Processing App")
    st.write("This app processes uploaded PDFs to generate a quiz or a summary.")

    # Initialize the LLM and parser
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=openai_api_key)
    output_parser = StrOutputParser()


    processing_type = st.selectbox("Choose processing type", ["Generate Quiz", "Summarize Document", "Translation"])
            

    if processing_type == "Generate Quiz":
        # Determine the uploader that accepts multiple PDF files
        uploader_files = st.file_uploader("Import your file", type="pdf", accept_multiple_files=True)

        if uploader_files:
            context = st.text_area("Specify the part to extract from the PDF(s)")
            num_questions = st.number_input("Enter the number of questions", min_value=1, max_value=6)
            quiz_type = st.selectbox("Select the quiz type", ["multiple-choice", "true-false", "text-based"])

            if context and num_questions and quiz_type and st.button("Generate Quiz"):
                process_files_for_quiz(uploader_files, context, num_questions, quiz_type, llm, output_parser)

    elif processing_type == "Summarize Document":
        # Determine the uploader that accepts multiple PDF files
        uploader_files = st.file_uploader("Import your file", type="pdf", accept_multiple_files=True)

        if uploader_files:
            process_files_for_summarization(uploader_files, llm, output_parser)


        

if __name__ == "__main__":
    main()




