"""
Implements bill summarization, and category and tags generation for all bills (no MGL sections data is available for ~1300 bills)

Most of the main logic is in `demo_app_functions.py.`
This code runs very slow as it requires accessing data from a large file
"""

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from sentence_transformers import CrossEncoder
from sidebar import *
from tagging import *
from demo_app_functions import *

 
MGL_NAMES_FILE_PATH = "demoapp/chapter_section_names.pq"
COMMITTEE_FILE_PATH = "demoapp/committee_info.pq"
CSV_FILE_PATH = "demoapp/generated_bills.csv"

def render_response():
    """
    Renders the user interface for summarizing bill information, including the generation of
    a summary, evaluation metrics, and the ability to download the summarized text.

    This function performs the following steps:
    1. Sets the page configuration for the Streamlit UI.
    2. Initializes a CrossEncoder model for hallucination evaluation.
    3. Loads a dataset containing bills and their associated information from a parquet file.
    4. Retrieves selected bill information (number, title, and content) based on user input.
    5. Displays a button for the user to initiate the summarization process.
    6. Upon clicking the 'Summarize' button, generates a response including the summary, category,
       and tags for the selected bill.
    7. Updates a CSV file with the new summary information.
    8. Allows the user to download the updated CSV file.
    9. Calculates and displays evaluation metrics for the generated summary, including:
       - ROUGE-1, ROUGE-2, and ROUGE-L scores for assessing the overlap between the generated summary and the original bill content.
       - Cosine similarity score for measuring the similarity between the original bill content and the generated summary.
       - A factual consistency score using the initialized CrossEncoder model to evaluate the factual alignment of the generated summary with the original bill content.


    The function does not return any values but instead directly interacts with the Streamlit interface to render content.
    """
    set_page_config()
    model = CrossEncoder('vectara/hallucination_evaluation_model')

    # load the dataset containing bills and MGL information
    df = get_mgl_sections_file_all_bills()
    #get bill information based on the bill selected
    bill_number, bill_title, bill_text = get_selected_bill_info_all_bills(df)
    #get MGL section text
    mgl_ref = get_MGL_for_bill(df, bill_number)
    answer_container = st.container()
    with answer_container:
        submit_button = st.button(label='Summarize')
        col1, col2, col3 = st.columns([1.5, 1.5, 1])

        if submit_button:
            with st.spinner("Working hard..."):
                query, response = generate_response(df, bill_number, bill_title, bill_text, mgl_ref, COMMITTEE_FILE_PATH, MGL_NAMES_FILE_PATH)
        
                with col1:
                    st.subheader(f"Original Bill: #{bill_number}")
                    st.write(bill_title)
                    st.write(bill_text)

                with col2:
                    st.subheader("Generated Text")
                    st.write(response)
                    st.write("###")
                    summary, category, tags = parse_response(response)
                    update_csv(bill_number, bill_title, summary,category, tags, CSV_FILE_PATH)
                    st.download_button(
                                label="Download Text",
                                data=pd.read_csv(CSV_FILE_PATH).to_csv(index=False).encode('utf-8'),
                                file_name='Bills_Summarization.csv',
                                mime='text/csv',)
                        
                with col3:
                    if bill_text is None or str(bill_text).strip().lower() =="nan" or bill_text == "": 
                        pass
                    else:
                        st.subheader("Evaluation Metrics")
                        # rouge score addition
                        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                    
                        rouge_scores = scorer.score(query, response)
                        st.write(f"ROUGE-1 Score: {rouge_scores['rouge1'].fmeasure:.2f}")
                        st.write(f"ROUGE-2 Score: {rouge_scores['rouge2'].fmeasure:.2f}")
                        st.write(f"ROUGE-L Score: {rouge_scores['rougeL'].fmeasure:.2f}")
                        
                        # calc cosine similarity
                        vectorizer = TfidfVectorizer()
                        tfidf_matrix = vectorizer.fit_transform([query, response])
                        cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
                        st.write(f"Cosine Similarity Score: {cosine_sim[0][0]:.2f}")

                        # test hallucination
                        scores = model.predict([
                                [query, response]
                            ])
                        score_result = float(scores[0])
                        st.write(f"Factual Consistency Score: {round(score_result, 2)}")
                        
                       
                                    
                            # st.write("###")
                            # st.subheader("Token Usage")
                            # st.write(f"Response Tokens: {response_tokens + tag_tokens + cate_tokens}")
                            # st.write(f"Prompt Response: {prompt_tokens + tag_tokens + cate_prompt}")
                            # st.write(f"Response Complete:{completion_tokens +  tag_completion + cate_completion}")
                            # st.write(f"Response Cost: $ {response_cost + tag_cost + cate_cost}")              
                            # st.write(f"Cost: response $ {response_cost + tag_cost + cate_cost}")  

render_response()