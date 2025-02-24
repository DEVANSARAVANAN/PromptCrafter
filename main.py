import streamlit as st
from app.pages.page1 import run_page1
from app.pages.page2 import run_page2
import pandas as pd
from app.utils.processor import preProcessing



def main():

    st.title("Prompt Optimization")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "current_dataset" not in st.session_state:
        st.session_state.current_dataset = None
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    if "custom_prompt" not in st.session_state:
        st.session_state.custom_prompt = ""
    if "show_prompt_input" not in st.session_state:
        st.session_state.show_prompt_input = False
    
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            st.session_state.dataset = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            st.session_state.dataset = pd.read_excel(uploaded_file)
        
        st.write("Uploaded Dataset:")
        st.write(st.session_state.dataset)


    #Data Processing 
    if st.session_state.dataset is not None:
        with st.expander("Data Processing Options"):
                if st.button("UnProcessed Data", key="data_processing"):
                    st.session_state.processed_data = preProcessing(st.session_state.dataset)
                    st.session_state.current_dataset =st.session_state.processed_data
                    st.write("Processed Data:")
                    st.write(st.session_state.processed_data)

                if st.button("Processed Data", key="data"):
                    st.session_state.current_dataset =st.session_state.dataset
    
    #Optimization
    if st.session_state.current_dataset is not None:
        with st.expander("Prompt Optimization Options"):
            if st.button("Start Optimization", key="main_start_optimization"):
                run_page1(st.session_state.current_dataset)
            if st.button("Custom Prompt", key="custom_prompt_button"):
                st.session_state.show_prompt_input = not st.session_state.show_prompt_input
            if st.session_state.show_prompt_input:
                st.session_state.custom_prompt = st.text_area("Enter Your Custom Prompt:", value=st.session_state.custom_prompt)
                if st.button("Evaluate Prompt", key="custom_prompt_evaluate_button"):
                    if st.session_state.custom_prompt.strip():
                        run_page2(st.session_state.current_dataset, st.session_state.custom_prompt)
                    else:
                        st.warning("Please enter a custom prompt before proceeding.")
                        
if __name__ == "__main__":
    main()
