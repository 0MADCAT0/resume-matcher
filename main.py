import streamlit as st
from utils import *
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def main():
    tab1, tab2, tab3 = st.tabs(["Home", "Score", "About"])
    with tab1:
        st.title("Resume Matcher App")
        st.image("logo.jpg")
    
    with tab2:
        st.title("Score")
        st.markdown("Upload your resume and job description to get your similarity score")
        st.sidebar.title("File  Upload")
        st.sidebar.image("resume (Phone).jpg")
        text1 = process_file_upload("Upload the CV")
        st.sidebar.image("job.jpg")
        text2 = process_file_upload("Upload the Job Description") 
        st.markdown("---")
        if st.button("Open Processed CV"):
            st.markdown("---")
            st.markdown(text1, unsafe_allow_html=True) 
        st.markdown("---")
        if st.button("Close", ):
            st.markdown("")
        if text1 and text2:
            cosine = compute_cosine_similarity(text1, text2)
            compare = compare_words(text1, text2)
            spacy_match = match_resume(text1, text2)
            st.write(f"**Similarity of CV and Job Description:** %{round(cosine, 2)}")  
            st.write(f"**The CV and Job Description have** %{round(compare, 2)} **words in common.**") 
            st.write(f"**The current Resume is {spacy_match}% matched to your requirements**") 
            
    with tab3:
        st.title("About")
        st.write("This is a resume matcher app. It can compare resumes and job descriptions to give a similarity score.")
        st.write("It uses the LLM model embedings of the documents.")
        st.write("The app is built using Streamlit and Python.")
        st.write("The app is deployed on Streamlit Cloud.")
        st.write(f"""The app is developed by AI-run Team.
                 \nTeam Memebers:\n
                 \nPhd Emine İldes Eğri\n
                 \nMBA Orhan Bulut\n
                 \nEng. Hüseyin Işık\n
                 \nMsc Serdar Çağlar
                 """)

    

def process_file_upload(file_label):
    file = st.sidebar.file_uploader(file_label, type=['docx', 'pdf'])
    
    if file is not None:
        if file.type in ("application/pdf", 
                         "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
            bytes_data = file.getvalue()
            
            if file.type == "application/pdf":
                return read_pdf(bytes_data)
            else:
                return read_docx(bytes_data)

    return None

if __name__ == "__main__":
    main()