from docx import Document
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF4 import PdfFileReader
from io import BytesIO
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from typing import Union
from pprint import pprint

 
use = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2")
tf.experimental.numpy.experimental_enable_numpy_behavior()
nltk.download('stopwords')    

def read_pdf(file: Union[str, bytes]) -> str:
    """Extract text from the given PDF file."""
    file_data = BytesIO(file)
    pdf_reader = PdfFileReader(file_data)
    text = ''.join([pdf_reader.getPage(i).extractText() for i in range(pdf_reader.getNumPages())])
    
    return spacy_cleaner(text) # clean_text(text, stop_words=True, skip=False)

def read_docx(file):
    file_data = BytesIO(file)
    doc = Document(file_data)
    temp = "" 
    text = ''.join([para.text for para in doc.paragraphs])
    return spacy_cleaner(text)  # clean_text(text, stop_words=True, skip=False)

def compute_cosine_similarity(doc1, doc2):
    doc1 = " ".join(list(get_skills(doc1)))
    doc2 = " ".join(list(get_skills(doc2)))
    jobs_embeding = use([doc1])
    candidates_embeding = use([doc2])
    return cosine_similarity(jobs_embeding, candidates_embeding)[0][0] * 100 

def compare_words(doc1, doc2):
    resume_words = clean_text(doc1, stop_words=True)
    job_listing_words = clean_text(doc2, stop_words=True)
    resume_words = set(re.findall(r'\b\w+\b', resume_words))
    job_listing_words = set(re.findall(r'\b\w+\b', job_listing_words))
    common_words = resume_words.intersection(job_listing_words)

    non_related_words = len(job_listing_words) - len(common_words)

    return (len(common_words) / non_related_words) * 100


def clean_text(text, stop_words=False, skip=False):
    """
    This function attempts to remove unnecessary parts of a text
    for potential embedding.
    """
    if skip:
        return text
    # Remove links
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    # Remove special characters except percentage sign
    text = re.sub(r'[^a-zA-Z0-9%]', ' ', text)
    # Remove specific resume related words (can be customized)
    remove_words = ['references', 'available', 'objective', 'summary', 'skills', 'phone', 'work experience', 'certifications', 'education', 'technical', "soft"]
    for word in remove_words:
        text = re.sub(rf'\b{word}\b', '', text, flags=re.IGNORECASE)
    if stop_words:
        stop_words = set(stopwords.words('english'))
        for word in stop_words:
            text = re.sub(rf'\b{word}\b', '', text, flags=re.IGNORECASE)
    # Remove leading/trailing whitespaces
    text = text.strip()

    return text


# Spacy section 

import spacy
from spacy.pipeline import EntityRuler
from spacy.lang.en import English
from spacy.tokens import Doc
  
nlp = spacy.load("en_core_web_lg")
skill_pattern_path = "jz_skill_patterns.jsonl"

ruler = nlp.add_pipe("entity_ruler")
ruler.from_disk(skill_pattern_path)
nlp.pipe_names

def get_skills(text):
    doc = nlp(text)
    myset = []
    subset = []
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            subset.append(ent.text)
    myset.append(subset)
    pprint(myset)
    return set(subset)


def spacy_cleaner(text):
    review = re.sub(
        '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"',
        " ",
        text,
    )
    review = review.lower()
    review = review.split()
    lm = WordNetLemmatizer()
    review = [
        lm.lemmatize(word)
        for word in review
        if not word in set(stopwords.words("english"))
    ]
    review = " ".join(review)
    return review

def match_resume(input_resume, input_skills):
    req_skills = get_skills(input_skills.lower())
    # print(req_skills)
    resume_skills = get_skills(input_resume.lower())
    # print(resume_skills)
    score = 0
    for x in req_skills:
        if x in resume_skills:
            score += 1
    req_skills_len = len(req_skills)
    match = round(score / req_skills_len * 100, 1)
    return match

    # print(f"The current Resume is {match}% matched to your requirements")



