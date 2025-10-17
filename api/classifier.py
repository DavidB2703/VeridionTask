import pandas as pd
import torch
import re
from html import unescape
import os

# ============================================
# INITIALIZATION - Run only once at startup
# ============================================

# Get the correct path to the data file
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'insurance_taxonomy - insurance_taxonomy.csv')
labels = pd.read_csv(data_path)
candidate_labels = labels['label'].tolist()

from sentence_transformers import SentenceTransformer, util

# Use CPU instead of CUDA to avoid compatibility issues
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = SentenceTransformer("intfloat/multilingual-e5-base", device=device)
embeddings_labels = model.encode(candidate_labels, convert_to_tensor=True)

from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

# ============================================
# HELPER FUNCTIONS
# ============================================

def soft_clean(s: str) -> str:
    if not s:
        return ""
    s = unescape(s)  # decode HTML entities
    s = re.sub(r'https?://\S+', '<URL>', s)
    s = re.sub(r'\S+@\S+', '<EMAIL>', s)
    # înlocuiește doar multiplele spații, fără a atinge newline-urile
    s = re.sub(r'[ \t]+', ' ', s)
    # normalizează liniile multiple: mai mult de 2 -> doar 2
    s = re.sub(r'\n{3,}', '\n\n', s)
    # curăță spații la capete de linii
    s = re.sub(r' *\n *', '\n', s)
    return s.strip()

# ============================================
# CLASSIFICATION FUNCTION - Call this for each company
# ============================================

def classify_company_emb_zs(text: str):
    """
    Classify a company based on its description text.
    Returns the top 3 predicted labels.
    """
    text = soft_clean(text)
    sentence_to_classify = text

    embeddings_company = model.encode(sentence_to_classify, convert_to_tensor=True)

    # similaritate cosinus
    scores = util.cos_sim(embeddings_company, embeddings_labels)

    top5 = torch.topk(scores, k=10)
    top_labels = []
    for score, idx in zip(top5.values[0], top5.indices[0]):
        top_labels.append(candidate_labels[idx])

    result = classifier(sentence_to_classify, top_labels, multi_label=True)

    return result['labels'][:3]

# ============================================
# TEST CODE
# ============================================

if __name__ == "__main__":
    text = ("This is a detailed company profile "
            "Description: Welchcivils is a civil engineering and construction company that specializes in "
            "designing and building utility network connections across the UK. They offer multi-utility solutions "
            "that combine electricity, gas, water, and fibre optic installation into a single contract. Their design "
            "engineer teams are capable of designing electricity, water and gas networks from existing network connection "
            "points to meter locations at the development, as well as project management of reinforcements and diversions. "
            "They provide custom connection solutions that take into account any existing assets, maximize the usage of every"
            " trench, and meet project deadlines. Welchcivils has considerable expertise installing gas and electricity "
            "connections in a variety of market categories, including residential, commercial, and industrial projects, as as well. "
            "Business Tags: Construction Services, Multi-utilities, Utility Network Connections Design and Construction, "
            "Water Connection Installation, Multi-utility Connections, Fiber Optic Installation "
            "Sector: Services "
            "Category: Civil Engineering Services "
            "Niche: Other Heavy and Civil Engineering Construction")

    result = classify_company_emb_zs(text)
    print("Top 3 predicted labels:")
    for r in result:
        print(r)