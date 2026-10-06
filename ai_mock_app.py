import os
import re
import tempfile
import random
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer  # type: ignore
import spacy  # type: ignore
import pyidaungsu as pds  # type: ignore
from spacy.tokens import Doc  # type: ignore
import streamlit as st

from parser import extract_text_from_pdf

# ============================================================
# PAGE CONFIGURATION & STYLING (FROM APP.PY)
# ============================================================
st.set_page_config(
    page_title="Technical အင်တာဗျူး အစမ်းလေ့ကျင့်သည့် စနစ်",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* HIDE DEFAULT STREAMLIT HEADERS & SIDEBAR */
    header { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }
    
    /* GLOBAL APP BACKGROUND & TEXT */
    .stApp { background-color: #E8F0F1; color: #172224; }
    .main { padding-top: 0rem; }
    .block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1400px; }

    /* GIANT TYPOGRAPHY */
    .giant-title { text-align: center; font-size: 40px; font-weight: 800; color: #111A1C; margin-top: 10px; margin-bottom: 5px; letter-spacing: -1.5px; }
    .giant-subtitle { text-align: center; font-size: 16px; color: #7B9294; margin-bottom: 30px; max-width: 650px; margin-left: auto; margin-right: auto; line-height: 1.6; }

    /* CUSTOM CARDS */
    .solid-teal-card { background-color: #618487; border-radius: 20px; padding: 35px; color: #FFFFFF; margin-bottom: 25px; box-shadow: 0px 10px 20px rgba(97, 132, 135, 0.15); }
    .solid-teal-title { font-size: 22px; font-weight: 700; color: #FFFFFF; margin-bottom: 12px; display: flex; align-items: center; gap: 10px;}
    .solid-teal-text { font-size: 15px; color: #D5E2E3; line-height: 1.7; }
    
    
    /* FILE UPLOADER & DARK INPUTS */
    div[data-testid="stFileUploader"] { 
        background-color: #172224 !important; 
        border: none !important; 
        border-radius: 15px !important; 
        color: #FFFFFF !important; 
    }

    div[data-baseweb="base-input"], 
    div[data-baseweb="textarea"],
    div[data-testid="stTextArea"] > div,
    div[data-testid="stTextInput"] > div { 
        background-color: #172224 !important; 
        border: none !important; 
        border-radius: 15px !important; 
    }

    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextInput"] input,
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="base-input"] input,
    div[data-baseweb="textarea"] textarea:disabled,
    div[data-baseweb="base-input"] input:disabled,
    textarea, input { 
        color: #FFFFFF !important; 
        -webkit-text-fill-color: #FFFFFF !important; 
        caret-color: #FFFFFF !important;
        opacity: 1 !important;
        font-weight: 500 !important;
    }

    /* Soft White Placeholder Text */
    div[data-testid="stTextArea"] textarea::placeholder,
    div[data-testid="stTextInput"] input::placeholder,
    textarea::placeholder, input::placeholder {
        color: rgba(255, 255, 255, 0.6) !important;
        -webkit-text-fill-color: rgba(255, 255, 255, 0.6) !important;
    }

    textarea:focus, input:focus { outline: 2px solid #618487 !important; }
    label { color: #4A6365 !important; font-weight: 700 !important; font-size: 14px !important; margin-bottom: 5px !important; }

    /* BUTTON STYLING */
    .stButton > button { background-color: #172224 !important; color: #FFFFFF !important; border: none; border-radius: 20px; padding: 10px 25px; font-weight: 600; transition: all 0.2s ease; }
    .stButton > button:hover { background-color: #26383B !important; transform: translateY(-2px); box-shadow: 0px 5px 15px rgba(23, 34, 36, 0.2); }

    /* METRICS DASHBOARD */
    .metric-card { background-color: #FFFFFF; border-radius: 20px; padding: 25px 20px; text-align: center; height: 100%; box-shadow: 0px 4px 15px rgba(0,0,0,0.03); transition: transform 0.2s; }
    .metric-card:hover { transform: translateY(-5px); }
    .metric-icon { font-size: 30px; color: #618487; margin-bottom: 10px; display: flex; justify-content: center; }
    .metric-value { font-size: 36px; font-weight: 800; color: #111A1C; }
    .metric-label { font-size: 14px; font-weight: 700; color: #7B9294; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; }
    
    .detail-metric { border-radius: 12px; padding: 15px; text-align: center; margin-top: 10px; background-color: #E8F0F1; }
    .detail-label { font-size: 11px; font-weight: 800; letter-spacing: 1px; color: #4A6365; }
    .detail-value { font-size: 24px; font-weight: 800; color: #172224; margin-top: 5px; }

    /* QUESTIONS & ANSWERS */
    .question-box { background-color: #FFFFFF; border-left: 5px solid #618487; padding: 20px 25px; border-radius: 15px; margin-top: 15px; margin-bottom: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.02); }
    .question-number { color: #7B9294; font-weight: 800; font-size: 11px; letter-spacing: 1.5px; margin-bottom: 8px; }
    .question-text { color: #111A1C; font-size: 17px; font-weight: 700; line-height: 1.5; }
    .user-answer-card { background-color: #172224; border-radius: 12px; padding: 20px; margin-top: 10px; color: #FFFFFF; font-size: 15px; line-height: 1.7; }
    
    /* EXPANDERS */
    [data-testid="stExpander"] { border: none !important; border-radius: 15px; background-color: #FFFFFF; box-shadow: 0px 4px 10px rgba(0,0,0,0.02); overflow: hidden; margin-top: 10px; }
    .streamlit-expanderHeader { color: #111A1C !important; font-weight: 700; background-color: #FFFFFF; }
    .correct-answer-text { padding: 15px; color: #4A6365; font-size: 15px; line-height: 1.7; border-left: 4px solid #92AEB0; background-color: #F8FAFA; }

    /* FOOTER */
    hr { border-color: #DCE5E6; margin-top: 40px; margin-bottom: 40px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIG & MODEL ARCHITECTURE DEFINITIONS
# ============================================================
MODEL_NAME = 'xlm-roberta-base'
SAVE_MODEL_PATH = 'best_tech_interview_model.pt'
DATASET_PATH = "Tech_Interview_Datasets_Clean_Negatives.xlsx"
MAX_LEN = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiOutputRegressor(nn.Module):
    def __init__(self, model_name):
        super(MultiOutputRegressor, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_dim = self.bert.config.hidden_size
        self.drop = nn.Dropout(p=0.2)

        self.acc_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.sent_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.anx_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(
            outputs.last_hidden_state * input_mask_expanded, dim=1
        )
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = self.drop(sum_embeddings / sum_mask)

        return torch.cat([
            self.acc_head(mean_pooled),
            self.sent_head(mean_pooled),
            self.anx_head(mean_pooled),
        ], dim=1)

# ============================================================
# MYANMAR TOKENIZER & PARSER HELPERS
# ============================================================
class MyanmarTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        if not text:
            return Doc(self.vocab, words=[], spaces=[])

        raw_tokens = pds.tokenize(text, form="word")
        split_tokens = []
        for tok in raw_tokens:
            sub_toks = re.split(r'([,၊])', tok)
            for st in sub_toks:
                if st:
                    split_tokens.append(st)

        words = []
        spaces = []
        curr = 0

        for token in split_tokens:
            pos = text.find(token, curr)
            if pos == -1:
                continue

            if pos > curr:
                gap = text[curr:pos]
                words.append(gap)
                spaces.append(False)

            words.append(token)
            spaces.append(False)
            curr = pos + len(token)

        if curr < len(text):
            words.append(text[curr:])
            spaces.append(False)

        return Doc(self.vocab, words=words, spaces=spaces)

MYANMAR_DIACRITICS = '[\u102B-\u103E\u1056-\u1059]'

def validate_input_text(text: str, min_chars: int = 5):
    if not text or not text.strip():
        return False, 'စာသား ရိုက်ထည့်ထားခြင်း မရှိပါ။'

    text = text.strip()

    if re.search(r'(.)\1{4,}', text):
        return False, 'စာလုံးများ အလွန်အမင်း ထပ်နေပါသည်။'

    if re.search(rf'({MYANMAR_DIACRITICS})\1+', text):
        return False, 'မြန်မာ စာလုံးပေါင်း/Font မှားယွင်းနေပါသည်။'

    if not re.search(r'[\u1000-\u102A a-zA-Z]', text):
        return False, 'အဓိပ္ပာယ်ရှိသော စာလုံးများ ရိုက်ထည့်ပါ။'

    if len(re.sub(r'\s+', '', text)) < min_chars:
        return False, f'အဖြေသည် အနည်းဆုံး စာလုံးရေ {min_chars} လုံး ရှိရပါမည်။'

    return True, 'Valid'

EXCLUDED_SECTION_MAP = {
    "ကိုယ်ရေးအကျဉ်းချုပ်": ["ကိုယ်ရေးအကျဉ်းချုပ်", "ကိုယ်ရေးအကျဉ်း", "summary", "about", "profile"],
    "လုပ်ငန်းအတွေ့အကြုံ": ["လုပ်ငန်းအတွေ့အကြုံ", "အလုပ်အတွေ့အကြုံ", "workexperience", "experience"],
    "ကိုယ်ရည်ကိုယ်သွေးဆိုင်ရာ ကျွမ်းကျင်မှုများ": ["ကိုယ်ရည်ကိုယ်သွေးဆိုင်ရာကျွမ်းကျင်မှုများ", "softskills", "personalskills"],
    "Projectများ": ["projectများ", "projects", "စီမံကိန်းများ", "project"],
    "ဆုများနှင့် အောင်မြင်မှုများ": ["ဆုများနှင့်အောင်မြင်မှုများ", "achievements", "awards"]
}

NON_EXCLUDED_HEADERS = [
    "နည်းပညာကျွမ်းကျင်မှုများ", "ကျွမ်းကျင်မှုများ", "skills", "technicalskills",
    "ပညာအရည်အချင်း", "education", "ဘာသာစကားကျွမ်းကျင်မှုများ", "languages",
    "အောင်လက်မှတ်များ", "certifications", "ကိုးကားနိုင်သူများ", "references"
]

CONTACT_PERSONAL_KEYWORDS = [
    "အီမေးလ်", "email", "linkedin", "github", "ဖုန်း", "phone", "tel",
    "အမည်", "အသက်", "လျှောက်ထားလိုသည့်ရာထူး", "ရာထူး", "မွေးသက္ကရာဇ်",
    "အဘအမည်", "အဖအမည်", "ကျား/မ", "လိင်", "အိမ်ထောင်ရှိ/မရှိ", "အိမ်ထောင်ရေး",
    "အောင်လက်မှတ်များ", "အောင်လက်မှတ်", "နည်းပညာကျွမ်းကျင်မှုများ", "နိုင်ငံသားမှတ်ပုံတင်အမှတ်", "မှတ်ပုံတင်", "လူမျိုး"
]

KEYVALUE_PATTERNS = {
    "အမည်": [r"(?:အမည်|Name)[\s:-]+([^\n]+)", r"^အမည်\s*\n\s*([^\n]+)"],
    "မွေးသက္ကရာဇ်": [r"(?:မွေးသက္ကရာဇ်|Date of Birth|DOB)[\s:-]+([^\n]+)"],
    "အသက်": [r"(?:အသက်|Age)[\s:-]+([^\n]+)"],
    "နိုင်ငံသားမှတ်ပုံတင်အမှတ်": [r"(?:နိုင်ငံသားမှတ်ပုံတင်အမှတ်|မှတ်ပုံတင်|NRC)[\s:-]+([^\n]+)"],
    "ပညာအရည်အချင်း": [r"(?:ပညာအရည်အချင်း|ပညာအရည[်း်\s]*အချင[်း်\s]*|Education|Qualification)[\s:-]+([^\n]+)"],
    "လျှောက်ထားလိုသည့်ရာထူး": [r"(?:လျှောက်ထားလိုသည့်ရာထူး|လျှောက်ထားသည့်ရာထူး|ရာထူး|Applied Position|Position)[\s:-]+([^\n]+)"],
    "ဖုန်းနံပါတ်": [r"(?:ဖုန်းနံပါတ်|ဖုန်း|Phone|Mobile|Tel)[\s:-]+([^\n]+)", r"(?:09|\+?959)\d{7,9}"],
    "အဘအမည်": [r"(?:အဘအမည်|အဖအမည်|Father's Name|Father Name)[\s:-]+([^\n]+)"],
    "ကျား/မ": [r"(?:ကျား/မ|ကျား\s*/\s*မ|လိင်|Gender|Sex)[\s:-]+([^\n]+)"],
    "အိမ်ထောင်ရှိ/မရှိ": [r"(?:အိမ်ထောင်ရှိ/မရှိ|အိမ်ထောင်ရှိ\s*/\s*မရှိ|အိမ်ထောင်ရေးအခြေအနေ|Marital Status)[\s:-]+([^\n]+)"],
    "လူမျိုး/ဘာသာ": [r"(?:လူမျိုး\s*/\s*ဘာသာ|လူမျိုး|ဘာသာ|Race/Religion)[\s:-]+([^\n]+)"],
    "အောင်လက်မှတ်များ": [r"(?:အောင်လက်မှတ်များ|အောင်လက်မှတ်|Certifications|Certificates)[\s:-]+([^\n]+)"],
    "နည်းပညာကျွမ်းကျင်မှုများ": [r"(?:နည်းပညာကျွမ်းကျင်မှုများ|Technical Skills|Technical Skill|Skills)[\s:-]+([^\n]+)"],
    "ဘာသာစကားကျွမ်းကျင်မှုများ": [r"(?:ဘာသာစကားကျွမ်းကျင်မှုများ|Languages|Language)[\s:-]+([^\n]+)"],
    "နေရပ်လိပ်စာ": [r"(?:နေရပ်လိပ်စာ|လိပ်စာ|Address)[\s:-]+([^\n]+)"]
}

STOPWORDS = {
    'in', 'of', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'an', 'a', 'the', 'and', 'or',
    'is', 'are', 'was', 'were', 'be', 'been', 'com', 'http', 'https', 'www', 'gmail', 'org', 'net'
}

TECH_SKILL_LABELS = {
    "နည်းပညာကျွမ်းကျင်မှုများ", "technical_skills", "tech_skills", 
    "skills", "skill", "technical skills"
}

def normalize_myanmar_spelling(text: str) -> str:
    if not text:
        return ""
    text = text.replace("ောာ", "ော").replace("ာာ", "ာ")
    text = text.replace("််", "်").replace("းး", "း")
    return text

def clean_header_text(text: str) -> str:
    text = normalize_myanmar_spelling(text)
    return re.sub(r'[\s\-:_၊။\d\.\*•\(\)]', '', text).lower()

def is_new_key_or_header(line_str: str) -> bool:
    normalized_line = clean_header_text(line_str)
    for key in KEYVALUE_PATTERNS.keys():
        if re.match(fr'^(?:{key})[\s:-]+', line_str, re.IGNORECASE):
            return True
    for label_name, keywords in EXCLUDED_SECTION_MAP.items():
        for kw in keywords:
            norm_kw = clean_header_text(kw)
            if norm_kw and norm_kw in normalized_line:
                return True
    for kw in NON_EXCLUDED_HEADERS:
        norm_kw = clean_header_text(kw)
        if norm_kw and norm_kw in normalized_line:
            return True
    if re.match(r'^[A-Za-z0-9\u1000-\u109F\s/]{2,25}\s*[:\-]', line_str):
        return True
    return False

def merge_multiline_text(text: str) -> str:
    lines = text.split('\n')
    merged_lines = []
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        if merged_lines and not is_new_key_or_header(line_str):
            merged_lines[-1] += " " + line_str
        else:
            merged_lines.append(line_str)
    return "\n".join(merged_lines)

def is_contact_or_personal_line(line_str: str) -> bool:
    line_lower = line_str.lower()
    if any(kw in line_lower for kw in CONTACT_PERSONAL_KEYWORDS):
        return True
    if re.search(r'[\w\.-]+@[\w\.-]+\.\w+', line_str):
        return True
    if re.search(r'(09|\+?959)\d{7,9}', line_str):
        return True
    if "linkedin.com" in line_lower or "github.com" in line_lower:
        return True
    return False

def match_section_header(line_text: str):
    parts = line_text.split(':', 1)
    potential_header = parts[0]
    normalized_header = clean_header_text(potential_header)

    if len(normalized_header) > 50:
        return None, False, line_text

    for label_name, keywords in EXCLUDED_SECTION_MAP.items():
        for kw in keywords:
            norm_kw = clean_header_text(kw)
            if norm_kw and norm_kw in normalized_header:
                inline_content = parts[1].strip() if len(parts) > 1 else ""
                return label_name, True, inline_content

    for kw in NON_EXCLUDED_HEADERS:
        norm_kw = clean_header_text(kw)
        if norm_kw and norm_kw in normalized_header:
            inline_content = parts[1].strip() if len(parts) > 1 else ""
            return "အထွေထွေ", False, inline_content

    return None, False, line_text

def extract_keyvalue_entities(cv_text: str):
    kv_entities = []
    normalized_text = normalize_myanmar_spelling(cv_text)

    for label, patterns in KEYVALUE_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, normalized_text, re.MULTILINE | re.IGNORECASE)
            if match:
                extracted_val = match.group(1 if match.groups() else 0).strip()
                extracted_val = re.sub(r'^[\s:-]+', '', extracted_val)
                if extracted_val:
                    kv_entities.append({"label": label, "text": extracted_val})
                    break

    return kv_entities

def parse_cv_complete(cv_text: str, nlp_model):
    lines = cv_text.split('\n')
    extracted_sections = {}
    current_section = "အထွေထွေ"
    extracted_sections[current_section] = []

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        detected_header, is_excluded, remaining_content = match_section_header(line_str)

        if detected_header:
            current_section = detected_header
            if current_section not in extracted_sections:
                extracted_sections[current_section] = []

            if remaining_content and not is_contact_or_personal_line(remaining_content):
                extracted_sections[current_section].append(remaining_content)
        else:
            if is_contact_or_personal_line(line_str):
                continue
            extracted_sections[current_section].append(line_str)

    sections_result = {
        sec: "\n".join(content)
        for sec, content in extracted_sections.items()
        if sec in EXCLUDED_SECTION_MAP and content
    }

    entities_result = extract_keyvalue_entities(cv_text)
    already_extracted_labels = {e["label"] for e in entities_result}

    curr_sec = "အထွေထွေ"
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        detected_header, is_excluded, _ = match_section_header(line_str)
        if detected_header:
            curr_sec = detected_header

        if curr_sec in EXCLUDED_SECTION_MAP and not is_contact_or_personal_line(line_str):
            continue

        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', line_str)
        if email_match and "အီးမေးလ်" not in already_extracted_labels:
            entities_result.append({"label": "အီးမေးလ်", "text": email_match.group(0)})
            already_extracted_labels.add("အီးမေးလ်")

        github_match = re.search(r'github\.com/[A-Za-z0-9_.-]+', line_str, re.I)
        if github_match and "GitHub" not in already_extracted_labels:
            entities_result.append({"label": "GitHub", "text": github_match.group(0)})
            already_extracted_labels.add("GitHub")

        linkedin_match = re.search(r'linkedin\.com/in/[A-Za-z0-9_.-]+', line_str, re.I)
        if linkedin_match and "LinkedIn" not in already_extracted_labels:
            entities_result.append({"label": "LinkedIn", "text": linkedin_match.group(0)})
            already_extracted_labels.add("LinkedIn")

        doc = nlp_model(line_str)
        for ent in doc.ents:
            if ent.label_ in already_extracted_labels:
                continue

            text = ent.text.strip()
            if text.count('(') > text.count(')'):
                text += ')'

            if not any(e["label"] == ent.label_ and e["text"] == text for e in entities_result):
                entities_result.append({"label": ent.label_, "text": text})

    return {
        "extracted_sections": sections_result,
        "extracted_entities": entities_result
    }
import re

import re


def is_meaningful_answer(text: str) -> bool:
    text = text.strip()

    # 1. Check minimum length
    if len(text) < 5:
        return False

    # 2. Check character variety (e.g., "aaaaa", "asdfasdf")
    if len(set(text.lower())) <= 3:
        return False

    # 3. Check for Keyboard Row Mashing Patterns (e.g., "asdf", "qwerty", "zxcv")
    mashing_patterns = [
        r"asdf",
        r"sdfg",
        r"dfgh",
        r"fghj",
        r"ghjk",
        r"hjkl",
        r"qwer",
        r"wert",
        r"erty",
        r"rtyu",
        r"tyui",
        r"yuio",
        r"uiop",
        r"zxcv",
        r"xcvb",
        r"cvbn",
        r"vbnm",
    ]
    if any(
        re.search(pat, text.lower()) for pat in mashing_patterns
    ):
        return False

    has_burmese = bool(re.search(r"[\u1000-\u109F]", text))

    if not has_burmese:
        # 4. English Consecutive Consonants Check
        # English စာလုံးအများစုတွင် ဗျည်း ၄ လုံး သို့မဟုတ် ၄ လုံးထက်ပို၍ ဆက်တိုက်ပါလေ့မရှိပါ (e.g., "jfdjf")
        if re.search(r"[bcdfghjklmnpqrstvwxz]{4,}", text.lower()):
            return False

        # 5. Vowel Ratio & Structure Check
        vowels = re.findall(r"[aeiouyAEIOUY]", text)
        vowel_ratio = len(vowels) / len(text)

        # Vowel ratio ပုံမှန် မဟုတ်ပါက (20% ထက်နည်းရင် သို့မဟုတ် 70% ထက်များရင်)
        if vowel_ratio < 0.20 or vowel_ratio > 0.70:
            return False

    return True

# def to_burmese_number(num: int) -> str:
#     """အင်္ဂလိပ် ဂဏန်းများကို မြန်မာ ဂဏန်းသို့ ပြောင်းပေးသော Function"""
#     burmese_digits = "၀၁၂၃၄၅၆၇၈၉"
#     return "".join(burmese_digits[int(d)] for d in str(num))


# def display_formatted_extracted_text(extracted_text: str):
#     """CV Extracted Text များကို (၁)၊ (၂) မြန်မာ နံပါတ်စဉ်တပ်၍ ပြသပေးသော Function"""
#     if not extracted_text or not str(extracted_text).strip():
#         st.info("အချက်အလက်များ မရှိသေးပါ။")
#         return

#     lines = [
#         line.strip()
#         for line in str(extracted_text).split("\n")
#         if line.strip()
#     ]

#     items_html = ""
#     for idx, line in enumerate(lines, 1):
#         myanmar_num = to_burmese_number(idx)
#         items_html += (
#             f'<div style="background-color: #1E293B; color: #F8FAFC;'
#             " padding: 10px 14px; margin-bottom: 8px; border-radius: 6px;"
#             " border-left: 4px solid #618487; font-size: 14px; line-height: 1.6;"
#             ' display: flex; align-items: flex-start;">'
#             '<span style="font-weight: bold; color: #618487 min-width: 45px;'
#             f' display: inline-block;">({myanmar_num})</span>'
#             f'<span style="flex-grow: 1;">{line}</span></div>'
#         )

#     container_html = (
#         '<div style="background-color: #0F172A; padding: 12px; border-radius:'
#         " 10px; border: 1px solid #334155; max-height: 400px; overflow-y:"
#         f' auto;">{items_html}</div>'
#     )

#     st.markdown(container_html, unsafe_allow_html=True)

# def to_burmese_number(num: int) -> str:
#     """အင်္ဂလိပ် ဂဏန်းများကို မြန်မာ ဂဏန်းသို့ ပြောင်းပေးသော Function"""
#     burmese_digits = "၀၁၂၃၄၅၆၇၈၉"
#     return "".join(burmese_digits[int(d)] for d in str(num))


# def display_formatted_extracted_text(extracted_text: str):
#     """CV Extracted Text များကို ခေါင်းစဉ်နှင့် စာကြောင်းများ ခွဲခြား၍ (၁)၊ (၂) မြန်မာ နံပါတ်စဉ်တပ် ပြသပေးသော Function"""
#     if not extracted_text or not str(extracted_text).strip():
#         st.info("အချက်အလက်များ မရှိသေးပါ။")
#         return

#     lines = [
#         line.strip()
#         for line in str(extracted_text).split("\n")
#         if line.strip()
#     ]

#     items_html = ""
#     item_counter = 1  # နံပါတ်စဉ်အတွက် တက်မည့် ကောင်တာ

#     for line in lines:
#         clean_line = line.lower()

#         # ခေါင်းစဉ် (Header) ဖြစ်နိုင်ချေရှိသော စာကြောင်းများကို စစ်ဆေးခြင်း
#         is_header = (
#             "curriculum vitae" in clean_line
#             or clean_line.startswith("ကိုယ်ရေးအချက်အလက်")
#             or line.endswith(":")
#         )

#         if is_header:
#             # Header စာကြောင်းများအတွက် (နံပါတ်စဉ်မပါဘဲ ခေါင်းစဉ် ပုံစံဖြင့် ပြသမည်)
#             items_html += (
#                 f'<div style="color: #F8FAFC;'
#                 " font-weight: bold; font-size: 15px; padding: 12px 14px;"
#                 " margin-top: 10px; margin-bottom: 10px; border-radius: 6px;"
#                 'text-align: center;">'
#                 f"{line}</div>"
#             )
#         else:
#             # အချက်အလက် စာကြောင်းများအတွက် (၁)၊ (၂) နံပါတ်စဉ် တပ်မည်
#             myanmar_num = to_burmese_number(item_counter)
#             items_html += (
#                 f'<div style="background-color: #1E293B; color: #F8FAFC;'
#                 " padding: 10px 14px; margin-bottom: 8px; border-radius: 6px;"
#                 " border-left: 2px solid #618487; font-size: 14px; line-height:"
#                 ' 1.6; display: flex; align-items: flex-start;">'
#                 '<span style="font-weight: bold; color: #618487; min-width: 45px;'
#                 f' display: inline-block;">({myanmar_num})</span>'
#                 f'<span style="flex-grow: 1;">{line}</span></div>'
#             )
#             item_counter += 1  # အချက်အလက် စာကြောင်းများမှသာ နံပါတ်စဉ်ကို ၁ တိုးမည်

#     container_html = (
#         '<div style="background-color: #020617; padding: 12px; border-radius:'
#         " 10px; border: 1px solid #1E293B; max-height: 400px; overflow-y:"
#         f' auto;">{items_html}</div>'
#     )

#     st.markdown(container_html, unsafe_allow_html=True)

def to_burmese_number(num: int) -> str:
    """အင်္ဂလိပ် ဂဏန်းများကို မြန်မာ ဂဏန်းသို့ ပြောင်းပေးသော Function"""
    burmese_digits = "၀၁၂၃၄၅၆၇၈၉"
    return "".join(burmese_digits[int(d)] for d in str(num))


def display_formatted_extracted_text(extracted_text: str):
    """CV Extracted Text များကို Key နဲ့ Value ပေါင်းစပ်ပြီး (၁)၊ (၂) မြန်မာ နံပါတ်စဉ်တပ် ပြသပေးသော Function"""
    if not extracted_text or not str(extracted_text).strip():
        st.info("အချက်အလက်များ မရှိသေးပါ။")
        return

    raw_lines = [
        line.strip()
        for line in str(extracted_text).split("\n")
        if line.strip()
    ]

    # ခွဲထွက်နေသော Key နှင့် Value များကို တစ်ကြောင်းတည်း ဖြစ်အောင် ပေါင်းစပ်ခြင်း
    merged_lines = []
    i = 0
    while i < len(raw_lines):
        curr_line = raw_lines[i]

        # စာကြောင်းသည် ':' သို့မဟုတ် 'း' ဖြင့် ဆုံးပြီး နောက်ထပ် စာကြောင်း ရှိပါက အလိုအလျောက် ပေါင်းပေးမည်
        if (
            curr_line.endswith(":") or curr_line.endswith("：")
        ) and i + 1 < len(raw_lines):
            next_line = raw_lines[i + 1]

            # နောက်တစ်ကြောင်းသည် နောက်ထပ် Header/Label မဟုတ်ပါက တစ်ကြောင်းတည်း ပေါင်းမည်
            if not (next_line.endswith(":") or next_line.endswith("：")):
                merged_lines.append(f"{curr_line} {next_line}")
                i += 2
                continue

        merged_lines.append(curr_line)
        i += 1

    items_html = ""
    item_counter = 1

    for line in merged_lines:
        clean_line = line.lower()

        is_main_header = (
            "curriculum vitae" in clean_line
            or clean_line == "ကိုယ်ရေးအချက်အလက်"
        )

        if is_main_header:
            items_html += (
                f'<div style="color: #F8FAFC; font-weight: bold; font-size:'
                " 16px; padding: 12px 14px; margin-top: 10px; margin-bottom:"
                ' 10px; text-align: center; border-bottom: 1px solid #334155;">'
                f"{line}</div>"
            )
        else:
            myanmar_num = to_burmese_number(item_counter)
            items_html += (
                f'<div style="background-color: #1E293B; color: #F8FAFC;'
                " padding: 10px 14px; margin-bottom: 8px; border-radius: 6px;"
                " border-left: 3px solid #618487; font-size: 14px; line-height:"
                ' 1.6; display: flex; align-items: flex-start;">'
                '<span style="font-weight: bold; color: #618487; min-width: 45px;'
                f' display: inline-block;">({myanmar_num})</span>'
                f'<span style="flex-grow: 1;">{line}</span></div>'
            )
            item_counter += 1

    container_html = (
        '<div style="background-color: #020617; padding: 12px; border-radius:'
        " 10px; border: 1px solid #1E293B; max-height: 450px; overflow-y:"
        f' auto;">{items_html}</div>'
    )

    st.markdown(container_html, unsafe_allow_html=True)
# ============================================================
# DATASET QUESTION MATCHING LOGIC
# ============================================================
import os
import re
import random
import pandas as pd
import streamlit as st
def match_questions_from_dataset(
    parsed_data, dataset_path, target_tech_count=7, target_total_count=20
):
    if not os.path.exists(dataset_path):
        st.warning(f"Dataset Excel not found at {dataset_path}.")
        return pd.DataFrame()

    df = pd.read_excel(dataset_path)
    df.columns = df.columns.str.strip()

    # Determine CV Keywords Column
    target_col = "cv_keywords" if "cv_keywords" in df.columns else None
    if not target_col:
        for col in df.columns:
            if "cv" in col.lower() and "keyword" in col.lower():
                target_col = col
                break
    if not target_col:
        target_col = "cv_keywords"

    # Normalize answer column
    if "candidate_answer" in df.columns and "answer" not in df.columns:
        df["answer"] = df["candidate_answer"]

    # Remove duplicates from Dataset
    if "question" in df.columns:
        df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)

    extracted_entities = parsed_data.get("extracted_entities", [])

    # Dynamic CV info map for placeholders
    cv_info_map = {
        "NAME": "အကဲဖြတ်ခံယူသူ",
        "POSITION": "လျှောက်ထားသည့်ရာထူး",
        "EDUCATION": "ပညာအရည်အချင်း",
        "PROJECTS": "Projectများ",
    }

    tech_entities = []
    non_tech_entities = []
    candidate_name = None

    # Process Extracted Entities
    for ent in extracted_entities:
        label = str(ent.get("label", "")).strip()
        label_lower = label.lower()
        text = str(ent.get("text", "")).strip()

        if not text:
            continue

        # Check if entity is Father Name to avoid mistaking it as Candidate Name
        is_father = any(
            f_kw in label_lower for f_kw in ["father", "အဘ", "အဖ"]
        )

        # 1. Candidate Name matching (Ignoring father's name)
        if not is_father and not candidate_name:
            if label_lower in [
                "person",
                "အမည်",
                "name",
                "candidate_name",
                "applicant_name",
                "candidate",
            ]:
                candidate_name = text
                cv_info_map["NAME"] = candidate_name

        # 2. Other CV Info mappings
        if any(k in label_lower for k in ["ရာထူး", "position"]):
            cv_info_map["POSITION"] = text
        elif any(
            k in label_lower for k in ["ပညာ", "education", "qualification"]
        ):
            cv_info_map["EDUCATION"] = text
        elif "project" in label_lower:
            cv_info_map["PROJECTS"] = text

        # Skip Father Name from Non-Tech matching entities
        if is_father:
            continue

        # Categorize into Tech vs Non-Tech Entities
        if label_lower in TECH_SKILL_LABELS or any(
            kw in label_lower for kw in ["skill", "tech", "tool", "language"]
        ):
            if text not in tech_entities:
                tech_entities.append(text)
        else:
            if (label, text) not in non_tech_entities:
                non_tech_entities.append((label, text))

    def format_question_text(q_raw):
        if pd.isna(q_raw):
            return ""
        q_str = str(q_raw)
        replacements = {
            "{အမည်}": cv_info_map["NAME"],
            "{Name}": cv_info_map["NAME"],
            "{name}": cv_info_map["NAME"],
            "{လျှောက်ထားလိုသည့်ရာထူး}": cv_info_map["POSITION"],
            "{Position}": cv_info_map["POSITION"],
            "{position}": cv_info_map["POSITION"],
            "{ပညာအရည်အချင်း}": cv_info_map["EDUCATION"],
            "{Education}": cv_info_map["EDUCATION"],
            "{Projectများ}": cv_info_map["PROJECTS"],
            "{Projects}": cv_info_map["PROJECTS"],
        }
        for placeholder, val in replacements.items():
            q_str = q_str.replace(placeholder, val)
        return q_str

    selected_indices = set()
    selected_rows = []

    # ------------------------------------------------------------
    # 1. Technical Skills (Target: 7 Questions)
    # ------------------------------------------------------------
    tech_added = 0
    for tech in tech_entities:
        if tech_added >= target_tech_count:
            break
        pattern = r"\b" + re.escape(tech.lower()) + r"\b"
        matched = []
        for idx, row in df.iterrows():
            if idx in selected_indices:
                continue
            cv_kw = str(row.get(target_col, "")).lower().strip()
            if cv_kw and re.search(pattern, cv_kw):
                matched.append((idx, row))

        if matched:
            chosen_idx, chosen_row = random.choice(matched)
            selected_indices.add(chosen_idx)
            row_copy = chosen_row.copy()
            row_copy["question"] = format_question_text(row_copy["question"])
            selected_rows.append(row_copy)
            tech_added += 1

    # Fallback if tech entities extracted are fewer than target_tech_count
    if tech_added < target_tech_count:
        fallback_candidates = [
            (idx, row)
            for idx, row in df.iterrows()
            if idx not in selected_indices
        ]
        random.shuffle(fallback_candidates)
        needed = target_tech_count - tech_added
        for idx, row in fallback_candidates[:needed]:
            selected_indices.add(idx)
            row_copy = row.copy()
            row_copy["question"] = format_question_text(row_copy["question"])
            selected_rows.append(row_copy)

    # ------------------------------------------------------------
    # 2. Non-Tech Questions (Target: Fill remaining to reach ~20 Total)
    # ------------------------------------------------------------
    max_non_tech_needed = target_total_count - len(selected_rows)
    non_tech_added = 0

    for label, text in non_tech_entities:
        if non_tech_added >= max_non_tech_needed:
            break

        matched = []
        for idx, row in df.iterrows():
            if idx in selected_indices:
                continue
            cv_kw = str(row.get(target_col, "")).lower().strip()
            if cv_kw and (
                label.lower() in cv_kw
                or text.lower() in cv_kw
                or cv_kw
                in [
                    "အမည်",
                    "name",
                    "position",
                    "education",
                    "experience",
                    "project",
                    "general",
                ]
            ):
                matched.append((idx, row))

        if matched:
            chosen_idx, chosen_row = random.choice(matched)
            selected_indices.add(chosen_idx)
            row_copy = chosen_row.copy()
            row_copy["question"] = format_question_text(row_copy["question"])
            selected_rows.append(row_copy)
            non_tech_added += 1

    # Fallback if total questions are still less than 20
    if len(selected_rows) < target_total_count:
        needed = target_total_count - len(selected_rows)
        unused_rows = [
            (idx, row)
            for idx, row in df.iterrows()
            if idx not in selected_indices
        ]
        random.shuffle(unused_rows)
        for idx, row in unused_rows[:needed]:
            selected_indices.add(idx)
            row_copy = row.copy()
            row_copy["question"] = format_question_text(row_copy["question"])
            selected_rows.append(row_copy)

    # Return exactly target_total_count (20) questions
    selected_df = (
        pd.DataFrame(selected_rows).head(target_total_count).reset_index(drop=True)
    )
    return selected_df
# ============================================================
# CACHED MODEL LOADERS & EVALUATION FUNCTIONS
# ============================================================
@st.cache_resource
def load_nlp_spacy():
    nlp = spacy.load("my_myanmar_ner_model", exclude=["tokenizer"])
    nlp.tokenizer = MyanmarTokenizer(nlp.vocab)
    return nlp

@st.cache_resource
def load_xlm_roberta():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = MultiOutputRegressor(MODEL_NAME).to(device)
    if os.path.exists(SAVE_MODEL_PATH):
        model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=device))
    model.eval()
    return tokenizer, model

def predict_single_with_guardrail(question, model_keywords, candidate_answer, tokenizer, model):
    is_valid, err_msg = validate_input_text(candidate_answer)
    if not is_valid:
        return {
            'target_accuracy': 0.0,
            'target_sentiment': 0.0,
            'target_anxiety': 0.9,
            'error': err_msg,
        }

    combined_text = f'Question: {question} | Expected Keywords: {model_keywords} | Answer: {candidate_answer}'
    inputs = tokenizer(
        combined_text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    ).to(device)

    with torch.no_grad():
        preds = model(inputs['input_ids'], inputs['attention_mask']).cpu().numpy()[0]

    return {
        "target_accuracy": round(float(preds[0]), 4),
        "target_sentiment": round(float(preds[1]), 4),
        "target_anxiety": round(float(preds[2]), 4),
    }

def process_uploaded_pdf(uploaded_file, nlp_model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    raw_text = extract_text_from_pdf(tmp_path)
    os.remove(tmp_path)

    cleaned_text = merge_multiline_text(raw_text)
    parsed_data = parse_cv_complete(cleaned_text, nlp_model)
    return cleaned_text, parsed_data

# ============================================================
# INITIALIZE MODELS
# ============================================================
nlp_model = load_nlp_spacy()
tokenizer, xlm_model = load_xlm_roberta()

# ============================================================
# NAVIGATION (STYLE FROM APP.PY)
# ============================================================
query = st.query_params
current_page = query.get("page", "Interviewer")

nav_html = f"""
<div style="display: flex; justify-content: center; gap: 60px; padding: 25px 0px; margin-bottom: 20px;">
    <a href="/?page=Interviewer" style="text-decoration: none; font-size: 16px; font-weight: {'800' if current_page == 'Interviewer' else '500'}; color: {'#111A1C' if current_page == 'Interviewer' else '#7B9294'}; transition: color 0.2s;">Interviewer Module</a>
    <a href="/?page=Interviewee" style="text-decoration: none; font-size: 16px; font-weight: {'800' if current_page == 'Interviewee' else '500'}; color: {'#111A1C' if current_page == 'Interviewee' else '#7B9294'}; transition: color 0.2s;">Interviewee Practice</a>
</div>
"""
st.markdown(nav_html, unsafe_allow_html=True)

# ============================================================
# INTERVIEWER PAGE (RECRUITER TOOL)
# ============================================================
if current_page == "Interviewer":
    st.markdown('<div class="giant-title">CVအချက်အလက်များ စစ်ဆေးသည့် ဒက်ရှ်ဘုတ်</div>', unsafe_allow_html=True)
    st.markdown('<div class="giant-subtitle">အင်တာဗျူးမေးသူများ (Interviewer) အတွက် လျှောက်ထားသူ၏ CV ကို စစ်ဆေးပြီး သင့်လျော်သော မေးခွန်းများကို ထုတ်ပေးသည့် စနစ်။</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1], gap="large")
    with col1:
        st.markdown('<div style="color: #111A1C; font-size: 18px; font-weight: 700; margin-bottom: 15px;">လျှောက်ထားသူ၏ CV တင်သွင်းရန်</div>', unsafe_allow_html=True)
        uploaded_cv = st.file_uploader("CV / PDF တင်ရန်", type=["pdf"], key="interviewer_cv", label_visibility="collapsed")
        
        if uploaded_cv:
            if "interviewer_extracted_text" not in st.session_state or st.session_state.get("interviewer_file_name") != uploaded_cv.name:
                with st.spinner("CV ဖိုင်အား စစ်ဆေးဖတ်ရှုနေပါသည်..."):
                    raw_txt, parsed_info = process_uploaded_pdf(uploaded_cv, nlp_model)
                    st.session_state["interviewer_extracted_text"] = raw_txt
                    st.session_state["interviewer_parsed_info"] = parsed_info
                    st.session_state["interviewer_file_name"] = uploaded_cv.name

            st.success(f"{uploaded_cv.name} တင်ပြီးပါပြီ")
            # st.text_area("CV တွင်ပါရှိသော အချက်အလက်များ (Extracted Text)", st.session_state["interviewer_extracted_text"], height=400, key="cv_white_box")
            # ⭐️ ဒီနေရာမှာ Function ကို လှမ်းခေါ်ပေးရပါမည် ⭐️
            display_formatted_extracted_text(
                st.session_state["interviewer_extracted_text"]
            )
        else:
            st.text_area("လျှောက်ထားသူ၏ CV အချက်အလက်များ", "လျှောက်ထားသူ၏ CV အချက်အလက်များကို ကြည့်ရှုရန် CV ဖိုင်ကို တင်သွင်းပါ။", height=250, disabled=True)
    
    with col2:
        st.markdown("""
        <div class="solid-teal-card">
            <div class="solid-teal-title">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                Entity ဖြင့်ရှာဖွေရန်
            </div>
            <div class="solid-teal-text">လျှောက်ထားသူ၏ CV ထဲတွင် ပါဝင်သော နည်းပညာအချက်အလက်များ၊ ပညာအရည်အချင်း သို့မဟုတ် Keyword တစ်ခုကို ထည့်သွင်း၍ မေးခွန်းထုတ်ပါ။</div>
        </div>
        """, unsafe_allow_html=True)
        
        entity = st.text_input("Entity Keyword", placeholder="e.g. Python, Machine Learning, Django...", label_visibility="collapsed")

        if st.button("မေးခွန်းများ ရှာရန်", use_container_width=True):
            if entity.strip():
                st.markdown(f'<div style="margin-top: 20px; font-weight: 700; color: #111A1C;">\'{entity}\' အတွက် ထုတ်ပေးထားသော မေးခွန်းများ</div>', unsafe_allow_html=True)
                
                # Dynamic Filter from Dataset if available
                if os.path.exists(DATASET_PATH):
                    df_dataset = pd.read_excel(DATASET_PATH)
                    target_col = "cv_keywords" if "cv_keywords" in df_dataset.columns else df_dataset.columns[0]
                    
                    # 1. Filter rows matching keyword
                    matched_q = df_dataset[df_dataset[target_col].astype(str).str.contains(entity, case=False, na=False)]
                    
                    # 2. Remove duplicate questions
                    if "question" in matched_q.columns:
                        matched_q = matched_q.drop_duplicates(subset=["question"])
                    
                    if not matched_q.empty:
                        for i, (_, row) in enumerate(matched_q.head(4).iterrows(), 1):
                            st.markdown(f"""
                            <div class="question-box" style="padding: 15px 20px;">
                                <div class="question-number">QUESTION {i}</div>
                                <div class="question-text" style="font-size: 15px;">{row['question']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info(f"'{entity}' နှင့် ပတ်သက်သော မေးခွန်း သီးသန့် မတွေ့ရှိပါ။ ဥပမာ မေးခွန်းများကို ပြသပေးနေပါသည်။")
                        matched_q = pd.DataFrame()
                else:
                    matched_q = pd.DataFrame()

                if matched_q.empty:
                    related_questions = [
                        f"How have you structured your {entity} projects?",
                        f"What are the primary advantages you've found utilizing {entity} over alternatives?",
                        f"What technical challenges have you faced while working with {entity}?"
                    ]
                    for i, q in enumerate(related_questions, 1):
                        st.markdown(f"""
                        <div class="question-box" style="padding: 15px 20px;">
                            <div class="question-number">QUESTION {i}</div>
                            <div class="question-text" style="font-size: 15px;">{q}</div>
                        </div>
                        """, unsafe_allow_html=True)

# ============================================================
# INTERVIEWEE PAGE (PRACTICE PORTAL)
# ============================================================
elif current_page == "Interviewee":
    st.markdown('<div class="giant-title">အင်တာဗျူး အစမ်းလေ့ကျင့်မှု (Interviewee)</div>', unsafe_allow_html=True)
    st.markdown('<div class="giant-subtitle">အလုပ်လျှောက်မည့်သူများ (Interviewee) ကိုယ်တိုင် နည်းပညာအင်တာဗျူး မေးခွန်းများကို လေ့ကျင့်ပြီး AI ဖြင့် အသေးစိတ် စွမ်းဆောင်ရည် သုံးသပ်ချက် ရယူရန် စနစ်။</div>', unsafe_allow_html=True)

    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown('<div style="color: #111A1C; font-size: 18px; font-weight: 700; margin-bottom: 15px;">စတင်ရန် သင့် CV ကို တင်ပါ</div>', unsafe_allow_html=True)
    interviewee_cv = st.file_uploader("သင့် CV ကို တင်ပါ", type=["pdf"], key="interviewee_cv", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if interviewee_cv:
        # Check if questions already matched for this file
        if "interviewee_questions_df" not in st.session_state or st.session_state.get("interviewee_file_name") != interviewee_cv.name:
            with st.spinner("CV အား Parsing လုပ်ပြီး သင့်လျော်သော မေးခွန်းများ ရွေးချယ်နေပါသည်..."):
                _, parsed_info = process_uploaded_pdf(interviewee_cv, nlp_model)
                # q_df = match_questions_from_dataset(parsed_info, DATASET_PATH, total_q=7, tech_q_count=5, non_tech_q_count=2)
                q_df = match_questions_from_dataset(parsed_info, DATASET_PATH, target_tech_count=7)
                st.session_state["interviewee_questions_df"] = q_df
                st.session_state["interviewee_file_name"] = interviewee_cv.name

        questions_df = st.session_state["interviewee_questions_df"]
        total_questions_count = len(questions_df)

        st.markdown("### နည်းပညာ စစ်ဆေးမှု (Technical Assessment)")
        st.caption(f"အောက်ပါ မေးခွန်း {total_questions_count} ခုလုံးကို ဖြေဆိုပါ။ သင်၏ ဖြေဆိုချက်များကို တိကျမှု (Accuracy)၊ စိတ်ခံစားချက် (Sentiment) နှင့် စိုးရိမ်ပူပန်မှု (Anxiety) တို့ဖြင့် ဆန်းစစ်ပေးမည် ဖြစ်သည်။")

        answers = []
        for i, row in questions_df.iterrows():
            q_text = row["question"]
            st.markdown(f"""
            <div class="question-box">
                <div class="question-number">မေးခွန်း {i + 1} / {total_questions_count}</div>
                <div class="question-text">{q_text}</div>
            </div>
            """, unsafe_allow_html=True)
            
            ans = st.text_area(
                f"သင့်အဖြေ {i + 1}",
                placeholder="ဤနေရာတွင် သင့်အဖြေကို ရေးပါ...",
                height=120,
                key=f"answer_input_{i}",
                label_visibility="collapsed"
            )
            answers.append(ans)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("အင်တာဗျူး သုံးသပ်ချက် ပေးပို့ရန်", use_container_width=True):
            if any(answer.strip() for answer in answers):
                with st.spinner("သင့်အင်တာဗျူး စွမ်းဆောင်ရည်ကို XLM-RoBERTa Model ဖြင့် စစ်ဆေးနေပါပြီ..."):
                    eval_results = []
                    tot_acc, tot_sent, tot_anx = 0.0, 0.0, 0.0

                    for i, row in questions_df.iterrows():
                        q_text = row["question"]
                        kw_text = str(row.get("model_keywords", ""))
                        
                        # Check for the candidate_answer column and handle empty/NaN values
                        raw_ref_ans = row.get("candidate_answer")
                        if pd.isna(raw_ref_ans) or str(raw_ref_ans).strip() in ["", "nan", "None"]:
                            ref_ans = "အကိုးအကား အဖြေမရှိပါ။"
                        else:
                            ref_ans = str(raw_ref_ans).strip()

                        user_ans = answers[i]

                        res = {}

                        # Check if answer is meaningful before scoring
                        if not is_meaningful_answer(user_ans):
                            acc = 0.0
                            sent = 0.0
                            anx = 90.0
                            res = {
                                    "error": (
                                        "အဓိပ္ပာယ်ရှိသော အဖြေမဟုတ်သည့်အတွက် စစ်ဆေးမှု"
                                        " မပြုလုပ်ပါ။"
                                    )
                                }
                        else:
                            res = predict_single_with_guardrail(q_text, kw_text, user_ans, tokenizer, xlm_model)
                            if res.get("guardrail_triggered"):
                                acc = 0.0
                                sent = 0.0
                                anx = 90.0
                            else:
                                acc = res["target_accuracy"] * 100
                                sent = res["target_sentiment"] * 100
                                anx = res["target_anxiety"] * 100

                        tot_acc += acc
                        tot_sent += sent
                        tot_anx += anx
                        
                        eval_results.append({
                            "question": q_text,
                            "user_answer": user_ans,
                            "reference_answer": ref_ans,
                            "accuracy": round(acc, 1),
                            "sentiment": round(sent, 1),
                            "anxiety": round(anx, 1),
                            "error": res.get("error")
                        })

                num_q = len(eval_results)
                avg_acc = round(tot_acc / num_q) if num_q > 0 else 0
                avg_sent = round(tot_sent / num_q) if num_q > 0 else 0
                avg_anx = round(tot_anx / num_q) if num_q > 0 else 0
                
                # Estimated Job Probability Logic based on evaluated metrics
                job_probability = max(0, min(100, round((avg_acc * 0.5) + (avg_sent * 0.4) + ((100 - avg_anx) * 0.1))))

                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown('<div class="giant-title" style="font-size: 40px;">စစ်ဆေးမှု ရလဒ်များ</div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="solid-teal-card" style="text-align: center; margin: 30px 0px;">
                    <div style="font-size: 14px; font-weight: 700; letter-spacing: 2px; color: #92AEB0; margin-bottom: 10px;">အလုပ်ရနိုင်ခြေ ခန့်မှန်းချက် (ESTIMATED JOB PROBABILITY)</div>
                    <div style="font-size: 65px; font-weight: 800; line-height: 1;">{job_probability}%</div>
                    <div style="font-size: 18px; font-weight: 600; margin-top: 15px;">{'အလုပ်ခန့်အပ်ရန် အလားအလာ ကောင်းမွန်သူ' if job_probability >= 60 else 'ထပ်မံ လေ့ကျင့်ရန် လိုအပ်သူ'}</div>
                    <div style="color: #D5E2E3; font-size: 14px; margin-top: 5px;">သင်၏ နည်းပညာ တိကျမှုနှင့် ဆက်သွယ်ရေး ပုံစံများအပေါ် အခြေခံထားပါသည်။</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### ခြုံငုံ စွမ်းဆောင်ရည် တိုင်းတာချက်များ")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-icon">
                            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
                        </div>
                        <div class="metric-value">{avg_acc}%</div>
                        <div class="metric-label">တိကျမှု (Accuracy)</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-icon">
                            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>
                        </div>
                        <div class="metric-value">{avg_sent}%</div>
                        <div class="metric-label">စိတ်ခံစားချက် (Sentiment)</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-icon">
                            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
                        </div>
                        <div class="metric-value">{avg_anx}%</div>
                        <div class="metric-label">စိုးရိမ်ပူပန်မှု အဆင့်</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("### မေးခွန်း တစ်ခုချင်းစီ၏ အသေးစိတ် သုံးသပ်ချက်")

                for i, res_item in enumerate(eval_results):
                    st.markdown(f"""
                    <div class="white-card" style="padding: 25px; margin-bottom: 15px;">
                        <div style="color:#618487; font-size:12px; font-weight:800; letter-spacing:1.5px; margin-bottom: 10px;">မေးခွန်း {i + 1}</div>
                        <div style="color:#111A1C; font-size:18px; font-weight:700; line-height:1.5; margin-bottom: 20px;">{res_item["question"]}</div>
                    """, unsafe_allow_html=True)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f'<div class="detail-metric"><div class="detail-label">တိကျမှု (ACCURACY)</div><div class="detail-value">{res_item["accuracy"]}%</div></div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div class="detail-metric"><div class="detail-label">စိတ်ခံစားချက် (SENTIMENT)</div><div class="detail-value">{res_item["sentiment"]}%</div></div>', unsafe_allow_html=True)
                    with c3:
                        st.markdown(f'<div class="detail-metric"><div class="detail-label">စိုးရိမ်ပူပန်မှု (ANXIETY)</div><div class="detail-value">{res_item["anxiety"]}%</div></div>', unsafe_allow_html=True)

                    if res_item.get("error"):
                        st.warning(f"သတိပေးချက်: {res_item['error']}")

                    user_answer = res_item["user_answer"]
                    if user_answer.strip():
                        st.markdown(f"""
                        <div style="margin-top:20px; font-weight: 700; font-size: 14px; color: #4A6365;">သင့်အဖြေ:</div>
                        <div class="user-answer-card">{user_answer}</div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("သင်ဖြေထားသောအဖြေ မရှိပါ။")

                    with st.expander("Candidates များ၏ မှန်ကန်သော အဖြေကို ကြည့်ရန်"):
                        st.markdown(f'<div class="correct-answer-text">{res_item["reference_answer"]}</div>', unsafe_allow_html=True)
                        
                    st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.warning("မပေးပို့မီ အနည်းဆုံး မေးခွန်းတစ်ခုကို ဖြေဆိုပါ။")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#7B9294; font-size:13px; font-weight: 500;">
    © 2026 AI Technical Interview Evaluation System • All Rights Reserved
</div>
""", unsafe_allow_html=True)