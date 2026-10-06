import json
import random
import re
from collections import Counter
from pathlib import Path

import pyidaungsu as pds
import spacy  # type: ignore
from sklearn.model_selection import train_test_split
from spacy.tokens import Doc, Span  # type: ignore
from spacy.training import Example  # type: ignore
from spacy.util import compounding, minibatch  # type: ignore

# ==============================================================================
# 1. Robust Myanmar Tokenizer (Splits Delimiters)
#  မြန်မာစာ NER Training မှာ Tokenizer ရဲ့ Alignment ဟာ အရေးအကြီးဆုံး ဖြစ်ပါတယ်။ ပုဒ်ဖြတ်ပုဒ်ရပ်တွေကြောင့် Boundary လွဲပြီး Entity တွေ ပျောက်မသွားစေဖို့ Custom Tokenizer ကို သုံးထားပါတယ်။
# ==============================================================================
class MyanmarTokenizer:

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        if not text:
            return Doc(self.vocab, words=[], spaces=[])

        raw_tokens = pds.tokenize(text, form="word")

        split_tokens = []
        for tok in raw_tokens:
            sub_toks = re.split(r'([/|:\-–—၊။,()\[\]{}\t\n\r"\'•*])', tok)
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

    def to_disk(self, path, **kwargs):
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    def from_disk(self, path, **kwargs):
        return self

    def to_bytes(self, **kwargs):
        return b""

    def from_bytes(self, bytes_data, **kwargs):
        return self


nlp = spacy.blank("xx")
nlp.tokenizer = MyanmarTokenizer(nlp.vocab)

# ==============================================================================
# 2. Dynamic Dynamic Offset Recovery & Conflict Resolver
#  Dataset ထဲမှာ Offset (Character Index) လွဲနေတာမျိုးနဲ့ Entity နေရာထပ်နေတာ (Overlap) များကို အလိုအလျောက် ပြင်ဆင်ပေးသည့် Utility Functions များ ဖြစ်ပါတယ်။
# ==============================================================================
PUNCT_AND_SPACE = " \t\n\r၊။;:()[]{}'\"/|-"

#လုပ်ဆောင်ချက်: Entity Span ရဲ့ အစ သို့မဟုတ် အဆုံးမှာ ပါဝင်နေသော မလိုအပ်သည့် Space များ၊ ကော်မာ၊ ပုဒ်ဖြတ်ပုဒ်ရပ် သင်္ကေတများကို ဖြတ်ထုတ်ပေးပါတယ်။
def trim_span(span):
    if span is None or len(span) == 0:
        return None

    start = span.start
    end = span.end

    while start < end and span.doc[start].text.strip(PUNCT_AND_SPACE) == "":
        start += 1

    while end > start and span.doc[end - 1].text.strip(PUNCT_AND_SPACE) == "":
        end -= 1

    if start < end:
        return Span(span.doc, start, end, label=span.label_)
    return None

# Dataset ထဲတွင် Offset လွဲမှားနေသော Entity များ (ဥပမာ - Index မှားပြီး NAME- ဦးမြင ဖြစ်နေပါက) doc.text ထဲတွင် စာသားအမှန် (entity_text) ကို Dynamic ပြန်လည် ရှာဖွေကာ အနီးစပ်ဆုံး Index အမှန်သို့ ပြန်လည် ပြင်ဆင်ပေးပါသည်။
def get_robust_doc_span_auto_fix(doc, entity_text, raw_start, raw_end, label):

    cleaned_ent_text = entity_text.strip(PUNCT_AND_SPACE)
    if not cleaned_ent_text:
        return None

    doc_text = doc.text

    # ၁။ Original Offset တိုင်း ကွက်တိကိုက်ညီပါက Direct Span ယူမည်
    if doc_text[raw_start:raw_end].strip() == cleaned_ent_text:
        span = doc.char_span(
            raw_start, raw_end, label=label, alignment_mode="expand"
        )
        if span:
            return trim_span(span)

    # ၂။ Offset လွဲနေပါက Document ထဲတွင် `cleaned_ent_text` တည်နေရာများကို လိုက်ရှာမည်
    matches = [
        m.start()
        for m in re.finditer(re.escape(cleaned_ent_text), doc_text)
    ]
    if not matches:
        return None

    # raw_start (မူလအမှား Index) နှင့် အနီးဆုံး တည်နေရာကို dynamic ပြန်လည် ရွေးချယ်မည်
    best_start = min(matches, key=lambda pos: abs(pos - raw_start))
    best_end = best_start + len(cleaned_ent_text)

    span = doc.char_span(
        best_start, best_end, label=label, alignment_mode="expand"
    )
    if span:
        return trim_span(span)

    return None

#Document တစ်ခုတည်းမှာ Entity နှစ်ခု နေရာထပ်နေပါက (Overlapping Spans) Inner/Specific Entities များကို ဦးစားပေး၍ Overlap ရှင်းပေးပါသည်။
def resolve_span_conflicts(spans):
    if not spans:
        return []

    sorted_spans = sorted(spans, key=lambda s: (len(s.text), s.start))
    selected_spans = []
    occupied_tokens = set()

    for span in sorted_spans:
        span_tokens = set(range(span.start, span.end))
        if not (span_tokens & occupied_tokens):
            selected_spans.append(span)
            occupied_tokens.update(span_tokens)

    return sorted(selected_spans, key=lambda s: s.start)


# ==============================================================================
# 3. Data Preprocessing
# JSON Dataset ကို ဖတ်ယူပြီး SpaCy မော်ဒယ် သင်ယူနိုင်သည့် Example Format သို့ ပြောင်းလဲပေးသည့် အဆင့်ဖြစ်ပါသည်။
# ==============================================================================
print("Data များ Load လုပ်ပြီး Preprocessing စတင်နေပါပြီ...")

DATASET_PATH = "cleaned_dataset1.json"

try:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
except FileNotFoundError:
    print(
        f"File '{DATASET_PATH}' ကို ရှာမတွေ့ပါ။ File နာမည် မှန်မမှန် စစ်ဆေးပေးပါ။"
    )
    exit()

EXCLUDE_LABELS = {"SUMMARY", "SOFT_SKILLS", "PROJECTS", "AWARDS"}

TRAIN_EXAMPLES = []

for entry in dataset:
    raw_text = entry.get("text", "")
    doc = nlp(raw_text)

    spans = []
    entities_list = entry.get("entities", entry.get("label", []))

    for annotation in entities_list:
        label = annotation["label"]
        if label in EXCLUDE_LABELS:
            continue

        start = annotation["start"]
        end = annotation["end"]
        ent_text = annotation.get("text", "")

        # Auto Offset Fixer ဖြင့် Span ဆွဲယူခြင်း
        span = get_robust_doc_span_auto_fix(doc, ent_text, start, end, label)
        if span is not None:
            spans.append(span)

    resolved_spans = resolve_span_conflicts(spans)
    entities = [
        (span.start_char, span.end_char, span.label_)
        for span in resolved_spans
    ]

    #Tokenized Doc နှင့် စိစစ်ပြီးသား Entities (Start, End, Label) များကို ပေါင်းစည်းကာ SpaCy NER Pipe သို့ ထည့်သွင်းရန် Training Instance တစ်ခု ဖန်တီးပေးသည်။
    if entities:
        try:
            example = Example.from_dict(doc, {"entities": entities})
            TRAIN_EXAMPLES.append(example)
        except Exception:
            continue

# ==============================================================================
# 4. Label Counts Output & Training Setup
# ==============================================================================
label_counts = Counter()
for example in TRAIN_EXAMPLES:
    for ent in example.reference.ents:
        label_counts[ent.label_] += 1

print("\n--- Dataset ထဲရှိ Label အရေအတွက်များ ---")
for label, count in label_counts.most_common():
    print(f" {label:35s} : {count:5d} ခု")

train_data, test_data = train_test_split(
    TRAIN_EXAMPLES, test_size=0.2, random_state=42
)
print(f"\nTraining Data အရေအတွက်: {len(train_data)} ခု")
print(f"Testing Data အရေအတွက် : {len(test_data)} ခု")

if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)

for example in TRAIN_EXAMPLES:
    for ent in example.reference.ents:
        ner.add_label(ent.label_)

#Test Set ပေါ်တွင် Precision, Recall နှင့် F1-Score များကို တွက်ချက်ပေးပါသည်။
def evaluate_model(nlp_model, data): #data (List[spacy.training.Example]): စမ်းသပ်မည့် Test Data Examples များ။
    eval_examples = []
    for gold_example in data:
        pred_doc = nlp_model(gold_example.text)
        eval_examples.append(Example(pred_doc, gold_example.reference))
    return nlp_model.evaluate(eval_examples)


# ==============================================================================
# 5. Training Loop
# ==============================================================================
optimizer = nlp.initialize(lambda: train_data)
epochs = 20

print("\n--- Training Loop စတင်ပါပြီ ---")

for epoch in range(epochs):
    random.shuffle(train_data)
    losses = {}

    #Training Data များကို Batch အဖြစ် ခွဲပေးပါသည်။ compounding သုံးထား၍ Batch Size သည် အစတွင် 4 မှစပြီး တဖြည်းဖြည်း 32 အထိ တိုးသွားပါမည်
    batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))

    #Batch တစ်ခုစာ Data ဖြင့် Model ၏ Weights များကို Update ပြုလုပ်ပေးသည့် အဓိက Training Function ဖြစ်ပါသည်။
    for batch in batches:
        nlp.update(batch, drop=0.2, losses=losses, sgd=optimizer)

    scores = evaluate_model(nlp, test_data)
    p = scores["ents_p"] * 100
    r = scores["ents_r"] * 100
    f1 = scores["ents_f"] * 100

    print(
        f"Epoch {epoch + 1:2d}/{epochs} - Loss: {losses.get('ner', 0.0):10.2f} | Precision: {p:6.2f}% | Recall: {r:6.2f}% | F1-Score: {f1:6.2f}%"
    )

# # ==============================================================================
# # 6. Final Evaluation & Save
# # ==============================================================================
# final_scores = evaluate_model(nlp, test_data)

# print("\n" + "=" * 65)
# print("FINAL OVERALL MODEL ACCURACY (TEST SET)")
# print("=" * 65)
# print(f"Overall Precision : {final_scores['ents_p'] * 100:.2f} %")
# print(f"Overall Recall    : {final_scores['ents_r'] * 100:.2f} %")
# print(f"Overall F1-Score  : {final_scores['ents_f'] * 100:.2f} %")
# print("=" * 65)

# nlp.to_disk("my_myanmar_ner_model")
# print("\n Model saved successfully!")
# ==============================================================================
# 6. Final Evaluation & Save
# ==============================================================================
final_scores = evaluate_model(nlp, test_data)

print("\n" + "=" * 65)
print("FINAL EVALUATION RESULTS PER ENTITY (TEST SET)")
print("=" * 65)
print(f"{'Entity Label':25s} | {'Precision':10s} | {'Recall':10s} | {'F1-Score':10s}")
print("-" * 65)

# ents_per_type holds detailed metrics per label: {label: {'p': val, 'r': val, 'f': val}}
for label, metrics in sorted(final_scores.get("ents_per_type", {}).items()):
    p = metrics["p"] * 100
    r = metrics["r"] * 100
    f1 = metrics["f"] * 100
    print(f"{label:25s} | {p:9.2f}% | {r:9.2f}% | {f1:9.2f}%")

print("=" * 65)
print("FINAL OVERALL MODEL ACCURACY (TEST SET)")
print("=" * 65)
print(f"Overall Precision : {final_scores['ents_p'] * 100:.2f} %")
print(f"Overall Recall    : {final_scores['ents_r'] * 100:.2f} %")
print(f"Overall F1-Score  : {final_scores['ents_f'] * 100:.2f} %")
print("=" * 65)

nlp.to_disk("my_myanmar_ner_model")
print("\nModel saved successfully!")


