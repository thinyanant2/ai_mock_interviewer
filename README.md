# 🇲🇲 Myanmar Resume/CV Parser & AI Interview System

PDF Format ဖြင့်ရှိသော မြန်မာ/အင်္ဂလိပ် CV ဖိုင်များထဲမှ အချက်အလက်များကို **Named Entity Recognition (NER)** နှင့် **Natural Language Processing (NLP)** နည်းပညာများ အသုံးပြု၍ အလိုအလျောက် သီးခြားခွဲထုတ်ပေးပြီး နည်းပညာဆိုင်ရာ အင်တာဗျူး မေးခွန်းများ ထုတ်ပေးသည့် စနစ်ဖြစ်ပါသည်။


Key Features

- **PDF Text Extraction & Cleaning**: PyMuPDF (`fitz`) ကို အသုံးပြု၍ မြန်မာစာ Unicode သင်္ကေတများ ထပ်နေခြင်းနှင့် Format မမှန်ခြင်းများကို သန့်စင်ပေးခြင်း။
- **Custom NER Prediction**: Keras / spaCy Deep Learning Model ကို အသုံးပြု၍ အမည်၊ ရာထူး၊ ကုမ္ပဏီ၊ ဖုန်းနံပါတ် စသည့် Entity များကို တိကျစွာ ခွဲထုတ်ခြင်း။
- **Section Parsing**: CV ထဲရှိ *Summary, Work Experience, Projects, Awards, Soft Skills* စသည့် အခန်းများကို သီးခြား Section စာပိုဒ်များအဖြစ် ခွဲထုတ်ပေးခြင်း။
- **Dynamic Interview Generator**: NER မှ ရရှိလာသော နည်းပညာ Keyword များနှင့် ကိုက်ညီသည့် Technical Interview မေးခွန်းများကို Dataset ထဲမှ အလိုအလျောက် စုစည်းမေးမြန်းပေးခြင်း။

---

Tech Stack & Libraries

- **Language**: Python 3.x
- **Deep Learning / NLP**: TensorFlow / Keras, spaCy, Pyidaungsu (`pds`)
- **PDF Processing**: PyMuPDF (`fitz`), pdfplumber
- **Data Handling**: Pandas, NumPy
- **LLM Integration (Optional)**: Ollama (`qwen2.5:3b`)

---

