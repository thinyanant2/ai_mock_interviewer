
import json

with open('cv_records_125.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

converted_data = []

# enumerate က index ကို 0 ကနေစပေးလို့ အလှည့်တိုင်းမှာ count လုပ်ရလွယ်စေပါတယ်
for index, item in enumerate(data, start=1):
    text_parts = []
    for key, val in item.items():
        if val:
            text_parts.append(f"{key} - {val}")
    
    full_text = " ၊ ".join(text_parts)
    converted_data.append({"text": full_text})
    
    # record ဦးရေ အများကြီးဆိုရင် ဘယ်လောက်ပြီးသွားပြီလဲဆိုတာ လှမ်းကြည့်လို့ရအောင်ပါ
    print(f"Processing item {index}...")

with open('label_studio_ready1.json', 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)

print(f"\nပြောင်းလဲခြင်း အောင်မြင်ပါပြီ! စုစုပေါင်း {len(converted_data)} records ပြီးစီးသွားပါပြီ။")