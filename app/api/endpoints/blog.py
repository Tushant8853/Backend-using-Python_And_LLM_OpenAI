import zipfile
import os
import json
import google.generativeai as genai
from arango import ArangoClient
from asyncio import to_thread

from firecrawl import FirecrawlApp
from pydantic import BaseModel
import re

class FirecrawlRequest(BaseModel):
    url: str

# --- STEP 1: Extract ZIP ---
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print(f"[✓] Extracted files to: {extract_dir}")

# --- STEP 2: Configure Gemini ---
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('models/gemini-1.5-flash')

def generate_keywords(description: str) -> list:
    try:
        prompt = f"Generate 5 relevant keywords for this content:\n\n{description}"
        response = model.generate_content(prompt)
        lines = response.text.strip().splitlines()
        keywords = [line.strip("-• ").strip() for line in lines if line.strip()]
        return keywords[:5]
    except Exception as e:
        print(f"[!] Gemini error: {e}")
        return []

# --- STEP 3: Connect to ArangoDB ---
client = ArangoClient()
db = client.db(db_name, username=username, password=password)

for name in ['blogs', 'products']:
    if not db.has_collection(name):
        db.create_collection(name)

collection_map = {'blog': 'blogs', 'product': 'products'}

# --- STEP 4: Process Files ---
# --- STEP 4: Process and Insert ---
inserted = {'blogs': 0, 'products': 0}

for file in os.listdir(extract_dir):
    if not file.endswith('.json'):
        continue

    path = os.path.join(extract_dir, file)
    filename = file.lower()
    if 'blog' in filename:
        target = 'blogs'
    elif 'product' in filename:
        target = 'products'
    else:
        continue

    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            metadata = raw.get('metadata', {})
            title = metadata.get('title', '').strip()
            description = metadata.get('description', '').strip()
            source_url = metadata.get('url') or metadata.get('sourceURL', '')

            if not description:
                continue

            keywords = generate_keywords(description)

            doc = {
                'title': title,
                'description': description,
                'source_url': source_url,
                'keywords': keywords
            }

            db.collection(target).insert(doc)
            inserted[target] += 1
            print(f"[✓] Inserted 1 into '{target}' from {file}")

    except Exception as e:
        print(f"[!] Error processing {file}: {e}")

# --- Summary ---
print("\n✅ Import complete!")
for col, count in inserted.items():
    print(f" - {col}: {count} documents")