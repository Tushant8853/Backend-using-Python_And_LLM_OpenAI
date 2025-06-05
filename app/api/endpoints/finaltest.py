import os
import json
from fastapi import APIRouter,Body
from datetime import datetime
from arango import ArangoClient
from firecrawl import FirecrawlApp 
from asyncio import to_thread
from pydantic import BaseModel
import google.generativeai as genai
from app.models.schemas import FirecrawlRequest
router = APIRouter()

# Initialize ArangoDB connection
client = ArangoClient()
db = client.db("company_scraping", username="root", password="openSesame")

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Ensure collections exist
def ensure_collections():
    required_docs = ["company", "products", "blogs", "keywords"]
    required_edges = ["has_products_module", "has_blogs", "has_keywords"]

    for name in required_docs:
        if not db.has_collection(name):
            db.create_collection(name)

    for name in required_edges:
        if not db.has_collection(name):
            db.create_collection(name, edge=True)

ensure_collections()

# Keyword generator using Gemini
def generate_keywords(text: str) -> list:
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    prompt = f"Extract 25 relevant keywords from the following content:\n\n{text}"
    try:
        response = model.generate_content(prompt)
        raw_output = response.text.strip()
        keywords = [kw.strip(" -•:") for kw in raw_output.splitlines() if kw.strip()]
        return keywords
    except Exception as e:
        print(f"Gemini error: {e}")
        return []

# Insert document if not already present
def insert_unique(collection_name, doc, unique_field):
    coll = db.collection(collection_name)
    existing = coll.find({unique_field: doc[unique_field]})
    for e in existing:
        return e
    return coll.insert(doc)

# Create an edge if it doesn't exist
def link_edge(edge_name, from_id, to_id):
    edge = db.collection(edge_name)
    existing = list(edge.find({"_from": from_id, "_to": to_id}))
    if not existing:
        edge.insert({"_from": from_id, "_to": to_id})



firecrawl = FirecrawlApp() 

# Endpoint: Scrape company metadata
@router.post("/company")
async def scrape_with_sdk(request: FirecrawlRequest):
    result = await to_thread(firecrawl.scrape_url, request.url)
    metadata = result.get("metadata", {})

    def extract_first(val):
        return val[1] if isinstance(val, list) and len(val) > 1 else val[0] if isinstance(val, list) else val

    company_name = "Nimblework"
    title = extract_first(metadata.get("title") or metadata.get("og:title") or "No title")
    description = extract_first(metadata.get("description") or metadata.get("ogDescription") or metadata.get("og:description") or "No description")
    og_url = extract_first(metadata.get("og:url") or metadata.get("ogUrl") or "No og:url")
    canonical_url = metadata.get("url") or "No URL provided"
    twitter_url = extract_first(metadata.get("twitter:site") or "No URL provided")
    scraping_date = datetime.utcnow().isoformat()

    # Insert company doc
    company_doc = insert_unique("company", {
        "company_name": company_name,
        "title": title,
        "description": description,
        "og_url": og_url,
        "twitter_url": twitter_url,
        "url": canonical_url,
        "scraping_date": scraping_date
    }, "url")

    return {"company_id": company_doc['_key']}

# File processing function
def process_json_files(directory_path, company_key):
    company_id = f"company/{company_key}"

    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if not filename.endswith(".json"):
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        metadata = data.get("metadata", {})

        def extract_first(val):
            return val[1] if isinstance(val, list) and len(val) > 1 else val[0] if isinstance(val, list) else val

        title = extract_first(metadata.get("title") or metadata.get("og:title") or "No title")
        description = extract_first(metadata.get("description") or metadata.get("ogDescription") or metadata.get("og:description") or "No description")
        url = extract_first(metadata.get("og:url") or metadata.get("ogUrl") or metadata.get("url") or "No URL")

        if "blog" in filename.lower():
            coll = "blogs"
            edge = "has_blogs"
        elif "product" in filename.lower():
            coll = "products"
            edge = "has_products_module"
        else:
            continue

        doc = insert_unique(coll, {
            "title": title,
            "description": description,
            "source_url": url
        }, "source_url")  # ignore the source_url 

        doc_id = f"{coll}/{doc['_key']}"
        link_edge(edge, company_id, doc_id)

        keywords = generate_keywords(description)
        db.collection(coll).update_match(
            {"_key": doc["_key"]},
            {"keywords": keywords}
        )
        for kw in keywords:
            kw_doc = insert_unique("keywords", {"value": kw}, "value")
            kw_id = f"keywords/{kw_doc['_key']}"
            link_edge("has_keywords", doc_id, kw_id)

    print(f"Processed JSONs for company {company_key}")

# Endpoint to process files
@router.post("/process-files")
def trigger_file_processing(company_key: str = Body(...), directory_path: str = Body(...)):
    try:
        process_json_files(directory_path, company_key)
        return {"message": f"Files processed for company {company_key}"}
    except Exception as e:
        return {"error": str(e)}