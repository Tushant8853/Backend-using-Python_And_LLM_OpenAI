from fastapi import HTTPException, APIRouter, Header
from pydantic import BaseModel
from firecrawl import FirecrawlApp
from arango import ArangoClient
from asyncio import to_thread
from datetime import datetime
import google.generativeai as genai
import re
import os
from app.core.config import settings

router = APIRouter()

# Initialize Firecrawl
firecrawl_app = FirecrawlApp(api_key=settings.FIRECRAWL_API_KEY)

# Connect to ArangoDB
client = ArangoClient()
db = client.db(
    os.getenv("ARANGO_DB_NAME"),
    username=os.getenv("ARANGO_USERNAME"),
    password=os.getenv("ARANGO_PASSWORD"),
    verify=True
)

# Ensure collections
for collection_name in ["companies", "products", "keywords", "blogs"]:
    if not db.has_collection(collection_name):
        db.create_collection(collection_name)

company_collection = db.collection("companies")
product_collection = db.collection("products")
keyword_collection = db.collection("keywords")
blogs_collection = db.collection("blogs")

class FirecrawlRequest(BaseModel):
    url: str

def clean_keywords(raw: str) -> list:
    raw = re.sub(r"(?m)^\s*[-•]\s*", "", raw)
    return [kw.strip() for kw in raw.replace("\n", ",").split(",") if kw.strip()]

async def extract_keywords_with_gemini(text: str, provider_key: str) -> list:
    try:
        genai.configure(api_key=provider_key)
        model = genai.GenerativeModel("models/gemini-1.5-flash")

        prompt = (
            "Extract 10 clear, concise keywords from the following text. "
            "Return only a comma-separated list with no explanations:\n\n"
            f"{text}"
        )

        response = await to_thread(model.generate_content, prompt)
        text_output = response.text.strip()
        return clean_keywords(text_output)

    except Exception as e:
        raise RuntimeError(f"Gemini keyword extraction failed: {e}")

@router.post("/scrape")
async def scrape_website(
    request: FirecrawlRequest,
    x_gemini_key: str = Header(default=None, convert_underscores=False)
):
    try:
        if not x_gemini_key:
            raise HTTPException(status_code=400, detail="Missing Gemini API key in 'X-Gemini-Key' header")

        # Crawl the website
        crawl_result = firecrawl_app.crawl_url(
            request.url,
            params={
                "limit": 100,
                "includePaths": ["/products/.+", "/blog/.+", "/news/.+"],
                "excludePaths": [],
                "scrapeOptions": {
                    "formats": ["markdown"]
                }
    })

        pages = crawl_result.get("data", [])
        if not pages:
            raise HTTPException(status_code=404, detail="No pages found during crawl.")

        scraping_date = datetime.utcnow().isoformat()
        company_data = {}
        products = []
        blogs = []

        for page in pages:
            page_url = page.get("url", "")
            metadata = page.get("metadata", {})
            title = metadata.get("title", "")
            description = metadata.get("description", "")

            # Extract company-level info from the base URL
            if page_url.rstrip('/') == request.url.rstrip('/'):
                company_data = {
                    "company_name": title,
                    "title": title,
                    "description": description,
                    "og_url": metadata.get("og:url", ""),
                    "url": page_url,
                    "twitter_url": metadata.get("twitter:site", ""),
                    "scraping_date": scraping_date
                }
                # Insert into ArangoDB
                if not company_collection.has(company_data["url"]):
                    company_collection.insert(company_data)

            # Identify product pages
            elif "/products/" in page_url:
                # Extract keywords using Gemini
                keywords = await extract_keywords_with_gemini(description, x_gemini_key)
                product_data = {
                    "product_name": title,
                    "title": title,
                    "description": description,
                    "url": page_url,
                    "keywords": keywords,
                    "scraping_date": scraping_date
                }
                products.append(product_data)
                # Insert into ArangoDB
                if not product_collection.has(product_data["url"]):
                    product_collection.insert(product_data)

            # Identify blog pages
            elif "/blog/" in page_url or "/news/" in page_url:
                # Extract keywords using Gemini
                keywords = await extract_keywords_with_gemini(description, x_gemini_key)
                blog_data = {
                    "blog_url": page_url,
                    "title": title,
                    "description": description,
                    "keywords": keywords,
                    "scraping_date": scraping_date
                }
                blogs.append(blog_data)
                # Insert into ArangoDB
                if not blogs_collection.has(blog_data["blog_url"]):
                    blogs_collection.insert(blog_data)

        return {
            "message": "Scraping and data extraction completed successfully.",
            "company_data": company_data,
            "products": products,
            "blogs": blogs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
