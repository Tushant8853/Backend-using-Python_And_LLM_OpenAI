from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
import httpx
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import tempfile
import fitz  # PyMuPDF
import gdown
import os
from firecrawl import FirecrawlApp
from app.core.config import settings

router = APIRouter()
firecrawl_app = FirecrawlApp(api_key=settings.FIRECRAWL_API_KEY)

# Request schema
class URLRequest(BaseModel):
    url: str

async def extract_web_text(url: str) -> str:
    page = firecrawl_app.scrape_url(
        url,
    )
    metadata = page.get("metadata")
    title = metadata.get("title") or metadata.get("og:title") or metadata.get("ogTitle") or "No title"
    description = metadata.get("description") or metadata.get("ogDescription") or metadata.get("og:description") or "No description"

    return f"""
        Title: {title}
        Description: {description}
        Content: {page.get("markdown")}
    """


# YouTube transcript handler
def extract_youtube_text(url: str) -> str:
    video_id = extract_youtube_id(url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)
    return "\n".join([entry.text for entry in transcript.snippets])

# Extract video ID from URL
def extract_youtube_id(url: str) -> str:
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query).get("v", [None])[0]
    elif "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")
    return None

# Dispatcher for scraping
async def dispatch_scraper(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if "youtube.com" in domain or "youtu.be" in domain:
        return extract_youtube_text(url)
    else:
        return await extract_web_text(url)

# --- Google Drive ---
# async def extract_google_drive_text(url: str) -> str:
#     if "document/d/" in url:  # Google Doc
#         async with httpx.AsyncClient() as client:
#             response = await client.get(url)
#             soup = BeautifulSoup(response.text, "html.parser")
#             return soup.get_text(separator="\n", strip=True)
#     elif "file/d/" in url or "uc?id=" in url:  # Google Drive PDF
#         file_id = extract_drive_file_id(url)
#         download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
#         async with httpx.AsyncClient() as client:
#             response = await client.get(download_url)
#             with open("temp.pdf", "wb") as f:
#                 f.write(response.content)
#         doc = fitz.open("temp.pdf")
#         text = "\n".join(page.get_text() for page in doc)
#         doc.close()
#         return text
#     raise Exception("Unsupported Google Drive format")

# def extract_drive_file_id(url: str) -> str:
#     if "file/d/" in url:
#         return url.split("/file/d/")[1].split("/")[0]
#     elif "uc?id=" in url:
#         return url.split("id=")[1].split("&")[0]
#     raise Exception("Invalid Google Drive file URL")

# --- Vimeo ---
async def extract_vimeo_text(url: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"https://vimeo.com/api/oembed.json?url={url}")
        data = r.json()
        return f"Vimeo: {data.get('title')} by {data.get('author_name')}\n{data.get('description')}"

# --- Wistia ---
# async def extract_wistia_text(url: str) -> str:
#     api_url = f"https://api.wistia.com/v1/medias/{url}.json"
#     async with httpx.AsyncClient() as client:
#         response = await client.get(api_url, auth=(WISTIA_API_KEY, ""))  # HTTP Basic Auth
#
#     if response.status_code == 401:
#         return {"error": "Unauthorized. Check your Wistia API key."}
#     elif response.status_code != 200:
#         return {"error": f"Failed to fetch Wistia media: {response.status_code}"}
#
#     data = response.json()
#     return {
#         "title": data.get("name"),
#         "description": data.get("description"),
#         "duration": data.get("duration"),
#     }
    
async def extract_google_doc_text(doc_url: str) -> str:
    options = Options()
    options.add_argument('--headless')  # run in background
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')

    # If chromedriver is not in PATH, specify executable_path
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(doc_url)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2)  # allow JavaScript to render

        text = driver.find_element(By.TAG_NAME, "body").text
        return text

    finally:
        driver.quit()




# gdrive_code_merge

async def extract_google_drive_text(url: str) -> str:
    if "document/d/" in url:  # Google Doc
        return await extract_google_doc_text(url)

    elif "file/d/" in url or "uc?id=" in url:  # File (PDF, audio, video, etc.)
        file_id = extract_drive_file_id(url)
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp_file:
            temp_path = tmp_file.name

        try:
            gdown.download(download_url, temp_path, quiet=False)

            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise HTTPException(status_code=500, detail="Downloaded file is empty or missing.")

            # Detect file type using file content
            # mime = magic.Magic(mime=True)
            # file_type = mime.from_file(temp_path)
            #
            # if "pdf" in file_type:
            #     return extract_text_from_pdf(temp_path)
            # elif "audio" in file_type or "video" in file_type:
            #     return "Audio/Video file detected. Transcription is not supported."
            # else:
            #     return f"Unsupported file type: {file_type}"

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    raise Exception("Unsupported Google Drive format")

def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    os.remove(file_path)
    return text

def extract_drive_file_id(url: str) -> str:
    if "file/d/" in url:
        return url.split("/file/d/")[1].split("/")[0]
    elif "uc?id=" in url:
        return url.split("id=")[1].split("&")[0]
    raise Exception("Invalid Google Drive file URL")

# --- Google Doc (via Selenium) ---
async def extract_google_doc_text(doc_url: str) -> str:
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(doc_url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2)  # Let JS render
        text = driver.find_element(By.TAG_NAME, "body").text
        return text
    finally:
        driver.quit()

@router.post("/web")

async def scrape_content(request: URLRequest):
    url = request.url
    domain = urlparse(url).netloc.lower()

    try:
        if "youtube.com" in domain or "youtu.be" in domain:
            text = extract_youtube_text(url)
        elif "drive.google.com" in domain:
            text = await extract_google_drive_text(url)
        # elif "wistia" in domain:
        #     text = await extract_wistia_text(url)
        elif "vimeo.com" in domain:
            text = await extract_vimeo_text(url)
        else:
            text = await extract_web_text(url)

        return {"url": url, "extracted_text": text}  # Limit for response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    




