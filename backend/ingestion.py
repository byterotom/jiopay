# backend/enhanced_ingestion.py
# Enhanced scraper for JavaScript-heavy sites like JioPay

import requests
from bs4 import BeautifulSoup
import trafilatura
import json
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from requests_html import HTMLSession
from urllib.parse import urljoin, urlparse

ROOT = "data/jiopay"
os.makedirs(ROOT, exist_ok=True)

# --- MORE SEED URLS ADDED ---
# We've expanded this list to include more key pages.
SEED_URLS = [
    # Core Pages
    "https://jiopay.com/",
    "https://jiopay.com/business",
    "https://jiopay.com/consumer",
    
    # Feature/Product Pages
    "https://jiopay.com/payment-gateway",
    "https://jiopay.com/payment-links",
    "https://jiopay.com/upi",
    
    # Informational & Legal Pages
    "https://jiopay.com/help",
    "https://jiopay.com/about-us",
    "https://jiopay.com/contact-us",
    "https://jiopay.com/terms-and-conditions",
    "https://jiopay.com/privacy-policy"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# ---------------------------
# Selenium WebDriver pipeline (Best for JS-heavy sites)
# ---------------------------
def fetch_selenium(url, headless=True, wait_time=10):
    """
    Use Selenium WebDriver to render JavaScript content
    """
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument(f"--user-agent={HEADERS['User-Agent']}")
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(30)
        
        print(f"Loading {url} with Selenium...")
        driver.get(url)
        
        # Wait for content to load
        time.sleep(wait_time)
        
        # Try to wait for specific elements that indicate content has loaded
        try:
            WebDriverWait(driver, 15).until(
                lambda d: len(d.find_elements(By.TAG_NAME, "p")) > 0 or 
                         len(d.find_elements(By.TAG_NAME, "div")) > 5
            )
        except:
            print(f"Timeout waiting for content to load on {url}")
        
        # Get page source after JS execution
        html_content = driver.page_source
        title = driver.title
        
        # Extract text content
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "noscript"]):
            script.decompose()
        
        # Extract meaningful text
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div', 'span'])
        text_content = []
        
        for elem in text_elements:
            text = elem.get_text(strip=True)
            if text and len(text) > 10:  # Only include substantial text
                text_content.append(text)
        
        final_text = "\n\n".join(text_content)
        
        driver.quit()
        
        return {
            "url": url,
            "title": title,
            "html": html_content, # Also return html to discover more links
            "text": final_text,
            "method": "selenium"
        }
        
    except Exception as e:
        if 'driver' in locals():
            driver.quit()
        raise e

# ---------------------------
# Enhanced Requests-HTML pipeline
# ---------------------------
def fetch_rhtml_enhanced(url, wait_time=10, scroll=True):
    """
    Enhanced requests-html with better JS rendering
    """
    session = HTMLSession()
    
    try:
        print(f"Loading {url} with requests-html...")
        r = session.get(url, headers=HEADERS, timeout=30)
        
        # Render JavaScript with longer timeout and scrolling
        r.html.render(
            timeout=30,
            wait=wait_time,
            scrolldown=3 if scroll else 0,
            sleep=2
        )
        
        title = r.html.find("title", first=True)
        title_text = title.text if title else url
        
        # Get all text content
        text_content = r.html.text
        
        return {
            "url": url,
            "title": title_text,
            "html": r.html.html, # Also return html
            "text": text_content,
            "method": "requests-html-enhanced"
        }
        
    except Exception as e:
        raise e
    finally:
        session.close()

# ---------------------------
# Playwright alternative (if available)
# ---------------------------
def fetch_playwright(url, wait_time=10):
    """
    Use Playwright for better JS rendering (requires: pip install playwright)
    """
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=HEADERS['User-Agent']
            )
            page = context.new_page()
            
            print(f"Loading {url} with Playwright...")
            page.goto(url, timeout=30000)
            
            # Wait for network to be idle
            page.wait_for_load_state('networkidle', timeout=15000)
            
            # Additional wait
            page.wait_for_timeout(wait_time * 1000)
            
            title = page.title()
            content = page.content()
            
            # Extract text using BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            for script in soup(["script", "style", "noscript"]):
                script.decompose()
            
            text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div', 'span'])
            text_content = []
            
            for elem in text_elements:
                text = elem.get_text(strip=True)
                if text and len(text) > 10:
                    text_content.append(text)
            
            final_text = "\n\n".join(text_content)
            
            browser.close()
            
            return {
                "url": url,
                "title": title,
                "html": content, # Also return html
                "text": final_text,
                "method": "playwright"
            }
            
    except ImportError:
        print("Playwright not installed. Run: pip install playwright && playwright install")
        return None
    except Exception as e:
        raise e

# --- Functions fetch_bs4 and fetch_trafilatura remain the same ---

# ---------------------------
# BeautifulSoup pipeline (original)
# ---------------------------
def fetch_bs4(url):
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    title = soup.title.string if soup.title else url
    ps = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
    text = "\n\n".join([p.get_text(strip=True) for p in ps if p.get_text(strip=True)])
    return {"url": url, "title": title, "html": r.text, "text": text, "method": "bs4"}

# ---------------------------
# Trafilatura pipeline (original)
# ---------------------------
def fetch_trafilatura(url):
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    result = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    return {"url": url, "title": None, "html": downloaded, "text": result, "method": "trafilatura"}


# ---------------------------
# Helpers
# ---------------------------
def save_page(obj, prefix=""):
    # Clean up the object before saving
    if "html" in obj:
        del obj["html"] # Don't save the full HTML to JSON
    safe = obj["url"].replace("https://", "").replace("http://", "").replace("/", "_")
    fname = f"{prefix}{safe}.json"
    with open(os.path.join(ROOT, fname), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def try_method(method_func, url, method_name):
    """
    Try a scraping method and handle errors gracefully
    """
    try:
        result = method_func(url)
        if result and result.get("text") and len(result["text"].strip()) > 50:
            print(f"✅ {method_name} succeeded for {url} ({len(result['text'])} chars)")
            return result
        else:
            print(f"❌ {method_name} returned insufficient content for {url}")
            return None
    except Exception as e:
        print(f"❌ {method_name} failed for {url}: {str(e)}")
        return None

# --- NEW: DYNAMIC LINK DISCOVERY ---
def discover_links(base_url, html_content):
    """
    Finds all links on a page that belong to the same domain.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    links = set()
    domain_name = urlparse(base_url).netloc
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        # Create a full URL from a relative link (e.g., /about-us)
        full_url = urljoin(base_url, href)
        # Check if the link belongs to the same domain
        if domain_name in urlparse(full_url).netloc:
            # Clean fragment identifiers (#) from URL
            links.add(full_url.split('#')[0])
            
    return links

if __name__ == "__main__":
    pages = []
    
    urls_to_visit = list(SEED_URLS)
    visited_urls = set()
    MAX_PAGES = 20 # Add a limit to prevent infinite crawling

    while urls_to_visit and len(visited_urls) < MAX_PAGES:
        url = urls_to_visit.pop(0)
        if url in visited_urls:
            continue
        
        visited_urls.add(url)
        print(f"\n🔍 Processing ({len(visited_urls)}/{MAX_PAGES}): {url}")
        successful_result = None
        
        methods = [
            (fetch_selenium, "Selenium"),
            (fetch_playwright, "Playwright"),
            (lambda u: fetch_rhtml_enhanced(u, wait_time=15), "Requests-HTML Enhanced"),
            (fetch_trafilatura, "Trafilatura"),
            (fetch_bs4, "BeautifulSoup4"),
        ]
        
        for method_func, method_name in methods:
            result = try_method(method_func, url, method_name)
            if result:
                successful_result = result
                save_page(result, prefix=f"{method_name.lower().replace(' ', '_')}_")
                pages.append(result)
                
                # Discover new links from the successfully fetched page
                if "html" in successful_result and successful_result["html"]:
                    new_links = discover_links(url, successful_result["html"])
                    for link in new_links:
                        if link not in visited_urls:
                            urls_to_visit.append(link)
                    print(f"Discovered {len(new_links)} new links.")

                break
        
        if not successful_result:
            print(f"❌ All methods failed for {url}")
    
    print(f"\n✅ Successfully fetched {len(pages)} pages out of {len(visited_urls)} attempted URLs.")
    
    summary = {
        "total_pages": len(pages),
        "pages": [{"url": p["url"], "method": p.get("method", "unknown"), "text_length": len(p.get("text", ""))} for p in pages]
    }
    
    with open(os.path.join(ROOT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)