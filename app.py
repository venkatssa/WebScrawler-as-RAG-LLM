import streamlit as st
import requests
import validators
import json
import xml.etree.ElementTree as ET
import psycopg2
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from rag_llm import fetch_and_store_scraped_data, retrieve_and_rerank, query_llm

# UI Configuration
st.set_page_config(page_title="Website Crawler", layout="centered")
st.markdown("""
    <h1 style='text-align: center;'>🌐 Website Crawler</h1>
    <hr>
""", unsafe_allow_html=True)

# PostgreSQL connection setup
DB_CONFIG = {
    "dbname": "postgres",
    "user": "venkat_user",
    "password": "12345",
    "host": "localhost",
    "port": "5432"
}

def get_table_name(url):
    """Generate a table name based on the website domain."""
    domain = urlparse(url).netloc.replace(".", "_").replace("-", "_")
    return f"scraped_{domain}"

def save_to_postgres(scraped_data, table_name):
    """Save all scraped data to PostgreSQL after processing."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                url TEXT UNIQUE,
                title TEXT,
                description TEXT,
                content TEXT,
                scrape_time TIMESTAMP DEFAULT NOW()
            )
        """)

        for data in scraped_data:
            cur.execute(f"""
                INSERT INTO {table_name} (url, title, description, content, scrape_time)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (url) DO NOTHING
            """, (data["url"], data["title"], data["description"], data["content"]))

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error("Error saving data to DB.")

def fetch_sitemap_from_robots(url):
    """Fetch sitemap URLs from robots.txt, ignoring .xml.gz files."""
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    
    try:
        response = requests.get(robots_url, timeout=10)
        response.raise_for_status()
        lines = response.text.split("\n")

        # Extract only valid sitemap URLs, ignoring .xml.gz files
        sitemap_urls = [line.split(": ")[1].strip() for line in lines 
                        if line.lower().startswith("sitemap:") and not line.strip().lower().endswith(".xml.gz")]

        if sitemap_urls:
            st.success(f"✅ Found {len(sitemap_urls)} valid sitemaps in robots.txt")
        else:
            st.warning("⚠️ No valid sitemaps found in robots.txt. Using default `/sitemap.xml` path.")
            sitemap_urls = [f"{parsed_url.scheme}://{parsed_url.netloc}/sitemap.xml"]

        return sitemap_urls

    except requests.exceptions.RequestException:
        st.error("Could not fetch robots.txt. Trying default `/sitemap.xml`.")
        return [f"{parsed_url.scheme}://{parsed_url.netloc}/sitemap.xml"]

def fetch_sitemap_links(sitemap_urls):
    """Fetch and parse the first 25 valid links from each sitemap file."""
    all_links = []

    for sitemap_url in sitemap_urls:
        try:
            response = requests.get(sitemap_url, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)

            # Extract <loc> links, ignoring .xml.gz files, and limit to 25 per sitemap
            links = [elem.text for elem in root.iter('{http://www.sitemaps.org/schemas/sitemap/0.9}loc') if not elem.text.endswith(".xml.gz")][:25]
            all_links.extend(links)
        except requests.exceptions.RequestException:
            st.warning(f"⚠️ Could not fetch sitemap: {sitemap_url}")

    if all_links:
        st.success(f"✅ Total {len(all_links)} pages found in sitemaps")
  
    return all_links

def scrape_page(url):
    """Scrape title, meta description, and main content from a webpage."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract Title
        title = soup.title.string.strip() if soup.title else "No Title"

        # Extract Meta Description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        description = meta_desc["content"].strip() if meta_desc and "content" in meta_desc.attrs else "No Description"

        # Extract Main Content (First 5 Paragraphs)
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text(strip=True) for p in paragraphs[:5]])

        return {
            "url": url,
            "title": title,
            "description": description,
            "content": content
        }

    except Exception as e:
        return {"url": url, "error": str(e)}

def scrape_pages(links):
    """Scrape multiple pages with a progress bar."""
    scraped_data = []
    progress_bar = st.progress(0)
    for idx, url in enumerate(links):
        scraped_page = scrape_page(url)
        # print("scraped_page:",scraped_page)
        if scraped_page:
            scraped_data.append(scraped_page)
        progress_bar.progress((idx + 1) / len(links))

    progress_bar.empty()
    return scraped_data

# UI for user input
website_url = st.text_input("Enter Website URL:", key="website_url")
submit_button = st.button("Enter")

if "scraped" not in st.session_state:
    st.session_state.scraped = False
if "index" not in st.session_state:
    st.session_state.index = None
if "metadata" not in st.session_state:
    st.session_state.metadata = []
if "feedback" not in st.session_state:
    st.session_state.feedback = None

if website_url and submit_button:
    if not validators.url(website_url):
        st.error("Invalid URL. Please enter a valid website URL.")
    else:
        st.info("Fetching robots.txt for sitemap information...")
        sitemap_urls = fetch_sitemap_from_robots(website_url)
        
        st.info("Fetching links from sitemap(s)...")
        links = fetch_sitemap_links(sitemap_urls)
        
        if links:
            scraped_data = scrape_pages(links)
            # Debugging: Print Scraped Data
            if scraped_data:
                st.write("🔍 Sample Scraped Data:", scraped_data[:2])  # Show first 2 results for verification
            else:
                st.error("No data scraped from the website.")

            index, metadata = fetch_and_store_scraped_data(scraped_data)
            if index:
                st.session_state.index = index
                st.session_state.metadata = metadata
                st.success("✅ FAISS database created successfully!")
            else:
                st.error("Failed to process data.")
            
            table_name = get_table_name(website_url)
            save_to_postgres(scraped_data, table_name)
            
            st.success("✅ Data scraping completed and saved to database!")
            st.session_state.scraped = True

# Question input with dummy response after scraping
if st.session_state.scraped and st.session_state.index is not None:
    question = st.text_input("Ask a question:", key="question_input")
    if question:
        reranked_results = retrieve_and_rerank(question, st.session_state.index, st.session_state.metadata, k=5, top_n=3, min_similarity=0.1)
        # Debugging: Print Retrieved Documents
        # if reranked_results:
        #     st.write("📄 Retrieved Document:", reranked_results[0]['title'])
        # else:
        #     st.warning("No matching document found in FAISS.")

        if reranked_results:
            llm_response = query_llm(question, reranked_results[:1])  # Only pass the most relevant document
            # source_url = reranked_results[0]['url']  # Extract only the most relevant source URL

            st.write("🤖 LLM Response:", llm_response)
            # st.write("🔗 Source:", source_url)

            col1, col2 = st.columns([0.1, 0.1])
            with col1:
                if st.button("👍", key="thumbs_up"):
                    st.session_state.feedback = "positive"
            with col2:
                if st.button("👎", key="thumbs_down"):
                    st.session_state.feedback = "negative"
            
            if st.session_state.feedback == "positive":
                st.success("Thanks for your feedback!")
            elif st.session_state.feedback == "negative":
                st.warning("Sorry about that! We'll improve the results.")
        else:
            st.write("No relevant results found.")
