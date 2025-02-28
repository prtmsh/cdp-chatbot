import os
import requests
import re
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import threading
import queue
from tqdm import tqdm
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Base URLs for each CDP
CDP_URLS = {
    "segment": "https://segment.com/docs/",
    "mparticle": "https://docs.mparticle.com/",
    "lytics": "https://docs.lytics.com/",
    "zeotap": "https://docs.zeotap.com/home/en-us/"
}

# Create directories for storing data
def create_directories():
    # Main data directory
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # CDP-specific directories
    for cdp in CDP_URLS.keys():
        if not os.path.exists(f'data/{cdp}'):
            os.makedirs(f'data/{cdp}')
    
    # Processed data directory
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')

# Clean HTML content
def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    
    # Get text
    text = soup.get_text()
    
    # Remove extra line breaks and whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text

# Check if URL is valid
def is_valid_url(url, base_url):
    parsed_url = urlparse(url)
    
    # Check if URL is absolute and from the same domain
    if parsed_url.netloc:
        return parsed_url.netloc in base_url
    
    # For relative URLs
    return True

# Crawl a website and extract content
def crawl_website(base_url, cdp_name, max_pages=100):
    print(f"Starting to crawl {cdp_name} documentation...")
    
    visited_urls = set()
    urls_to_visit = queue.Queue()
    urls_to_visit.put(base_url)
    
    # For tracking progress
    pages_crawled = 0
    
    with tqdm(total=max_pages, desc=f"Crawling {cdp_name}") as pbar:
        while not urls_to_visit.empty() and pages_crawled < max_pages:
            # Get the next URL
            current_url = urls_to_visit.get()
            
            # Skip if already visited
            if current_url in visited_urls:
                continue
            
            # Mark as visited
            visited_urls.add(current_url)
            
            try:
                # Get page content
                response = requests.get(current_url, timeout=10)
                
                # Skip if not a successful response
                if response.status_code != 200:
                    continue
                
                # Parse HTML content
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract the content
                clean_text = clean_html(str(soup))
                
                # Create a URL-safe filename
                safe_filename = re.sub(r'[^\w\-_]', '_', current_url.replace(base_url, ''))
                if not safe_filename or safe_filename == '_':
                    safe_filename = 'index'
                
                # Save the content
                with open(f'data/{cdp_name}/{safe_filename}.txt', 'w', encoding='utf-8') as file:
                    file.write(clean_text)
                
                # Update metadata
                with open(f'data/{cdp_name}/{safe_filename}.meta.json', 'w', encoding='utf-8') as file:
                    json.dump({
                        'url': current_url,
                        'title': soup.title.string if soup.title else current_url,
                        'cdp': cdp_name
                    }, file)
                
                # Find and queue all links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    
                    # Handle relative URLs
                    full_url = urljoin(current_url, href)
                    
                    # Check if URL is valid
                    if is_valid_url(full_url, base_url) and full_url not in visited_urls:
                        urls_to_visit.put(full_url)
                
                # Update progress
                pages_crawled += 1
                pbar.update(1)
                
                # Be nice to the server
                time.sleep(0.5)
            
            except Exception as e:
                print(f"Error processing {current_url}: {str(e)}")
    
    print(f"Finished crawling {cdp_name}. Crawled {pages_crawled} pages.")

# Process all CDPs in parallel
def process_all_cdps():
    create_directories()
    
    # Use ThreadPoolExecutor to run crawling in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        for cdp_name, url in CDP_URLS.items():
            executor.submit(crawl_website, url, cdp_name)

# Create processed data for indexing
def process_data_for_indexing():
    print("Processing data for indexing...")
    
    all_documents = []
    
    # Process each CDP
    for cdp in CDP_URLS.keys():
        cdp_dir = f'data/{cdp}'
        
        # Skip if directory doesn't exist
        if not os.path.exists(cdp_dir):
            continue
        
        # Process each file
        for filename in os.listdir(cdp_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(cdp_dir, filename)
                meta_path = os.path.join(cdp_dir, filename.replace('.txt', '.meta.json'))
                
                # Read content
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Read metadata if available
                metadata = {}
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as file:
                        metadata = json.load(file)
                
                # Split content into chunks (simple approach)
                chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
                
                # Create document entries
                for i, chunk in enumerate(chunks):
                    all_documents.append({
                        'cdp': cdp,
                        'title': metadata.get('title', filename),
                        'url': metadata.get('url', ''),
                        'content': chunk,
                        'chunk_id': i,
                        'source_file': filename
                    })
    
    # Save as CSV for easy loading
    df = pd.DataFrame(all_documents)
    df.to_csv('data/processed/all_documents.csv', index=False)
    
    print(f"Processed {len(all_documents)} document chunks.")

if __name__ == "__main__":
    process_all_cdps()
    process_data_for_indexing()