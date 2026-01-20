"""
Manual development test script.

Purpose:
- Ad-hoc testing of server and client behavior during development

Usage:
    uv run python dev_test_manual.py

Notes:
- Prints outputs for inspection
- No assertions or test framework used
"""

import os
import json
import logging
from typing import List, Dict, Optional
from firecrawl import FirecrawlApp
from urllib.parse import urlparse
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

SCRAPE_DIR = Path("scraped_content_dev")

def firecrawl(): 
    print("Starting firecrawl...")

    # Load API key 
    api_key = None
    if api_key is None:
        api_key = os.getenv('FIRECRAWL_API_KEY')
    if not api_key:
        raise ValueError("API key must be provided or set as FIRECRAWL_API_KEY environment variable")

    # Init app, create scraping directory 
    firecrawl_app = FirecrawlApp(api_key=api_key)
    SCRAPE_DIR.mkdir(parents=True, exist_ok=True) 

    # Load json (if exists)
    filename = "scrapped_data.json"
    json_path = SCRAPE_DIR / filename
    scrape_results = {}
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            scrape_results = json.load(f)
        print(f"Loaded JSON file: {json_path}")
    else:
        print(f"[INFO] JSON file does not exist: {json_path}")

    # Mapping of provider names to websites 
    websites = {
        'cloudrift_ai': 'https://www.cloudrift.ai/inference', 
        'deepinfra': 'https://deepinfra.com/pricing', 
    }

    # Scrape a website 
    for provider in websites: 
        website = websites[provider]

        # Scrape provider if not in JSON file 
        if provider not in scrape_results: 
            print(f"{provider} not in JSON, proceeding to scrape.")
            print(f"Firecrawl is scraping: {website}")
            doc = firecrawl_app.scrape(website, formats=["markdown", "html"])
            doc_dict = doc.model_dump() 
            doc_dict["provider_name"] = provider 
            scrape_results[provider] = {provider: doc_dict} 
        else: 
            print(f"{provider} is already present, skipping scrapping process.")  

    # Save to JSON 
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scrape_results, f)
        print(f"Scrapped JSON saved to: {json_path}")
    

def main(): 
    print("Manual dev testing...")
    firecrawl()

if __name__ == "__main__":
    main()