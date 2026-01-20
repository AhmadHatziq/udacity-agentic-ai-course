
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
from datetime import datetime

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRAPE_DIR = Path("scraped_content")

mcp = FastMCP("llm_inference")

@mcp.tool()
def scrape_websites(
    websites: Dict[str, str],
    formats: List[str] = ['markdown'],
    api_key: Optional[str] = None
) -> List[str]:
    """
    Scrape multiple websites using Firecrawl and stores their content.
    Content can be retrieved via tool 'extract_scraped_info'. 
    
    Args:
        websites: Dictionary of provider_name -> URL mappings eg {"cloudrift_ai": "https://www.cloudrift.ai/inference", ...}
        formats: List of formats to scrape ['markdown', 'html'] (default: both)
        api_key: Firecrawl API key (if None, expects environment variable)
        
    Returns:
        List of provider names for successfully scraped websites. 
        If no names returned, means all websites have been scrapped before. 

    Example: 
        >> scrape_websites(websites = {"cloudrift_ai": "https://www.cloudrift.ai/inference"})
        >> ["clouddrift_ai", ...] # Where these are the provides that have successfully been scraped. 

        >> scrape_websites(websites = {"cloudrift_ai": "https://www.cloudrift.ai/inference"})
        >> [] # As all websites have already been scraped, please use tool 'extract_scraped_info' to get the data 
    """
    # Init api key for Firecrawl API 
    if api_key is None:
        api_key = os.getenv('FIRECRAWL_API_KEY')
        if not api_key:
            raise ValueError("API key must be provided or set as FIRECRAWL_API_KEY environment variable")
    
    # Init app, create scraping directory 
    firecrawl_app = FirecrawlApp(api_key=api_key)
    SCRAPE_DIR.mkdir(parents=True, exist_ok=True) 
    
    # Load json (if exists)
    filename = "scraped_metadata.json"
    json_path = SCRAPE_DIR / filename
    scraped_metadata = {}
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            scraped_metadata = json.load(f)
        logging.info(f"Loaded JSON file: {json_path}")
    else:
        logging.info(f"JSON file does not exist: {json_path}")
    
    # save the scraped content to files and then create scraped_metadata.json as a summary file
    # check if the provider has already been scraped and decide if you want to overwrite
    # {
    #     "cloudrift_ai": {
    #         "provider_name": "cloudrift_ai",
    #         "url": "https://www.cloudrift.ai/inference",
    #         "domain": "www.cloudrift.ai",
    #         "scraped_at": "2025-10-23T00:44:59.902569",
    #         "formats": [
    #             "markdown",
    #             "html"
    #         ],
    #         "success": "true",
    #         "content_files": {
    #             "markdown": "cloudrift_ai_markdown.txt",
    #             "html": "cloudrift_ai_html.txt"
    #         },
    #         "title": "AI Inference",
    #         "description": "Scraped content goes here"
    #     }
    # }

    successful_scrapes: List[str] = []

    # Scrape the websites 
    for provider in websites: 
        website = websites[provider]

        # Scrape provider if not in JSON file 
        if provider not in scraped_metadata: 
            try: 
                logger.info(f"{provider} not in JSON, proceeding to scrape.")
                logger.info(f"Firecrawl is scraping: {website}")
 
                scrape_result = firecrawl_app.scrape(website, formats=formats).model_dump()

                # Save each requested file format 
                content_files = {}
                for format_type in formats: 
                    content = scrape_result[format_type]
                    file_name = f"{provider}_{format_type}.txt"
                    file_path = SCRAPE_DIR / file_name
                    file_path.write_text(content, encoding="utf-8")
                    content_files[format_type] = file_path

                # Save metadata 
                provider_dict = {
                    "provider_name": provider, 
                    "url": website,
                    "domain": urlparse(website).netloc,
                    "cached_at": scrape_result["metadata"]["cached_at"], 
                    "scrape_time": datetime.utcnow().isoformat(),
                    "title": scrape_result["metadata"]["title"], 
                    "description": scrape_result["metadata"]["description"], 
                    "success": True, 
                    "formats": formats, 
                    "content_files": content_files
                }
                scraped_metadata[provider] = provider_dict

                successful_scrapes.append(provider)
            except Exception as e:
                logger.exception(f"Failed to scrape {provider} ({url}): {e}")
        else: 
            logger.info(f"{provider} is already present, skipping scrapping process.")  
            continue 

    # Save to JSON 
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scraped_metadata, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"JSON saved to: {json_path}")

    # metadata_file = os.path.join(path, "scraped_metadata.json")
    # continue your solution here ...
    # return [f"Successfully scraped data for {str(websites)}", successful_scrapes]
    logger.info(f"Successfully scrapped {len(successful_scrapes)} out of {len(websites)} websites. \nScraped websites are: {successful_scrapes}")
    return successful_scrapes

@mcp.tool()
def extract_scraped_info(identifier: str) -> str:
    """
    Extract information about a scraped website. Use this for comparing costs/prices.
    
    Args:
        identifier: The provider name, full URL, or domain to look for
        
    Returns:
        Formatted JSON string with the scraped information

    Example: 
        >> extract_scraped_info(identifier="cloudrift_ai")
        >> {
        result:
        "{
        "content": {
            "markdown": "Smart, Fast, and Affordable\n\n# Unmatched Price Performance\n\nFast responses, scalable performance"
            }... }"
    """
    
    # Log down files, args 
    try: 
        logger.info(f"Extracting information for identifier: {identifier}")
        logger.info(f"Files in {SCRAPE_DIR}: {os.listdir(SCRAPE_DIR)}")
        metadata_file = os.path.join(SCRAPE_DIR, "scraped_metadata.json")
        logger.info(f"Checking metadata file: {metadata_file}")
    except Exception as e: 
        return f"Error. There's no saved information related to identifier '{identifier}'."

    # contine your response here ...

    # Load json (if exists)
    filename = "scraped_metadata.json"
    json_path = SCRAPE_DIR / filename
    scraped_metadata = {}
    if json_path.exists():
        try: 
            with json_path.open("r", encoding="utf-8") as f:
                scraped_metadata = json.load(f)
            logging.info(f"Loaded JSON file: {json_path}")
        except Exception as e: 
            logging.error(f"Error loading JSON file: {e}")
            return f"There's no saved information related to identifier '{identifier}'."
    else:
        logging.info(f"JSON file does not exist: {json_path}")
        return f"There's no saved information related to identifier '{identifier}'."

    # Process JSON file and look for a match 
    found_match = False 
    found_result = None 
    for provider_name, metadata in scraped_metadata.items(): 
        url = metadata.get('url', '')
        domain = metadata.get('domain', '')

        if identifier in [provider_name, url, domain]: 
            found_match = True 
            found_result = metadata.copy() 
            logger.info(f"Found match in {metadata}")
            break
    
    if not found_match: 
        return f"There's no saved information related to identifier '{identifier}'."

    # Process matching results 
    """
    Sanple object: "content_files": {
      "markdown": "scraped_content/cloudrift_ai_markdown.txt",
      "html": "scraped_content/cloudrift_ai_html.txt"
    }
    """
    results = {"content": {}}
    content_files = found_result.get('content_files', {})
    if content_files: 
        for content_type, file_path in content_files.items(): 
            file_path_obj = Path(file_path)
            content = file_path_obj.read_text(encoding="utf-8")
            results['content'][content_type] = content
    
    logging.info(f"Returning results found: {found_result}")
    return json.dumps(results, indent=2)

if __name__ == "__main__":
    mcp.run(transport="stdio")