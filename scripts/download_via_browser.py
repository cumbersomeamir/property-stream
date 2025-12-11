#!/usr/bin/env python3
"""Download by extracting fresh URLs from page source"""
import requests
import re
import json
from pathlib import Path
from urllib.parse import unquote

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# URLs to download
pages = [
    {
        "url": "https://www.data.gov.in/resource/unemployment-rate-lucknow-jan-2019",
        "path": DATA_DIR / "data-gov-in" / "workforce" / "unemployment-rate-lucknow-jan-2019.csv",
        "name": "Unemployment Rate"
    },
    {
        "url": "https://www.data.gov.in/resource/demographic-profile-lucknow-2011",
        "path": DATA_DIR / "data-gov-in" / "urbanization" / "demographic-profile-lucknow-2011.csv",
        "name": "Demographic Profile"
    },
    {
        "url": "https://www.data.gov.in/resource/household-profile-lucknow-census-2011-0",
        "path": DATA_DIR / "data-gov-in" / "urbanization" / "household-profile-lucknow-2011.csv",
        "name": "Household Profile"
    },
    {
        "url": "https://www.data.gov.in/resource/slum-housing-and-population-data-lucknow-jan-2019",
        "path": DATA_DIR / "data-gov-in" / "slum-housing" / "slum-housing-population-lucknow-jan-2019.csv",
        "name": "Slum Housing"
    }
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://www.data.gov.in/'
}

for page in pages:
    print(f"\nDownloading {page['name']}...")
    try:
        # Get the page
        resp = requests.get(page['url'], headers=headers, timeout=30)
        resp.raise_for_status()
        html = resp.text
        
        # Extract URL from window.__NUXT__ JavaScript
        # Look for field_datafile in the JavaScript
        pattern = r'field_datafile["\']:\s*["\']([^"\']+\.csv[^"\']*)["\']'
        matches = re.findall(pattern, html)
        
        if not matches:
            # Try alternative pattern
            pattern = r'["\']field_datafile["\']:\s*["\']([^"\']+\.csv[^"\']*)["\']'
            matches = re.findall(pattern, html)
        
        if matches:
            csv_url = matches[0].replace('\\/', '/').replace('\\u002F', '/')
            csv_url = unquote(csv_url)
            print(f"  Found URL: {csv_url[:100]}...")
            
            # Download immediately with same session
            file_resp = requests.get(csv_url, headers=headers, timeout=30, cookies=resp.cookies, stream=True)
            
            if file_resp.status_code == 200:
                page['path'].parent.mkdir(parents=True, exist_ok=True)
                with open(page['path'], 'wb') as f:
                    for chunk in file_resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                size = page['path'].stat().st_size
                print(f"  ✓ Downloaded {size} bytes")
                
                # Verify
                with open(page['path'], 'r', encoding='utf-8', errors='ignore') as f:
                    first = f.readline()
                    if 'html' not in first.lower() and '<!doctype' not in first.lower():
                        print(f"  ✓ Verified as CSV")
                    else:
                        print(f"  ⚠ WARNING: File is HTML, not CSV")
            else:
                print(f"  ✗ Download failed: {file_resp.status_code}")
        else:
            print(f"  ✗ Could not find CSV URL in page")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\nDone!")

