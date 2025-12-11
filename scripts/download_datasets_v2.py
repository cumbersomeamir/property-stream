#!/usr/bin/env python3
"""
Script to download datasets for property investment analysis in Lucknow
Extracts actual download URLs from data.gov.in pages
"""
import os
import requests
import json
import re
from pathlib import Path
import time
import html
from urllib.parse import unquote

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

def extract_csv_url(html_content):
    """Extract CSV download URL from data.gov.in page HTML"""
    # Look for field_datafile in window.__NUXT__ JavaScript object
    # Pattern: field_datafile:"https://...csv?..."
    patterns = [
        r'field_datafile["\']:\s*["\']([^"\']+\.csv[^"\']*)["\']',
        r'field_datafile["\']:\s*["\']([^"\']+)["\']',
        r'["\']field_datafile["\']:\s*["\']([^"\']+\.csv[^"\']*)["\']',
        r'field_datafile[=:]\s*["\']([^"\']+\.csv[^"\']*)["\']',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, html_content, re.IGNORECASE)
        for match in matches:
            url = match.replace('\\/', '/').replace('\\u002F', '/').replace('\\/', '/')
            if 'csv' in url.lower() or 'datafile' in url.lower():
                if url.startswith('http'):
                    return url
    
    # Try to find any CSV URL with datafile in path
    csv_pattern = r'https?://[^"\'\s<>]+datafile[^"\'\s<>]+\.csv[^"\'\s<>]*'
    matches = re.findall(csv_pattern, html_content, re.IGNORECASE)
    if matches:
        return matches[0]
    
    # Last resort: find any CSV URL
    csv_pattern = r'https?://[^"\'\s<>]+\.csv[^"\'\s<>]*'
    matches = re.findall(csv_pattern, html_content)
    if matches:
        for url in matches:
            if 'data.gov.in' in url or 'datafile' in url:
                return url
    
    return None

def download_from_url(url, filepath, description=""):
    """Download a file from URL"""
    print(f"Downloading {description}...")
    try:
        # Decode HTML entities
        url = html.unescape(url)
        url = unquote(url)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/csv,application/csv,*/*',
            'Referer': 'https://data.gov.in/'
        }
        
        response = requests.get(url, headers=headers, timeout=60, allow_redirects=True, stream=True)
        response.raise_for_status()
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        file_size = filepath.stat().st_size
        print(f"✓ Downloaded: {filepath} ({file_size} bytes)")
        
        # Verify it's actually CSV
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            if 'html' in first_line.lower() or '<!doctype' in first_line.lower():
                print(f"⚠ Warning: File appears to be HTML, not CSV")
                return False
        return True
    except Exception as e:
        print(f"✗ Failed to download {description}: {e}")
        return False

def download_data_gov_in_resource(resource_url, filepath, description):
    """Download from data.gov.in by extracting CSV URL from page"""
    try:
        print(f"Fetching page: {resource_url}")
        response = requests.get(resource_url, timeout=30)
        response.raise_for_status()
        
        csv_url = extract_csv_url(response.text)
        if csv_url:
            return download_from_url(csv_url, filepath, description)
        else:
            print(f"✗ Could not find CSV URL in page")
            # Create README
            readme_path = filepath.with_suffix('.README.txt')
            with open(readme_path, 'w') as f:
                f.write(f"Dataset: {description}\n")
                f.write(f"Source URL: {resource_url}\n")
                f.write(f"Please download manually from the above URL\n")
            return False
    except Exception as e:
        print(f"✗ Error fetching page: {e}")
        return False

# Data.gov.in datasets
data_gov_datasets = [
    {
        "url": "https://data.gov.in/resource/unemployment-rate-lucknow-jan-2019",
        "path": DATA_DIR / "data-gov-in" / "workforce" / "unemployment-rate-lucknow-jan-2019.csv",
        "description": "Unemployment Rate Lucknow"
    },
    {
        "url": "https://data.gov.in/resource/demographic-profile-lucknow-2011",
        "path": DATA_DIR / "data-gov-in" / "urbanization" / "demographic-profile-lucknow-2011.csv",
        "description": "Demographic Profile Lucknow 2011"
    },
    {
        "url": "https://data.gov.in/resource/household-profile-lucknow-census-2011-0",
        "path": DATA_DIR / "data-gov-in" / "urbanization" / "household-profile-lucknow-2011.csv",
        "description": "Household Profile Lucknow 2011"
    },
    {
        "url": "https://data.gov.in/resource/slum-housing-and-population-data-lucknow-jan-2019",
        "path": DATA_DIR / "data-gov-in" / "slum-housing" / "slum-housing-population-lucknow-jan-2019.csv",
        "description": "Slum Housing and Population Lucknow"
    }
]

def main():
    print("Starting dataset downloads from data.gov.in...\n")
    
    for dataset in data_gov_datasets:
        download_data_gov_in_resource(
            dataset['url'],
            dataset['path'],
            dataset['description']
        )
        time.sleep(2)  # Be polite
    
    print("\n=== Download process completed! ===")
    print("\nNext steps:")
    print("1. Download Kaggle datasets manually or using Kaggle API")
    print("2. Download from dataportalforcities.org")
    print("3. Download from censusindia.gov.in")
    print("4. Access UP Bhulekh, India Data Portal, and NDAP data")

if __name__ == "__main__":
    main()
