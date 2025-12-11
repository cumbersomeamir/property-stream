#!/usr/bin/env python3
"""
Script to download datasets for property investment analysis in Lucknow
"""
import os
import requests
import json
from pathlib import Path
import time

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Create directories
DATA_DIR.mkdir(exist_ok=True)

def download_file(url, filepath, description=""):
    """Download a file from URL"""
    print(f"Downloading {description}...")
    try:
        response = requests.get(url, timeout=30, allow_redirects=True)
        response.raise_for_status()
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Downloaded: {filepath} ({len(response.content)} bytes)")
        return True
    except Exception as e:
        print(f"✗ Failed to download {description}: {e}")
        return False

def get_data_gov_in_resource(resource_id):
    """Get data from data.gov.in using CKAN API"""
    api_url = "https://data.gov.in/api/datastore/resource.json"
    params = {
        "resource_id": resource_id,
        "limit": 100000
    }
    
    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get('records', [])
    except Exception as e:
        print(f"Error fetching resource {resource_id}: {e}")
        return None

# Data.gov.in datasets
data_gov_datasets = {
    "city-profile-lucknow": {
        "url": "https://data.gov.in/catalog/city-profile-lucknow-0",
        "path": DATA_DIR / "data-gov-in" / "population" / "city-profile-lucknow.csv",
        "description": "City Profile Lucknow"
    },
    "unemployment-lucknow": {
        "url": "https://data.gov.in/resource/unemployment-rate-lucknow-jan-2019",
        "path": DATA_DIR / "data-gov-in" / "workforce" / "unemployment-rate-lucknow-jan-2019.csv",
        "description": "Unemployment Rate Lucknow"
    },
    "demographic-profile-lucknow": {
        "url": "https://data.gov.in/resource/demographic-profile-lucknow-2011",
        "path": DATA_DIR / "data-gov-in" / "urbanization" / "demographic-profile-lucknow-2011.csv",
        "description": "Demographic Profile Lucknow 2011"
    },
    "household-profile-lucknow": {
        "url": "https://data.gov.in/resource/household-profile-lucknow-census-2011-0",
        "path": DATA_DIR / "data-gov-in" / "urbanization" / "household-profile-lucknow-2011.csv",
        "description": "Household Profile Lucknow 2011"
    },
    "slum-housing-lucknow": {
        "url": "https://data.gov.in/resource/slum-housing-and-population-data-lucknow-jan-2019",
        "path": DATA_DIR / "data-gov-in" / "slum-housing" / "slum-housing-population-lucknow-jan-2019.csv",
        "description": "Slum Housing and Population Lucknow"
    }
}

def download_from_data_gov_in():
    """Download datasets from data.gov.in"""
    print("\n=== Downloading from data.gov.in ===")
    
    # Try to get resource IDs and download via API
    # For now, we'll create placeholder files with download instructions
    for key, dataset in data_gov_datasets.items():
        print(f"\nAttempting to download: {dataset['description']}")
        # Try direct CSV download first
        csv_url = dataset['url'].replace('/resource/', '/resource/').replace('/catalog/', '/catalog/')
        
        # Try common CSV download patterns
        possible_urls = [
            f"{dataset['url']}/download",
            f"{dataset['url']}.csv",
            f"{dataset['url']}?format=csv",
        ]
        
        downloaded = False
        for url in possible_urls:
            if download_file(url, dataset['path'], dataset['description']):
                downloaded = True
                break
        
        if not downloaded:
            # Create a README with the URL
            readme_path = dataset['path'].with_suffix('.README.txt')
            with open(readme_path, 'w') as f:
                f.write(f"Dataset: {dataset['description']}\n")
                f.write(f"Source URL: {dataset['url']}\n")
                f.write(f"Please download manually from the above URL\n")
            print(f"Created README at: {readme_path}")

if __name__ == "__main__":
    print("Starting dataset downloads...")
    download_from_data_gov_in()
    print("\nDownload process completed!")
