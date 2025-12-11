#!/usr/bin/env python3
"""Download datasets with fresh URLs"""
import requests
import json
from pathlib import Path
import time

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

datasets = [
    {
        "api": "https://www.data.gov.in/backend/dms/v1/resource/unemployment-rate-lucknow-jan-2019?_format=json",
        "path": DATA_DIR / "data-gov-in" / "workforce" / "unemployment-rate-lucknow-jan-2019.csv",
        "name": "Unemployment Rate Lucknow"
    },
    {
        "api": "https://www.data.gov.in/backend/dms/v1/resource/demographic-profile-lucknow-2011?_format=json",
        "path": DATA_DIR / "data-gov-in" / "urbanization" / "demographic-profile-lucknow-2011.csv",
        "name": "Demographic Profile Lucknow"
    },
    {
        "api": "https://www.data.gov.in/backend/dms/v1/resource/household-profile-lucknow-census-2011-0?_format=json",
        "path": DATA_DIR / "data-gov-in" / "urbanization" / "household-profile-lucknow-2011.csv",
        "name": "Household Profile Lucknow"
    },
    {
        "api": "https://www.data.gov.in/backend/dms/v1/resource/slum-housing-and-population-data-lucknow-jan-2019?_format=json",
        "path": DATA_DIR / "data-gov-in" / "slum-housing" / "slum-housing-population-lucknow-jan-2019.csv",
        "name": "Slum Housing Lucknow"
    }
]

for dataset in datasets:
    print(f"\nDownloading {dataset['name']}...")
    try:
        # Get fresh URL
        resp = requests.get(dataset['api'], timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if 'field_datafile' in data and len(data['field_datafile']) > 0:
            url = data['field_datafile'][0]['url']
            print(f"  Found URL: {url[:80]}...")
            
            # Download immediately
            dataset['path'].parent.mkdir(parents=True, exist_ok=True)
            file_resp = requests.get(url, timeout=30, stream=True, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            })
            file_resp.raise_for_status()
            
            with open(dataset['path'], 'wb') as f:
                for chunk in file_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            size = dataset['path'].stat().st_size
            print(f"  ✓ Downloaded {size} bytes to {dataset['path']}")
            
            # Verify it's CSV
            with open(dataset['path'], 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
                if 'html' in first_line.lower() or '<!doctype' in first_line.lower():
                    print(f"  ⚠ WARNING: File appears to be HTML!")
                else:
                    print(f"  ✓ Verified as CSV")
        else:
            print(f"  ✗ No download URL found")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    time.sleep(1)

print("\nDone!")

