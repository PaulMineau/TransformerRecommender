"""
Data downloader for Amazon Beauty dataset
"""
import os
import gzip
import json
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any

class AmazonDataDownloader:
    """Download and process Amazon Beauty dataset"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        # Try multiple potential URLs for Amazon dataset
        self.urls = {
            "reviews": [
                "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Beauty_5.json.gz",
                "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Beauty_5.json.gz",
                "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz"
            ],
            "metadata": [
                "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Beauty.json.gz",
                "https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Beauty.json.gz",
                "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz"
            ]
        }
        
        os.makedirs(data_dir, exist_ok=True)
    
    def download_file(self, urls: List[str], filename: str) -> str:
        """Download a file from URL with progress bar, trying multiple URLs"""
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            print(f"File {filename} already exists, skipping download")
            return filepath
        
        print(f"Downloading {filename}...")
        
        # Try each URL until one works
        for i, url in enumerate(urls):
            try:
                print(f"  Trying URL {i+1}/{len(urls)}: {url}")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filepath, 'wb') as f, tqdm(
                    desc=filename,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                
                print(f"  ✅ Successfully downloaded {filename}")
                return filepath
                
            except Exception as e:
                print(f"  ❌ Failed to download from {url}: {e}")
                continue
        
        # If all URLs failed, raise an exception
        raise Exception(f"Failed to download {filename} from any of the provided URLs")
    
    def parse_json_gz(self, filepath: str) -> List[Dict[str, Any]]:
        """Parse gzipped JSON file"""
        data = []
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Parsing {os.path.basename(filepath)}"):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return data
    
    def generate_synthetic_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic beauty product data for demonstration"""
        print("Generating synthetic beauty product data...")
        
        import random
        import time
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Beauty product categories and brands
        categories = [
            "Makeup", "Skincare", "Haircare", "Fragrance", "Tools & Accessories",
            "Nail Care", "Bath & Body", "Men's Grooming", "Sun Care", "Eye Care"
        ]
        
        brands = [
            "L'Oreal", "Maybelline", "Revlon", "CoverGirl", "Neutrogena", "Olay",
            "Clinique", "MAC", "Urban Decay", "Too Faced", "Sephora", "NYX",
            "Essence", "e.l.f.", "Milani", "Wet n Wild", "Almay", "Rimmel"
        ]
        
        product_types = {
            "Makeup": ["Lipstick", "Foundation", "Mascara", "Eyeshadow", "Blush", "Concealer"],
            "Skincare": ["Moisturizer", "Cleanser", "Serum", "Toner", "Sunscreen", "Face Mask"],
            "Haircare": ["Shampoo", "Conditioner", "Hair Oil", "Hair Spray", "Hair Mask", "Dry Shampoo"],
            "Fragrance": ["Perfume", "Body Spray", "Cologne", "Body Mist"],
            "Tools & Accessories": ["Makeup Brush", "Beauty Sponge", "Mirror", "Tweezers", "Eyelash Curler"]
        }
        
        # Generate products
        n_products = 2000
        products = []
        
        for i in range(n_products):
            category = random.choice(categories)
            brand = random.choice(brands)
            product_type = random.choice(product_types.get(category, ["Beauty Product"]))
            
            product = {
                'asin': f'B{i:06d}',
                'title': f'{brand} {product_type} - {random.choice(["Natural", "Long-lasting", "Waterproof", "Organic", "Professional", "Deluxe"])}',
                'brand': brand,
                'categories': [category],
                'price': f'${random.uniform(5.99, 89.99):.2f}',
                'imUrl': f'https://example.com/product_{i}.jpg'
            }
            products.append(product)
        
        # Generate users and reviews
        n_users = 5000
        n_reviews = 50000
        
        reviews = []
        base_time = int(time.time()) - (365 * 24 * 3600)  # Start from 1 year ago
        
        for i in range(n_reviews):
            user_id = f'U{random.randint(1, n_users):06d}'
            product = random.choice(products)
            
            review = {
                'reviewerID': user_id,
                'asin': product['asin'],
                'reviewerName': f'User{user_id[-3:]}',
                'helpful': [random.randint(0, 20), random.randint(0, 25)],
                'reviewText': f'Great {product["title"].split()[-1].lower()}! Really love this product.',
                'overall': random.choices([1, 2, 3, 4, 5], weights=[5, 10, 15, 35, 35])[0],
                'summary': 'Good product',
                'unixReviewTime': base_time + random.randint(0, 365 * 24 * 3600),
                'reviewTime': '01 1, 2024'
            }
            reviews.append(review)
        
        reviews_df = pd.DataFrame(reviews)
        metadata_df = pd.DataFrame(products)
        
        print(f"Generated {len(reviews_df):,} synthetic reviews")
        print(f"Generated {len(metadata_df):,} synthetic products")
        
        return reviews_df, metadata_df

    def download_and_process(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Download and process Amazon Beauty dataset, with synthetic fallback"""
        print("Starting Amazon Beauty dataset download...")
        
        try:
            # Try to download real data
            reviews_file = self.download_file(self.urls["reviews"], "Beauty_5.json.gz")
            metadata_file = self.download_file(self.urls["metadata"], "meta_Beauty.json.gz")
            
            # Parse reviews
            print("Processing reviews...")
            reviews_data = self.parse_json_gz(reviews_file)
            reviews_df = pd.DataFrame(reviews_data)
            
            # Parse metadata
            print("Processing metadata...")
            metadata_data = self.parse_json_gz(metadata_file)
            metadata_df = pd.DataFrame(metadata_data)
            
            print("✅ Successfully downloaded and processed real Amazon dataset")
            
        except Exception as e:
            print(f"❌ Failed to download real dataset: {e}")
            print("🔄 Falling back to synthetic data generation...")
            
            # Generate synthetic data as fallback
            reviews_df, metadata_df = self.generate_synthetic_data()
        
        # Save processed data
        reviews_csv = os.path.join(self.data_dir, "reviews.csv")
        metadata_csv = os.path.join(self.data_dir, "metadata.csv")
        
        reviews_df.to_csv(reviews_csv, index=False)
        metadata_df.to_csv(metadata_csv, index=False)
        
        print(f"Reviews saved to: {reviews_csv}")
        print(f"Metadata saved to: {metadata_csv}")
        print(f"Reviews shape: {reviews_df.shape}")
        print(f"Metadata shape: {metadata_df.shape}")
        
        return reviews_df, metadata_df

if __name__ == "__main__":
    downloader = AmazonDataDownloader()
    reviews_df, metadata_df = downloader.download_and_process()
    
    # Display basic statistics
    print("\n=== Dataset Statistics ===")
    print(f"Number of reviews: {len(reviews_df):,}")
    print(f"Number of unique users: {reviews_df['reviewerID'].nunique():,}")
    print(f"Number of unique products: {reviews_df['asin'].nunique():,}")
    print(f"Number of products with metadata: {len(metadata_df):,}")
    
    # Show sample data
    print("\n=== Sample Reviews ===")
    print(reviews_df.head())
    
    print("\n=== Sample Metadata ===")
    print(metadata_df.head())
