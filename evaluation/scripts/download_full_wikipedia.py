#!/usr/bin/env python3
"""Download full Russian Wikipedia articles for proper RAG knowledge base."""

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Set
from urllib.parse import quote

import pandas as pd
import requests
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class WikipediaDownloader:
    """Downloads full Russian Wikipedia articles."""
    
    def __init__(self, output_dir: str = "./data/wikipedia_articles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAG-Research-Tool/1.0 (https://github.com/user/rag-platform)'
        })
    
    def get_article_ids_from_dataset(self, dataset_path: str) -> Set[str]:
        """Extract Wikipedia article IDs from the dataset."""
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract unique source files (these are Wikipedia page IDs)
        source_files = data['Файл'].unique()
        
        # Remove .txt extension to get page IDs
        page_ids = set()
        for filename in source_files:
            if filename and filename.endswith('.txt'):
                page_id = filename[:-4]  # Remove .txt
                if page_id.isdigit():
                    page_ids.add(page_id)
        
        print(f"Found {len(page_ids)} unique Wikipedia page IDs")
        return page_ids
    
    def download_article_by_id(self, page_id: str) -> Dict:
        """Download a Wikipedia article by page ID."""
        # Wikipedia API endpoint for Russian Wikipedia
        api_url = "https://ru.wikipedia.org/w/api.php"
        
        params = {
            'action': 'query',
            'format': 'json',
            'pageids': page_id,
            'prop': 'extracts|info',
            'exintro': False,  # Get full article, not just intro
            'explaintext': True,  # Plain text, no HTML
            'exsectionformat': 'plain',
            'inprop': 'url'
        }
        
        try:
            response = self.session.get(api_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'query' in data and 'pages' in data['query']:
                page_data = list(data['query']['pages'].values())[0]
                
                if 'missing' in page_data:
                    print(f"⚠️  Page ID {page_id} not found")
                    return None
                
                return {
                    'page_id': page_id,
                    'title': page_data.get('title', ''),
                    'content': page_data.get('extract', ''),
                    'url': page_data.get('fullurl', ''),
                    'length': len(page_data.get('extract', ''))
                }
            
        except Exception as e:
            print(f"❌ Error downloading page {page_id}: {e}")
            return None
        
        return None
    
    def download_all_articles(self, dataset_path: str) -> List[Dict]:
        """Download all Wikipedia articles referenced in the dataset."""
        page_ids = self.get_article_ids_from_dataset(dataset_path)
        
        print(f"\\n📥 Downloading {len(page_ids)} Wikipedia articles...")
        
        articles = []
        failed_downloads = []
        
        for page_id in tqdm(page_ids, desc="Downloading articles"):
            # Check if already downloaded
            cache_file = self.output_dir / f"{page_id}.json"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        article = json.load(f)
                        articles.append(article)
                        continue
                except:
                    pass
            
            # Download article
            article = self.download_article_by_id(page_id)
            
            if article:
                # Save to cache
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(article, f, ensure_ascii=False, indent=2)
                
                articles.append(article)
                
                # Respect Wikipedia rate limits
                time.sleep(0.1)
            else:
                failed_downloads.append(page_id)
        
        print(f"\\n✅ Successfully downloaded: {len(articles)} articles")
        if failed_downloads:
            print(f"❌ Failed downloads: {len(failed_downloads)} articles")
            print(f"   Failed IDs: {failed_downloads[:10]}{'...' if len(failed_downloads) > 10 else ''}")
        
        return articles
    
    def create_knowledge_base(self, articles: List[Dict]) -> None:
        """Create a comprehensive knowledge base from articles."""
        print("\\n📚 Creating knowledge base...")
        
        # Filter out empty articles
        valid_articles = [a for a in articles if a.get('content') and len(a['content']) > 100]
        
        # Calculate statistics
        total_chars = sum(len(a['content']) for a in valid_articles)
        avg_length = total_chars / len(valid_articles) if valid_articles else 0
        
        print(f"Valid articles: {len(valid_articles)}")
        print(f"Total characters: {total_chars:,}")
        print(f"Average article length: {avg_length:,.0f} characters")
        print(f"Estimated tokens: {total_chars//4:,}")
        
        # Save complete knowledge base
        kb_path = self.output_dir / "wikipedia_knowledge_base.json"
        with open(kb_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'total_articles': len(valid_articles),
                    'total_characters': total_chars,
                    'average_length': avg_length,
                    'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'articles': valid_articles
            }, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Knowledge base saved: {kb_path}")
        print(f"📊 File size: {kb_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        return valid_articles
    
    def show_sample_articles(self, articles: List[Dict], n: int = 3) -> None:
        """Show sample articles for verification."""
        print(f"\\n📖 Sample Articles (first {n}):")
        print("=" * 60)
        
        for i, article in enumerate(articles[:n]):
            print(f"\\n{i+1}. {article['title']}")
            print(f"   Page ID: {article['page_id']}")
            print(f"   Length: {article['length']:,} characters")
            print(f"   Content preview: {article['content'][:200]}...")
            print(f"   URL: {article['url']}")


def main():
    """Download full Wikipedia articles for RAG knowledge base."""
    print("📚 Russian Wikipedia Articles Downloader")
    print("=" * 60)
    
    # Check if dataset exists
    dataset_path = "data/ru_rag_test_dataset.pkl"
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please run: python scripts/download_dataset.py")
        return
    
    # Initialize downloader
    downloader = WikipediaDownloader()
    
    # Download all articles
    articles = downloader.download_all_articles(dataset_path)
    
    if not articles:
        print("❌ No articles downloaded successfully")
        return
    
    # Create knowledge base
    knowledge_base = downloader.create_knowledge_base(articles)
    
    # Show samples
    downloader.show_sample_articles(knowledge_base)
    
    print("\\n" + "=" * 60)
    print("🎉 Wikipedia knowledge base created!")
    print("\\nNext steps:")
    print("1. Update RAG config to use full Wikipedia articles")
    print("2. Run experiments with proper knowledge base")
    print("3. Compare performance vs snippet-based approach")
    print("=" * 60)


if __name__ == "__main__":
    main()