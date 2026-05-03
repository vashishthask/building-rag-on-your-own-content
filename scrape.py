import requests
import json
import os
import re
import html

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
}

def clean_html(raw_html):
    """Strip HTML tags and clean up whitespace"""
    clean = re.sub(r'<[^>]+>', '', raw_html)
    clean = re.sub(r'\n\s*\n', '\n\n', clean)
    return clean.strip()

def get_all_posts(base_url):
    all_posts = []
    page = 1
    
    while True:
        api_url = f"{base_url}/wp-json/wp/v2/posts"
        params = {
            'per_page': 20,
            'page': page,
            '_fields': 'title,link,content'
        }
        
        response = requests.get(api_url, params=params, headers=headers)
        
        if response.status_code != 200:
            break
            
        posts = response.json()
        
        if not posts:
            break
            
        all_posts.extend(posts)
        print(f"Fetched page {page} — {len(posts)} posts")
        page += 1
    
    return all_posts

def save_posts(posts, output_dir="docs"):
    """Save each post as a clean text file"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, post in enumerate(posts):
        title = html.unescape(post['title']['rendered'])
        url = post['link']
        content = clean_html(post['content']['rendered'])
        
        # Create a clean filename
        filename = f"{i+1:03d}_{re.sub(r'[^a-z0-9]+', '-', title.lower())[:50]}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"TITLE: {title}\n")
            f.write(f"URL: {url}\n")
            f.write(f"---\n\n")
            f.write(content)
        
    print(f"\nSaved {len(posts)} posts to /{output_dir} folder")

# Run it
print("Fetching posts from agilebuddha.in...\n")
posts = get_all_posts("https://www.agilebuddha.in")
print(f"\nTotal posts found: {len(posts)}")
save_posts(posts)
