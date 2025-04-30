#!/usr/bin/env python3
import os
import glob
import datetime

# Get the site URL from _quarto.yml
site_url = "https://cetagostini.github.io/cetagostini.github.io/"
with open("_quarto.yml", "r") as f:
    for line in f:
        if "site-url:" in line:
            site_url = line.split("site-url:")[1].strip()
            if not site_url.endswith("/"):
                site_url += "/"
            break

# Get all .qmd files recursively
qmd_files = []
for root, dirs, files in os.walk('.'):
    if '.quarto' in root or '_site' in root or '.git' in root:
        continue
    for file in files:
        if file.endswith('.qmd') and not file.startswith('_'):
            file_path = os.path.join(root, file)
            if file_path.startswith('./'):
                file_path = file_path[2:]
            qmd_files.append(file_path)

# Create sitemap content
today = datetime.date.today().isoformat()
sitemap_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
sitemap_content += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'

for qmd_file in sorted(qmd_files):
    # Convert .qmd path to URL
    url_path = qmd_file.replace('.qmd', '.html')
    if url_path == 'index.html':
        url_path = ''
    
    # Set priority based on file path
    priority = '0.7'
    if url_path == '':
        priority = '1.0'
    elif url_path == 'about.html':
        priority = '0.8'
    elif url_path == 'articles.html':
        priority = '0.9'
    
    # Set change frequency
    changefreq = 'monthly'
    if 'articles/' in url_path:
        changefreq = 'weekly'

    sitemap_content += '  <url>\n'
    sitemap_content += f'    <loc>{site_url}{url_path}</loc>\n'
    sitemap_content += f'    <lastmod>{today}</lastmod>\n'
    sitemap_content += f'    <changefreq>{changefreq}</changefreq>\n'
    sitemap_content += f'    <priority>{priority}</priority>\n'
    sitemap_content += '  </url>\n'

sitemap_content += '</urlset>'

# Write sitemap to file
with open('sitemap.xml', 'w') as f:
    f.write(sitemap_content)

print(f"Sitemap.xml generated with {len(qmd_files)} URLs!") 