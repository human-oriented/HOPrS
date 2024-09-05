import requests
import argparse
import os
import sys
import re

# Centralized error handler
def handle_request_error(e):
    print(f"Error: {str(e)}")
    sys.exit(1)

# Download file given a URL
def download_file(url):
    local_filename = url.split('/')[-1]
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded: {local_filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {url}: {str(e)}")

# Recursively check JSON for URLs and download them
def process_response_for_urls(response_json):
    if response_json is None:
        print("No URLs to pull down")
        return
    if isinstance(response_json, dict):
        for key, value in response_json.items():
            if isinstance(value, str) and is_url(value):
                download_file(value)
            elif isinstance(value, (dict, list)):
                process_response_for_urls(value)
    elif isinstance(response_json, list):
        for item in response_json:
            if isinstance(item, str) and is_url(item):
                download_file(item)
            elif isinstance(item, (dict, list)):
                process_response_for_urls(item)

# Check if a string is a URL
def is_url(string):
    url_pattern = re.compile(r'https?://[^\s]+')
    return re.match(url_pattern, string) is not None

# Function to handle /hoprs/compare
def hoprs_compare(server_url, original_image_qt, new_image, threshold=10, compare_depth=5):
    url = f"{server_url}/hoprs/compare"
    try:
        files = {
            'original_image_qt': open(original_image_qt, 'rb'),
            'new_image': open(new_image, 'rb')
        }
        params = {'threshold': threshold, 'compare_depth': compare_depth}
        response = requests.post(url, files=files, params=params)
        response.raise_for_status()
        response_json = response.json()
        print(response_json)
        process_response_for_urls(response_json)
    except requests.exceptions.RequestException as e:
        handle_request_error(e)

# Function to handle /hoprs/encode
def hoprs_encode(server_url, file, depth=5, algorithm='pdq', resize=None, crop=None, note=None):
    url = f"{server_url}/hoprs/encode"
    try:
        print(f"Uploading file: {file}")
        files = {'file': open(file, 'rb')}
        print(f"Encoding with depth={depth}, algorithm={algorithm}, resize={resize}, crop={crop}, note={note}")
        params = {
            'depth': depth,
            'algorithm': algorithm,
            'resize': resize,
            'crop': crop,
            'note': note or "From CLI"
        }
        response = requests.post(url, files=files, params=params)
        response.raise_for_status()
        response_json = response.json()
        print(str(response_json))
        
    except requests.exceptions.RequestException as e:
        handle_request_error(e)

# Function to handle /hoprs/download
def hoprs_download(server_url, qt_ref):
    url = f"{server_url}/hoprs/download"
    try:
        params = {'qt_ref': qt_ref}
        response = requests.post(url, params=params)
        response.raise_for_status()
        response_json = response.json()
        print(response_json)
        process_response_for_urls(response_json)
    except requests.exceptions.RequestException as e:
        handle_request_error(e)

# Function to handle /hoprs/count
def hoprs_count(server_url):
    url = f"{server_url}/hoprs/count"
    try:
        response = requests.get(url)
        response.raise_for_status()
        response_json = response.json()
        print(response_json)
    except requests.exceptions.RequestException as e:
        handle_request_error(e)

# Function to handle /hoprs/list
def hoprs_list(server_url):
    url = f"{server_url}/hoprs/list"
    try:
        response = requests.get(url)
        response.raise_for_status()
        response_json = response.json()
        print(response_json)
        process_response_for_urls(response_json)
    except requests.exceptions.RequestException as e:
        handle_request_error(e)

# Function to handle /hoprs/search
def hoprs_search(server_url, image):
    url = f"{server_url}/hoprs/search"
    try:
        files = {'image': open(image, 'rb')}
        response = requests.post(url, files=files)
        response.raise_for_status()
        response_json = response.json()
        print(response_json)
        process_response_for_urls(response_json)
    except requests.exceptions.RequestException as e:
        handle_request_error(e)

# Function to handle /hoprs/version
def hoprs_version(server_url):
    url = f"{server_url}/hoprs/version"
    try:
        response = requests.get(url)
        response.raise_for_status()
        response_json = response.json()
        print(response_json)
        process_response_for_urls(response_json)
    except requests.exceptions.RequestException as e:
        handle_request_error(e)

# Main dispatcher function to choose API function based on first argument
def main():
    parser = argparse.ArgumentParser(description='HOPrS CLI utility to interact with the HOPrS API.')
    parser.add_argument('--server_url', default=os.getenv('HOPRS_SERVER_URL', 'http://localhost:8000'),
                        help='Server URL (default: http://localhost:8000 or environment variable HOPRS_SERVER_URL)')
    subparsers = parser.add_subparsers(dest='command', required=True, help='API function to call')

    # Subparser for /compare
    compare_parser = subparsers.add_parser('compare', help='Compare two images')
    compare_parser.add_argument('original_image_qt', help='Path to the original QT image')
    compare_parser.add_argument('new_image', help='Path to the new image to compare')
    compare_parser.add_argument('--threshold', type=int, default=10, help='Threshold for comparison (default: 10)')
    compare_parser.add_argument('--compare_depth', type=int, default=5, help='Depth for comparison (default: 5)')

    # Subparser for /encode
    encode_parser = subparsers.add_parser('encode', help='Encode an image')
    encode_parser.add_argument('file', help='Path to the image file')
    encode_parser.add_argument('--depth', type=int, default=5, help='Depth for encoding (default: 5)')
    encode_parser.add_argument('--algorithm', default='pdq', help='Perceptual algorithm to use (default: pdq)')
    encode_parser.add_argument('--resize', help='Resize dimensions (comma-separated)')
    encode_parser.add_argument('--crop', help='Crop coordinates (comma-separated)')
    encode_parser.add_argument('--note', help='Optional comment or note')

    # Subparser for /download
    download_parser = subparsers.add_parser('download', help='Download a QT reference')
    download_parser.add_argument('qt_ref', help='QT reference to find in the database and download')

    # Subparser for /count
    subparsers.add_parser('count', help='Get count from the database')

    # Subparser for /list
    subparsers.add_parser('list', help='List items from the database')

    # Subparser for /search
    search_parser = subparsers.add_parser('search', help='Search an image')
    search_parser.add_argument('image', help='Path to the image file')

    # Subparser for /version
    subparsers.add_parser('version', help='Get version information')

    # Parse arguments
    args = parser.parse_args()

    # Dispatch to the correct function
    if args.command == 'compare':
        hoprs_compare(args.server_url, args.original_image_qt, args.new_image, args.threshold, args.compare_depth)
    elif args.command == 'encode':
        hoprs_encode(args.server_url, args.file, args.depth, args.algorithm, args.resize, args.crop, args.note)
    elif args.command == 'download':
        hoprs_download(args.server_url, args.qt_ref)
    elif args.command == 'count':
        hoprs_count(args.server_url)
    elif args.command == 'list':
        hoprs_list(args.server_url)
    elif args.command == 'search':
        hoprs_search(args.server_url, args.image)
    elif args.command == 'version':
        hoprs_version(args.server_url)

if __name__ == "__main__":
    main()

