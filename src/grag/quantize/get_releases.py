import requests

# 
# def get_github_releases(user, repo):
#     url = f"https://api.github.com/repos/{user}/{repo}/releases/latest"
#     response = requests.get(url)
#     releases = response.json()
#     return releases
# 
# 
# # Example usage
# latest_release = get_github_releases('ggerganov',
#                                      'llama.cpp')  # Replace 'nodejs' and 'node' with the appropriate user and repository
# if 'tag_name' in latest_release:
#     print(f"Latest Release Tag: {latest_release['tag_name']}, Assets: {len(latest_release['assets'])}")
# else:
#     print("Error fetching latest release:", latest_release.get('message', 'No error message provided'))


def get_asset_download_url(user, repo, asset_name_pattern):
    """Fetches the download URL of the first asset that matches a given name pattern in the latest release of the specified repository.

    Args:
        user (str): GitHub username or organization of the repository.
        repo (str): Repository name.
        asset_name_pattern (str): Substring to match in the asset's name.

    Returns:
        str: The download URL of the matching asset, or None if no match is found.
    """
    url = f"https://api.github.com/repos/{user}/{repo}/releases/latest"
    response = requests.get(url)
    if response.status_code == 200:
        release = response.json()
        for asset in release.get('assets', []):
            if asset_name_pattern in asset['name']:
                return asset['browser_download_url']
        print("No asset found matching the pattern.")
    else:
        print("Failed to fetch release info:", response.status_code)
    return None


def download_release_asset(download_url, target_path):
    """Downloads a file from a given URL and saves it to a specified path.

    Args:
        download_url (str): The URL of the file to download.
        target_path (str): Path where the file will be saved.
    """
    response = requests.get(download_url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded successfully to {target_path}")
    else:
        print(f"Failed to download file: {response.status_code}")


# Example usage
user = 'ggerganov'
repo = 'llama.cpp'
asset_name_pattern = 'ubuntu-x64'  # Adjust the pattern to match the desired asset
target_path = 'llama-cpp-ubuntu-x64.zip'
download_url = get_asset_download_url(user, repo, asset_name_pattern)
if download_url:
    download_release_asset(download_url, target_path)
