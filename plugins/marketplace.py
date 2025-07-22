import requests
import os

class Marketplace:
    def __init__(self, repo_url="https://api.github.com/repos/aisis/plugins/contents"):
        self.repo_url = repo_url

    def list_available_plugins(self):
        response = requests.get(self.repo_url)
        if response.status_code == 200:
            return [item['name'] for item in response.json() if item['type'] == 'file' and item['name'].endswith('.py')]
        return []

    def download_plugin(self, plugin_name, download_dir):
        url = f"{self.repo_url}/{plugin_name}"
        response = requests.get(url)
        if response.status_code == 200:
            download_url = response.json()['download_url']
            content = requests.get(download_url).text
            with open(os.path.join(download_dir, plugin_name), 'w') as f:
                f.write(content)
            return True
        return False
