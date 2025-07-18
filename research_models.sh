#!/bin/bash

set -e

echo "Searching for Cat-AIR"
curl -s "https://api.github.com/search/repositories?q=Cat-AIR+image+restoration" | jq -r '.items[0].html_url'

echo "Searching for Invert2Restore"
curl -s "https://api.github.com/search/repositories?q=Invert2Restore+image+restoration" | jq -r '.items[0].html_url'

echo "Searching for RAIM NTIRE 2025"
curl -s "https://api.github.com/search/repositories?q=RAIM+NTIRE+2025+image+restoration" | jq -r '.items[0].html_url'

echo "Searching for DarkIR"
curl -s "https://api.github.com/search/repositories?q=DarkIR+low-light+restoration" | jq -r '.items[0].html_url'

echo "Searching for Reelmind"
curl -s "https://api.github.com/search/repositories?q=Reelmind+AI+restoration" | jq -r '.items[0].html_url'

echo "Searching for MambaIRv2"
curl -s "https://api.github.com/search/repositories?q=MambaIRv2+image+restoration" | jq -r '.items[0].html_url'

echo "Searching for ZipIR"
curl -s "https://api.github.com/search/repositories?q=ZipIR+image+restoration" | jq -r '.items[0].html_url'

echo "Searching for DreamClear"
curl -s "https://api.github.com/search/repositories?q=DreamClear+image+restoration" | jq -r '.items[0].html_url'