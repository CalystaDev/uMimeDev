import requests

url = "http://127.0.0.1:8000/generate-video"

payload = {
    "prompt": "how to learn to code",
    "voice_id": "jsCqWAovK2LkecY7zXl4", #freya?
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

# Print the response from the server
print(response.status_code)
print(response.json())


#todo:
# move subway surfers to storage --> DONE!
# connect gpt api -- prompt generation for each character --> DONE!
# image gen --> DONE!
# push final videos to storage --> DONE!
# test
# deploy api and how to speed up video gen