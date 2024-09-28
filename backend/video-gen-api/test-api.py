import requests

url = "http://127.0.0.1:8000/generate-video"

payload = {
    "prompt": "how to fold clothes",
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
# update prompts for gpt + image gen --> DONE!
# incorporate ari fix for timing --> DONE!
# test
# deploy server

#notes:
# gpt output quality may be impacted by the example since example is valley girl. 
# perhaps we should change the example prompt for each character.