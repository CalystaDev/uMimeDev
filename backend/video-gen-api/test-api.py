import requests

url = "http://127.0.0.1:8000/generate-video"

payload = {
    "prompt": "Okay, so like, making mac n' cheese is super easy $ First, you boil the pasta until it's, like, perfectly al dente, then drain it $ In a separate pan, melt some butter and whisk in flour to make this cute little roux $ Slowly add milk and stir until it's, like, thick and creamy $ Then, mix in your cheese until it’s all melty and dreamy $ Combine that with your pasta, and if you want it extra fabulous, sprinkle more cheese on top and bake until it’s golden and bubbly $ And, voila! You’ve got, like, the yummiest mac n' cheese ever",
    "voice_id": "jsCqWAovK2LkecY7zXl4",
    "api_key": "YOUR_API_KEY"
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

# Print the response from the server
print(response.status_code)
print(response.json())
