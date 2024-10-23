import requests
import json

# Base URL of the FastAPI server
# base_url = "http://127.0.0.1:8000"
base_url = "https://fastapi-video-app-1010927570704.us-east1.run.app"

# Payload for getting script and title
script_request_payload = {
    "prompt": "how to get a boyfriend",
    "voice_id": "jsCqWAovK2LkecY7zXl4"  # Example voice_id, you can replace this
}

# Headers for the requests
headers = {
    "Content-Type": "application/json"
}

# Step 1: Get the script, title, and image prompts from the /get-script-and-title endpoint
get_script_url = f"{base_url}/get-script-and-title"
script_response = requests.post(get_script_url, json=script_request_payload, headers=headers)

# Check if the response from /get-script-and-title is successful
if script_response.status_code == 200:
    script_data = script_response.json()
    print("Script and Title Response:")
    print(json.dumps(script_data, indent=2))

    # Extract the data required for the /generate-video endpoint
    image_prompts = script_data.get("image_prompts", [])
    script = script_data.get("script", "")
    script_with_time_delimiter = script_data.get("script_with_time_delimiter", "")
    voice_id = script_request_payload["voice_id"]

    print("THIS IS THE TITLE: ", script_data.get("title"))

    # Prepare payload for /generate-video endpoint
    generate_video_payload = {
        "image_prompts": image_prompts,
        "script": script,
        "script_with_time_delimiter": script_with_time_delimiter,
        "voice_id": voice_id
    }

    # Step 2: Use the response from /get-script-and-title to generate the video
    generate_video_url = f"{base_url}/generate-video"
    video_response = requests.post(generate_video_url, json=generate_video_payload, headers=headers)

    # Print the response from the /generate-video endpoint
    print("Generate Video Response:")
    print(video_response.status_code)
    print(video_response.json())
else:
    print("Failed to get script and title.")
    print(f"Status Code: {script_response.status_code}")
    print(script_response.text)








# import requests

# url = "http://127.0.0.1:8000/generate-video"

# payload = {
#     "prompt": "how to fold clothes",
#     "voice_id": "jsCqWAovK2LkecY7zXl4", #freya?
# }

# headers = {
#     "Content-Type": "application/json"
# }

# response = requests.post(url, json=payload, headers=headers)

# # Print the response from the server
# print(response.status_code)
# print(response.json())


#todo:
# update prompts for gpt + image gen --> DONE!
# incorporate ari fix for timing --> DONE!
# test
# deploy server

#notes:
# gpt output quality may be impacted by the example since example is valley girl. 
# perhaps we should change the example prompt for each character.

#