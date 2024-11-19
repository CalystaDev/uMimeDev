import requests

# Set the base URL of your Cloud Run service
BASE_URL = "https://apitaketwo-1010927570704.us-east1.run.app"

# Example data for testing
test_prompt = "how to choose wine to drink"
test_voice_id = "jsCqWAovK2LkecY7zXl4"  # Replace with the actual voice ID if needed

def test_generate_script():
    url = f"{BASE_URL}/generate_script"
    payload = {
        "prompt": test_prompt,
        "voice_id": test_voice_id
    }
    response = requests.post(url, json=payload)
    
    # Check for successful response
    if response.status_code == 200:
        try:
            response_json = response.json()
            print("Generate Script Response:", response_json)
            return response_json.get("video_id")
        except requests.exceptions.JSONDecodeError:
            print("Error: Response is not in JSON format.")
            print("Raw Response Content:", response.text)
    else:
        print(f"Error: Received status code {response.status_code}")
        print("Raw Response Content:", response.text)
        return None


# Step 2: Test the generate_images endpoint
def test_generate_images(video_id):
    url = f"{BASE_URL}/generate_images/{video_id}"
    response = requests.post(url)
    print("Generate Images Response:", response.json())
    return response.json().get("image_paths")

# Step 3: Test the generate_audio endpoint
def test_generate_audio(video_id):
    url = f"{BASE_URL}/generate_audio/{video_id}"
    response = requests.post(url)
    print("Generate Audio Response:", response.json())
    return response.json().get("audio_file")

# Step 4: Test the create_video endpoint
def test_create_video(video_id):
    url = f"{BASE_URL}/create_video/{video_id}"
    response = requests.post(url)
    
    # Check if the response is successful and contains JSON
    if response.status_code == 200:
        try:
            json_response = response.json()
            print("Create Video Response:", json_response)
            return json_response.get("video_file")
        except ValueError:  # Handle JSON decode error
            print("Error: Response is not JSON.")
    else:
        print(f"Error: Received status code {response.status_code}")
        print("Response text:", response.text)  # Log the response content for debugging
    
    return None

# Run the tests in sequence
if __name__ == "__main__":
    video_id = test_generate_script()
    if video_id:
        image_paths = test_generate_images(video_id)
        if image_paths:
            audio_file = test_generate_audio(video_id)
            if audio_file:
                video_file = test_create_video(video_id)
                print("Final Video File URL:", video_file)





















# import requests
# import json
# import uuid

# # Base URL of the FastAPI server
# # base_url = "http://127.0.0.1:8000"
# base_url = ""

# # Payload for getting script and title
# script_request_payload = {
#     "prompt": "how to get a boyfriend",
#     "voice_id": "jsCqWAovK2LkecY7zXl4"  # Example voice_id, you can replace this
# }

# # Headers for the requests
# headers = {
#     "Content-Type": "application/json"
# }

# # Step 1: Get the script, title, and image prompts from the /get-script-and-title endpoint
# get_script_url = f"{base_url}/get-script-and-title"
# script_response = requests.post(get_script_url, json=script_request_payload, headers=headers)

# # Check if the response from /get-script-and-title is successful
# if script_response.status_code == 200:
#     script_data = script_response.json()
#     print("Script and Title Response:")
#     print(json.dumps(script_data, indent=2))

#     # Extract the data required for the /generate-video endpoint
#     image_prompts = script_data.get("image_prompts", [])
#     script = script_data.get("script", "")
#     script_with_time_delimiter = script_data.get("script_with_time_delimiter", "")
#     voice_id = script_request_payload["voice_id"]

#     print("THIS IS THE TITLE: ", script_data.get("title"))

#     # Step 2: Generate unique video_id
#     video_id = str(uuid.uuid4())

#     # Store the script, title, image prompts in memory or a database (not shown here)
#     # In your case, make sure you store them properly in the backend.
#     generation_data = {
#         video_id: {
#             "image_prompts": image_prompts,
#             "script": script,
#             "script_with_time_delimiter": script_with_time_delimiter,
#             "voice_id": voice_id
#         }
#     }

#     # Step 3: Generate images
#     generate_images_url = f"{base_url}/generate_images/{video_id}"
#     images_response = requests.post(generate_images_url, json={}, headers=headers)

#     if images_response.status_code == 200:
#         print("Images generated successfully")
#         print(images_response.json())
        
#         # Step 4: Generate audio
#         generate_audio_url = f"{base_url}/generate_audio/{video_id}"
#         audio_response = requests.post(generate_audio_url, json={}, headers=headers)

#         if audio_response.status_code == 200:
#             print("Audio generated successfully")
#             print(audio_response.json())

#             # Step 5: Create the video
#             create_video_url = f"{base_url}/create_video/{video_id}"
#             video_response = requests.post(create_video_url, json={}, headers=headers)

#             if video_response.status_code == 200:
#                 print("Video created successfully")
#                 print(video_response.json())
#             else:
#                 print("Failed to create video")
#                 print(video_response.status_code)
#                 print(video_response.text)
#         else:
#             print("Failed to generate audio")
#             print(audio_response.status_code)
#             print(audio_response.text)
#     else:
#         print("Failed to generate images")
#         print(images_response.status_code)
#         print(images_response.text)

# else:
#     print("Failed to get script and title.")
#     print(f"Status Code: {script_response.status_code}")
#     print(script_response.text)