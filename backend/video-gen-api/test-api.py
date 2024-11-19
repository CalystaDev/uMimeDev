import requests
import json
import uuid

# Base URL of the FastAPI server
# base_url = "http://127.0.0.1:8000"
base_url = "https://apitaketwo-1010927570704.us-east1.run.app"

# Payload for getting script and title
script_request_payload = {
    "prompt": "how to make a vanilla latte",
    "voice_id": "2EiwWnXFnvU5JabPnv8n"  # Example voice_id, you can replace this
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

    # Step 2: Generate unique video_id
    video_id = str(uuid.uuid4())

    # Store the script, title, image prompts in memory or a database (not shown here)
    # In your case, make sure you store them properly in the backend.
    generation_data = {
        video_id: {
            "image_prompts": image_prompts,
            "script": script,
            "script_with_time_delimiter": script_with_time_delimiter,
            "voice_id": voice_id
        }
    }

    # Step 3: Generate images
    generate_images_url = f"{base_url}/generate_images/{video_id}"
    images_response = requests.post(generate_images_url, json={}, headers=headers)

    if images_response.status_code == 200:
        print("Images generated successfully")
        print(images_response.json())
        
        # Step 4: Generate audio
        generate_audio_url = f"{base_url}/generate_audio/{video_id}"
        audio_response = requests.post(generate_audio_url, json={}, headers=headers)

        if audio_response.status_code == 200:
            print("Audio generated successfully")
            print(audio_response.json())

            # Step 5: Create the video
            create_video_url = f"{base_url}/create_video/{video_id}"
            video_response = requests.post(create_video_url, json={}, headers=headers)

            if video_response.status_code == 200:
                print("Video created successfully")
                print(video_response.json())
            else:
                print("Failed to create video")
                print(video_response.status_code)
                print(video_response.text)
        else:
            print("Failed to generate audio")
            print(audio_response.status_code)
            print(audio_response.text)
    else:
        print("Failed to generate images")
        print(images_response.status_code)
        print(images_response.text)

else:
    print("Failed to get script and title.")
    print(f"Status Code: {script_response.status_code}")
    print(script_response.text)
