import requests

# Set the base URL of your Cloud Run service
BASE_URL = "https://videogenapi-410774176567.us-east1.run.app"

# Example data for testing
test_prompt = "how to make a vision board"
test_voice_id = "2EiwWnXFnvU5JabPnv8n"
background_id = "minecraft.mov" #background video id

def test_generate_script():
    url = f"{BASE_URL}/generate_script"
    payload = {
        "prompt": test_prompt,
        "voice_id": test_voice_id
    }
    response = requests.post(url, json=payload)
    
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

# Step 2: Test the select_background endpoint
def test_select_background(video_id):
    url = f"{BASE_URL}/select_background/{video_id}"
    payload = {
        "background_id": background_id  # Now passing background_id as a string
    }
    response = requests.post(url, json=payload)
    
    # Check for successful response
    if response.status_code == 200:
        try:
            response_json = response.json()
            print("Select Background Response:", response_json)
            return response_json.get("video_path")
        except requests.exceptions.JSONDecodeError:
            print("Error: Response is not in JSON format.")
            print("Raw Response Content:", response.text)
    else:
        print(f"Error: Received status code {response.status_code}")
        print("Raw Response Content:", response.text)
        return None

# Step 3: Test the generate_images endpoint
def test_generate_images(video_id):
    url = f"{BASE_URL}/generate_images/{video_id}"
    response = requests.post(url)
    print("Generate Images Response:", response.json())
    return response.json().get("image_paths")

# Step 4: Test the generate_audio endpoint
def test_generate_audio(video_id):
    url = f"{BASE_URL}/generate_audio/{video_id}"
    response = requests.post(url)
    print("Generate Audio Response:", response.json())
    return response.json().get("audio_file")

# Step 5: Test the create_video endpoint
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
        background_video = test_select_background(video_id)
        if background_video:
            image_paths = test_generate_images(video_id)
            if image_paths:
                audio_file = test_generate_audio(video_id)
                if audio_file:
                    video_file = test_create_video(video_id)
                    print("Final Video File URL:", video_file)