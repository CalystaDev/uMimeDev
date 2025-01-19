import requests

BASE_URL = "https://videogenapi-410774176567.us-east1.run.app"

# Example data for testing
test_prompt = "explain to me how the water cycle works in solid detail"
test_voice_id = "ZQe5CZNOzWyzPSCn5a3c"
background_id = "subwaysurfers"
music_id = "sport"

def test_generate_title():
    """
    Test the title generation endpoint.
    """
    url = f"{BASE_URL}/generate_title"
    payload = {
        "prompt": test_prompt,
        "voice_id": test_voice_id
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        try:
            response_json = response.json()
            print("Generate Title Response:", response_json)
            return response_json.get("title")
        except requests.exceptions.JSONDecodeError:
            print("Error: Response is not in JSON format.")
            print("Raw Response Content:", response.text)
    else:
        print(f"Error: Received status code {response.status_code}")
        print("Raw Response Content:", response.text)
    return None

def test_generate_script():
    """
    Test the script generation endpoint.
    """
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

def test_select_background(video_id):
    """
    Test the background video selection endpoint.
    """
    url = f"{BASE_URL}/select_background/{video_id}"
    payload = {
        "background_id": background_id
    }
    response = requests.post(url, json=payload)
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

def test_select_background_music(video_id):
    """
    Test the background music selection endpoint.
    """
    url = f"{BASE_URL}/select_background_music/{video_id}"
    payload = {
        "music_id": music_id
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        try:
            response_json = response.json()
            print("Select Background Music Response:", response_json)
            return response_json.get("music_path")
        except requests.exceptions.JSONDecodeError:
            print("Error: Response is not in JSON format.")
            print("Raw Response Content:", response.text)
    else:
        print(f"Error: Received status code {response.status_code}")
        print("Raw Response Content:", response.text)
    return None

def test_generate_images(video_id):
    """
    Test the image generation endpoint.
    """
    url = f"{BASE_URL}/generate_images/{video_id}"
    response = requests.post(url)
    if response.status_code == 200:
        print("Generate Images Response:", response.json())
        return response.json().get("image_paths")
    else:
        print(f"Error: Received status code {response.status_code}")
        print("Raw Response Content:", response.text)
    return None

def test_generate_audio(video_id):
    """
    Test the audio generation endpoint.
    """
    url = f"{BASE_URL}/generate_audio/{video_id}"
    payload = {
        "voice_id": test_voice_id
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Generate Audio Response:", response.json())
        return response.json().get("audio_file")
    else:
        print(f"Error: Received status code {response.status_code}")
        print("Raw Response Content:", response.text)
    return None

def test_create_video(video_id):
    """
    Test the video creation endpoint.
    """
    url = f"{BASE_URL}/create_video/{video_id}"
    response = requests.post(url)
    if response.status_code == 200:
        try:
            json_response = response.json()
            print("Create Video Response:", json_response)
            return json_response.get("video_file")
        except ValueError:
            print("Error: Response is not JSON.")
    else:
        print(f"Error: Received status code {response.status_code}")
        print("Response text:", response.text)
    return None

# Run the tests in sequence
if __name__ == "__main__":
    title = test_generate_title()
    if title:
        video_id = test_generate_script()
        if video_id:
            background_video = test_select_background(video_id)
            if background_video:
                background_music = test_select_background_music(video_id)
                if background_music or music_id == "none":
                    image_paths = test_generate_images(video_id)
                    if image_paths:
                        audio_file = test_generate_audio(video_id)
                        if audio_file:
                            video_file = test_create_video(video_id)
                            print("Final Video File URL:", video_file)