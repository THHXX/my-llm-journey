import requests
import json

def test_api():
    base_url = "http://localhost:8000"
    
    # Test Root
    try:
        resp = requests.get(f"{base_url}/")
        print(f"Root endpoint status: {resp.status_code}")
        print(f"Root response: {resp.json()}")
    except Exception as e:
        print(f"Root endpoint failed: {e}")
        return

    # Test Ask
    try:
        payload = {"question": "你好，请问你是谁？"}
        headers = {"Content-Type": "application/json"}
        resp = requests.post(f"{base_url}/ask", json=payload, headers=headers)
        print(f"Ask endpoint status: {resp.status_code}")
        print(f"Ask response: {resp.json()}")
    except Exception as e:
        print(f"Ask endpoint failed: {e}")

if __name__ == "__main__":
    test_api()
