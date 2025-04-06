import httpx
import json
from pathlib import Path

credentials_path = Path(__file__).resolve().parent.parent / "Credentials" / "awm_api_credentials.json"
with open(credentials_path) as data_file:
    api_credentials = json.load(data_file)
    
LOGIN_URL = api_credentials["login_url"]
TRACKINGS_URL = api_credentials["trackings_url"]
TRACKINGS_MANY_URL = api_credentials["trackings_many_url"]

USERNAME = api_credentials["username"]
PASSWORD = api_credentials["password"]
GRANT_TYPE = api_credentials["grant_type"]

async def get_access_token():
    async with httpx.AsyncClient() as client:
        
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "username": USERNAME,
            "password": PASSWORD,
            "grant_type": GRANT_TYPE
        }
        response = await client.post(LOGIN_URL, data=data, headers=headers)
        
        print(response)
        
        if response.status_code == 200:
            return response.json().get("access_token")  
        else:
            print(f"Failed to authenticate: {response.status_code}, {response.text}")
            return None
        
async def get_single_tracking(access_token, tracking_id = ""):
    headers = {"Authorization": f"Bearer {access_token}"}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(TRACKINGS_URL + tracking_id, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to get trackings: {response.status_code}, {response.text}")
            return None