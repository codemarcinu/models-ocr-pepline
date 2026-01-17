import logging
import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

# Configuration
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token_drive.json'
SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate():
    print(f"--- Google Drive Authentication Setup ---")
    
    if not os.path.exists(CREDENTIALS_FILE):
        print(f"Error: {CREDENTIALS_FILE} not found!")
        return

    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
    
    print("\n1. Please visit the following URL to authorize this application:")
    # We use run_local_server with a fixed port if possible, or random.
    # open_browser=False prevents the xdg-open error.
    try:
        creds = flow.run_local_server(port=0, open_browser=False)
    except OSError as e:
        print(f"\nError starting local server: {e}")
        print("Ensure you are running this on a machine that can bind to localhost.")
        return

    # Save the credentials for the next run
    with open(TOKEN_FILE, 'w') as token:
        token.write(creds.to_json())
    
    print(f"\nSuccess! Token saved to {TOKEN_FILE}")
    print("You can now restart the DriveBridge service.")

if __name__ == '__main__':
    authenticate()
