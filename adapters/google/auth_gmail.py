"""
Gmail OAuth 2.0 Authentication Setup

Run this script once to authorize the application for Gmail access.
Requires credentials.json from Google Cloud Console with Gmail API enabled.

Usage:
    python setup_gmail_auth.py
"""

import logging
import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

# Configuration
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token_gmail.json'

# Gmail scopes - gmail.modify allows read, archive, label, delete (but not send)
SCOPES = [
    'https://www.googleapis.com/auth/gmail.modify',  # Read, archive, label, trash
    'https://www.googleapis.com/auth/gmail.labels',  # Manage labels
]


def authenticate():
    print("--- Gmail Authentication Setup ---")
    print(f"Scopes requested: {SCOPES}")

    if not os.path.exists(CREDENTIALS_FILE):
        print(f"\nError: {CREDENTIALS_FILE} not found!")
        print("Please download it from Google Cloud Console:")
        print("1. Go to https://console.cloud.google.com/apis/credentials")
        print("2. Create OAuth 2.0 Client ID (Desktop app)")
        print("3. Download JSON and save as 'credentials.json'")
        print("4. Enable Gmail API in APIs & Services > Library")
        return

    try:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)

        print("\n1. A browser window will open for authorization.")
        print("   If not, copy the URL from below and paste in browser.")
        print("2. Sign in with your Google account")
        print("3. Grant requested permissions\n")

        creds = flow.run_local_server(port=0, open_browser=False)

        # Save the credentials for the next run
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())

        print(f"\n✅ Success! Token saved to {TOKEN_FILE}")
        print("You can now use the Gmail Agent.")

    except OSError as e:
        print(f"\nError starting local server: {e}")
        print("Ensure you are running this on a machine that can bind to localhost.")
    except Exception as e:
        print(f"\nError during OAuth flow: {e}")


if __name__ == '__main__':
    authenticate()
