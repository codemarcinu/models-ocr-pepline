import argparse
import sys
import logging
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from adapters.google.keep_adapter import KeepAdapter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Sync Google Keep notes to Obsidian (Unofficial API)")
    parser.add_argument("--login", action="store_true", help="Perform login (requires username and password)")
    parser.add_argument("--username", help="Google Account Email")
    parser.add_argument("--password", help="App Password (required for 2FA/Passkeys)")
    parser.add_argument("--label", default="Obsidian", help="Label to sync (default: Obsidian)")
    
    args = parser.parse_args()
    
    adapter = KeepAdapter()
    
    if args.login:
        if not args.username or not args.password:
            print("Error: --username and --password are required for login.")
            return
        
        print(f"Attempting login for {args.username}...")
        if adapter.login(args.username, args.password):
            print("Login successful! Token saved.")
        else:
            print("Login failed. Please check your credentials.")
    else:
        # Try to sync using cached token
        if adapter.login():
            adapter.sync(label_name=args.label)
        else:
            print("Not authenticated. Please run with --login --username <email> --password <app_password>")

if __name__ == "__main__":
    main()
