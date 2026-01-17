
from adapters.google.drive_adapter import DriveBridge
import logging

logging.basicConfig(level=logging.INFO)
bridge = DriveBridge()
if bridge.authenticate():
    print("✅ Authenticated. Checking remote inbox...")
    bridge.check_remote_inbox()
else:
    print("❌ Authentication failed.")
