import gpsoauth
import argparse
import sys

def test_login(email, password):
    print(f"Testing gpsoauth login for {email}...")
    
    # Common android ID for keep
    android_id = "0123456789abcdef" 
    
    clean_password = password.replace(" ", "")
    
    try:
        response = gpsoauth.perform_master_login(
            email, 
            clean_password, 
            android_id
        )
        
        print("\nResponse:")
        print(response)
        
        if "Token" in response:
            print("\nSUCCESS! Master Token obtained.")
        elif "Error" in response:
            print(f"\nFAILURE: {response.get('Error')}")
            if response.get('Error') == "BadAuthentication":
                print("Tip: Check App Password. Ensure 2FA is on.")
            if response.get('Error') == "NeedsBrowser":
                print("Tip: Account might be protected by advanced security or captcha.")
        else:
            print("\nUnknown response format.")
            
    except Exception as e:
        print(f"\nEXCEPTION: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("email")
    parser.add_argument("password")
    args = parser.parse_args()
    
    test_login(args.email, args.password)
