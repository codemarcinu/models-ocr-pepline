from core.tools.transcriber import VideoTranscriber
import sys

def verify():
    print("Testing VideoTranscriber with YouTubeTranscriptApi...")
    vt = VideoTranscriber()
    
    # Video: "The infinite hotel paradox" by TED-Ed (Video ID: OxGsU8oIWjY)
    video_id = "OxGsU8oIWjY" 
    print(f"Fetching transcript for {video_id}...")
    
    result = vt._try_fetch_transcript(video_id)
    
    if result:
        print("SUCCESS! Transcript fetched.")
        print(f"Text length: {len(result['text'])}")
        print(f"Segments: {len(result['segments'])}")
        print(f"Sample: {result['text'][:100]}...")
        return 0
    else:
        print("FAILURE! Could not fetch transcript.")
        return 1

if __name__ == "__main__":
    sys.exit(verify())
