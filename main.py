# ================================================================================
# FULL YOUTUBE VIDEO SUMMARIZER - MAIN.PY
# ================================================================================
"""
This script extracts YouTube transcripts and summarizes them using CrewAI
and the Gemini LLM.
"""

# --------------------------
# IMPORT DEPENDENCIES
# --------------------------
import re
import os
from typing import Optional, List
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi

# Error handling for YouTube Transcript API
try:
    from youtube_transcript_api._errors import TranscriptsDisabled, VideoUnavailable, NoTranscriptFound
except ImportError:
    TranscriptsDisabled = Exception
    VideoUnavailable = Exception
    NoTranscriptFound = Exception

print("✓ All dependencies imported successfully")

# --------------------------
# ENVIRONMENT CONFIGURATION
# --------------------------
# --------------------------
# GEMINI API KEY CONFIGURATION
# --------------------------

# Option 1: Paste your Gemini API key directly here (replace placeholder)
GEMINI_API_KEY = "YOUR_GEMINI_KEY_HERE"  # <-- Replace with your key

# Option 2: Or load from .env if you want (optional)
# from dotenv import load_dotenv
# import os
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Safety check
if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_KEY_HERE":
    raise ValueError(
        "❌ GEMINI_API_KEY is missing! Replace 'YOUR_GEMINI_KEY_HERE' with your own key."
    )

print(f"✓ Gemini API key configured (first 5 chars: {GEMINI_API_KEY[:5]}..., length: {len(GEMINI_API_KEY)})")

print("✓ Environment variables loaded")
print(f"✓ Gemini API key configured (starts with {GEMINI_API_KEY[:5]}..., length: {len(GEMINI_API_KEY)})")

# --------------------------
# UTILITY FUNCTIONS
# --------------------------
def get_video_id(youtube_url: str) -> str:
    """
    Extract YouTube video ID from various URL formats.
    """
    regex_patterns = [
        r'(?:v=|\/videos\/|embed\/|\.be\/|shorts\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'youtu\.be\/([a-zA-Z0-9_-]{11})'
    ]
    for pattern in regex_patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    raise ValueError(f"Invalid YouTube URL format: {youtube_url}")

def validate_video_id(video_id: str) -> bool:
    """
    Validate that a video ID has the correct 11-character format.
    """
    return bool(re.match(r'^[a-zA-Z0-9_-]{11}$', video_id))

print("✓ Utility functions defined")

# --------------------------
# YOUTUBE TRANSCRIPT EXTRACTOR TOOL
# --------------------------
@tool("YouTube Transcript Extractor")
def fetch_youtube_transcript(youtube_url: str) -> str:
    """
    Fetches the complete transcript from a YouTube video URL.
    Returns the transcript as a single string.
    Handles errors if captions are missing or video is unavailable.
    """
    try:
        # Extract video ID
        video_id = get_video_id(youtube_url)
        print(f"🔹 Processing video ID: {video_id}")

        # Create API instance
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)

        # Select transcript in preferred language
        transcript = transcript_list.find_transcript([
            'en-CA', 'en-US', 'en-GB', 'en'
        ])

        print(f"   Selected: {transcript.language} ({transcript.language_code})")
        print(f"   Generated: {'Yes' if transcript.is_generated else 'No'}")

        # Fetch transcript data
        fetched_data = transcript.fetch()
        snippets = getattr(fetched_data, 'snippets', fetched_data)

        # Combine all text
        # Combine all text safely (works for both objects and dicts)
        transcript_text = " ".join([
            getattr(s, "text", s.get("text") if isinstance(s, dict) else "")
            for s in snippets
        ])

        print(f"✓ Extracted {len(transcript_text)} characters")
        print(f"✓ {len(list(snippets))} segments, ~{len(transcript_text.split())} words\n")

        return transcript_text

    except TranscriptsDisabled:
        error_msg = "ERROR: Captions/subtitles are disabled."
        print(f"❌ {error_msg}")
        return error_msg
    except VideoUnavailable:
        error_msg = "ERROR: Video unavailable or private."
        print(f"❌ {error_msg}")
        return error_msg
    except NoTranscriptFound:
        error_msg = "ERROR: No transcript found in requested languages."
        print(f"❌ {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

print("✓ YouTube transcript extraction tool created")

# --------------------------
# CONFIGURE GEMINI LLM
# --------------------------
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.3,
)

print("✓ Gemini LLM configured")
print("   Model: gemini-2.5-flash")
print("   Temperature: 0.3 (focused output)")

# --------------------------
# DEFINE AGENTS
# --------------------------
extractor_agent = Agent(
    role="YouTube Transcript Extraction Specialist",
    goal="Extract complete transcripts using the fetch_youtube_transcript tool.",
    backstory="Expert in YouTube transcript extraction and handling errors.",
    tools=[fetch_youtube_transcript],
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False,
)

summarizer_agent = Agent(
    role="Content Analysis and Summarization Expert",
    goal="Create comprehensive summaries from video transcripts.",
    backstory="Senior analyst experienced in summarizing complex content.",
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False,
)

print("✓ Agents created successfully")

# --------------------------
# DEFINE TASKS
# --------------------------
extract_task = Task(
    description=(
        "Extract the complete transcript from the YouTube video at this URL: {youtube_url}\n"
        "Use fetch_youtube_transcript and verify content."
    ),
    expected_output="The full transcript text ready for analysis.",
    agent=extractor_agent,
)

summarize_task = Task(
    description=(
        "Analyze the transcript and create a professional summary.\n"
        "Include Overview, Key Points, Detailed Insights, and Key Takeaways.\n"
        "Length: 250-350 words. Base summary ONLY on transcript."
    ),
    expected_output="Well-structured summary with clear sections and insights.",
    agent=summarizer_agent,
    context=[extract_task],
)

print("✓ Tasks defined successfully")

# --------------------------
# ASSEMBLE CREW
# --------------------------
video_summarizer_crew = Crew(
    agents=[extractor_agent, summarizer_agent],
    tasks=[extract_task, summarize_task],
    process=Process.sequential,
    verbose=True,
)

print("✓ Crew assembled successfully")

# --------------------------
# MAIN EXECUTION
# --------------------------
if __name__ == "__main__":
    print("🚀 YouTube Video Summarizer Agent is ready!")

    youtube_url = "https://youtu.be/9PXluC2FMD0?si=n5kYku3saX8CuKwv" # Replace with any video URL

    try:
        # Run the crew workflow
        result = video_summarizer_crew.kickoff(inputs={'youtube_url': youtube_url})

        # Display results
        print(f"\n{'='*80}")
        print("✅ SUMMARY COMPLETE")
        print(f"{'='*80}\n")
        print(result)
        print(f"\n{'='*80}\n")

    except Exception as e:
        print(f"❌ Error during execution: {e}")
