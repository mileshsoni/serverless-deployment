import os
from openai import OpenAI

# Choose provider: "groq" or "openai"
PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()  # Default to groq

if PROVIDER == "groq":
    BASE_URL = "https://api.groq.com/openai/v1"
    API_KEY = os.getenv("GROQ_API_KEY")
    MODEL = "llama3-70b-8192"
else:
    BASE_URL = "https://api.openai.com/v1"
    API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL = "gpt-3.5-turbo"

if not API_KEY:
    raise ValueError(f"Missing API key for provider: {PROVIDER.upper()}")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

INPUT_FOLDER = "input_files"
OUTPUT_FOLDER = "output_summaries"

def summarize_text(text):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful text summarizer."},
                {"role": "user", "content": f"Summarize this text:\n\n{text}"}
            ],
            temperature=0.5,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Summary could not be generated due to an error."

def summarize_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return summarize_text(text)

def process_files():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".txt"):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, f"summary_{filename}")

            summary = summarize_text_from_file(input_path)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(summary)

            print(f"Summarized: {filename} -> {output_path}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_FOLDER):
        raise FileNotFoundError(f"Input folder '{INPUT_FOLDER}' does not exist.")
    process_files()
