import pandas as pd
import re

# 1. Path to your transcript file (txt file you saved)
file_path = "../data/raw/transcript.txt"

# 2. Normalize whitespace
text = re.sub(r'\n+', ' ', text)   # Replace line breaks with spaces
text = re.sub(r'\s+', ' ', text)   # Remove multiple spaces

# 3. Split text into sentences using punctuation (. ! ?)
sentences = re.split(r'(?<=[.!?])\s+', text)

# 4. Remove empty or whitespace-only entries
sentences = [s.strip() for s in sentences if s.strip()]

# 5. Create a DataFrame
df_sentences = pd.DataFrame({
    "sentence_id": range(1, len(sentences) + 1),
    "sentence": sentences
})

# 6. Save to CSV
df_sentences.to_csv("../data/processed/podcast_sentences.csv", index=False)

# 7. Preview first rows
df_sentences.head()