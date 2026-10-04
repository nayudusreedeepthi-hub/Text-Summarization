import re
import nltk

from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize


# -----------------------------------------
# Download required NLTK resources
# -----------------------------------------

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# -----------------------------------------
# Text Summarization Function
# -----------------------------------------

def summarize_text(text, number_of_sentences=3):

    # Split text into sentences
    sentences = sent_tokenize(text)

    if len(sentences) <= number_of_sentences:
        return sentences

    # Find words
    words = re.findall(
        r'\b[a-zA-Z]+\b',
        text.lower()
    )

    # Remove stop words
    filtered_words = [
        word
        for word in words
        if word not in stop_words
        and len(word) > 2
    ]

    # Count word frequency
    word_frequency = Counter(
        filtered_words
    )

    # Normalize frequencies
    if word_frequency:

        maximum_frequency = max(
            word_frequency.values()
        )

        for word in word_frequency:

            word_frequency[word] = (
                word_frequency[word]
                / maximum_frequency
            )

    # -----------------------------------------
    # Calculate sentence scores
    # -----------------------------------------

    sentence_scores = {}

    for sentence in sentences:

        sentence_words = re.findall(
            r'\b[a-zA-Z]+\b',
            sentence.lower()
        )

        score = 0

        for word in sentence_words:

            if word in word_frequency:

                score += word_frequency[word]

        sentence_scores[sentence] = score

    # Select highest scoring sentences
    ranked_sentences = sorted(
        sentence_scores,
        key=sentence_scores.get,
        reverse=True
    )

    selected_sentences = ranked_sentences[
        :number_of_sentences
    ]

    # Keep original order
    summary = [
        sentence
        for sentence in sentences
        if sentence in selected_sentences
    ]

    return summary


# -----------------------------------------
# Main Program
# -----------------------------------------

print("=" * 60)
print("              TEXT SUMMARIZATION SYSTEM")
print("=" * 60)

print("\nEnter your paragraph.")
print("Type END on a new line when finished.\n")


lines = []

while True:

    line = input()

    if line.strip().upper() == "END":
        break

    lines.append(line)


text = " ".join(lines)


# -----------------------------------------
# Check input
# -----------------------------------------

if not text.strip():

    print("\n❌ No text entered.")

else:

    # Ask number of summary sentences
    try:

        number = int(
            input(
                "\nHow many sentences should the summary contain? "
            )
        )

    except ValueError:

        number = 3

    # Generate summary
    summary = summarize_text(
        text,
        number
    )

    # -----------------------------------------
    # Display result
    # -----------------------------------------

    print("\n" + "=" * 60)
    print("                    SUMMARY")
    print("=" * 60)

    for sentence in summary:

        print("\n" + sentence)

    # -----------------------------------------
    # Calculate compression
    # -----------------------------------------

    original_sentences = nltk.sent_tokenize(
        text
    )

    summary_sentences = len(summary)

    compression = (
        1 -
        summary_sentences /
        len(original_sentences)
    ) * 100

    print("\n" + "=" * 60)

    print(
        f"Original sentences: "
        f"{len(original_sentences)}"
    )

    print(
        f"Summary sentences: "
        f"{summary_sentences}"
    )

    print(
        f"Text reduction: "
        f"{compression:.2f}%"
    )

    print("=" * 60)

    print("\n✅ Summarization completed.")