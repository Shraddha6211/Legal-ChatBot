# chunking.py
# This module implements a recursive, Markdown-aware chunker.
# It replaces the fixed-size chunker from ingest.py, while leaving
# everything else in the pipeline (embeddings, ChromaDB, retrieval,
# generation) completely untouched.

import re
import tiktoken

# We use the same tokenizer OpenAI's models use, so our "token count"
# matches what the model actually sees. "cl100k_base" is the encoding
# used by gpt-4o-mini, text-embedding-3-small, and most current OpenAI models.
tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    """
    Returns how many tokens a piece of text would consume.
    We use this instead of len(text) (character count) because
    token count is what actually matters for model context limits.
    """
    tokens = tokenizer.encode(text)
    return len(tokens)


def split_by_headings(text):
    """
    Splits a Markdown document into sections based on heading lines
    (#, ##, ###, ####), and tracks the heading path leading to each section.

    Returns a list of dicts, one per section:
    {
        "heading_path": "Intro > Setup > Prerequisites",
        "text": "the raw text content under this heading (not including subheadings' own text is fine, we keep it simple and include everything until the next heading of same-or-higher level)"
    }

    We keep this deliberately simple: we walk through the document line
    by line, and whenever we see a heading line, we update our "current
    path" of headings and start a new section.
    """
    lines = text.split("\n")

    sections = []
    # current_path tracks the active heading at each level, e.g.
    # {1: "Introduction", 2: "Setup", 3: "Prerequisites"}
    current_path = {}
    current_section_lines = []

    # Regex to detect a markdown heading line and capture its level (#, ##, ###, ####)
    # and its text. Example match: "## Setup" -> level=2, title="Setup"
    heading_pattern = re.compile(r"^(#{1,4})\s+(.*)")

    def flush_section():
        """Saves whatever text we've accumulated so far as one section."""
        section_text = "\n".join(current_section_lines).strip()
        if section_text:  # don't save empty sections
            # Build a readable heading path string from whatever levels are active,
            # e.g. "Introduction > Setup > Prerequisites"
            path_parts = [current_path[level] for level in sorted(current_path.keys())]
            heading_path = " > ".join(path_parts) if path_parts else "Untitled"
            sections.append({
                "heading_path": heading_path,
                "text": section_text
            })

    for line in lines:
        match = heading_pattern.match(line)

        if match:
            # We hit a new heading. First, save the section we were building.
            flush_section()
            current_section_lines = []

            heading_level = len(match.group(1))   # number of '#' characters
            heading_title = match.group(2).strip()

            # A new heading at level N replaces the title at level N,
            # and clears out any deeper levels (e.g. a new H2 clears any H3/H4
            # that were nested under the previous H2).
            current_path[heading_level] = heading_title
            deeper_levels = [lvl for lvl in current_path if lvl > heading_level]
            for lvl in deeper_levels:
                del current_path[lvl]

            # Include the heading line itself in the section content,
            # so the chunk text still shows its own heading when read.
            current_section_lines.append(line)
        else:
            current_section_lines.append(line)

    # Don't forget the last section after the loop ends
    flush_section()

    return sections


def split_by_paragraphs(text):
    """
    Splits text into paragraphs, separated by one or more blank lines.
    This is our fallback when a heading-defined section is still too big.
    """
    # \n\s*\n matches a blank line (possibly with whitespace on it)
    paragraphs = re.split(r"\n\s*\n", text)
    # Remove empty strings that can result from splitting
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def split_by_sentences(text):
    """
    Splits text into sentences using simple punctuation-based rules.
    This is a basic approach (not perfect — e.g. "Mr. Smith" would
    incorrectly split), but it's good enough for our learning purposes
    and avoids adding a heavy NLP dependency like spaCy or NLTK.
    """
    # Split after '.', '!', or '?' followed by a space or newline
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def split_by_fixed_size(text, max_tokens, overlap_tokens):
    """
    Last-resort splitter: breaks text into fixed-size pieces by TOKEN count
    (not character count, unlike our old chunker). Used only when a sentence
    itself is still too long, which should be rare.
    """
    tokens = tokenizer.encode(text)
    pieces = []
    start = 0

    while start < len(tokens):
        end = start + max_tokens
        token_slice = tokens[start:end]
        # Convert the token slice back into readable text
        piece_text = tokenizer.decode(token_slice)
        pieces.append(piece_text)
        start = end - overlap_tokens

    return pieces


def recursive_split(text, max_tokens, overlap_tokens):
    """
    The core recursive algorithm described in the explanation above.

    Tries splitters in priority order:
    paragraphs -> sentences -> fixed-size token split.

    (Heading-level splitting happens separately, BEFORE this function is
    called, in build_chunks() below — because headings need special
    metadata handling, not just plain text splitting.)

    Returns a list of text pieces, each within the max_tokens limit
    (except in rare edge cases where a single "sentence" itself exceeds
    max_tokens even after fixed-size splitting attempts... which
    split_by_fixed_size handles directly).
    """
    # Base case: this piece already fits within our token budget
    if count_tokens(text) <= max_tokens:
        return [text]

    # Try splitting by paragraphs first
    paragraphs = split_by_paragraphs(text)

    if len(paragraphs) > 1:
        # Paragraph splitting actually did something (found more than 1 piece)
        result = []
        for paragraph in paragraphs:
            if count_tokens(paragraph) <= max_tokens:
                result.append(paragraph)
            else:
                # This paragraph is still too big — recurse down to sentence level
                result.extend(recursive_split_sentences(paragraph, max_tokens, overlap_tokens))
        return result
    else:
        # No paragraph breaks found in this text — go straight to sentence level
        return recursive_split_sentences(text, max_tokens, overlap_tokens)


def recursive_split_sentences(text, max_tokens, overlap_tokens):
    """
    Second tier of recursion: try splitting by sentences.
    If a single sentence is still too big, fall back to fixed-size token splitting.
    """
    sentences = split_by_sentences(text)

    if len(sentences) <= 1:
        # No sentence boundaries found (or just one giant sentence) —
        # last resort: fixed-size token splitting
        return split_by_fixed_size(text, max_tokens, overlap_tokens)

    # Group sentences together into chunks up to max_tokens,
    # instead of making every single sentence its own chunk.
    # This keeps chunks reasonably sized rather than tiny.
    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        if sentence_tokens > max_tokens:
            # A single sentence alone is bigger than our limit — rare, but handle it.
            # First, flush whatever we were building.
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_chunk_tokens = 0
            # Then split this oversized sentence with fixed-size splitting.
            chunks.extend(split_by_fixed_size(sentence, max_tokens, overlap_tokens))
            continue

        if current_chunk_tokens + sentence_tokens > max_tokens:
            # Adding this sentence would exceed our limit — save current chunk,
            # start a new one with this sentence.
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentence]
            current_chunk_tokens = sentence_tokens
        else:
            # Still room — add this sentence to the current chunk being built.
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sentence_tokens

    # Don't forget the last chunk being built when the loop ends
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks


def build_chunks(document_text, source_name, max_tokens=300, overlap_tokens=30):
    """
    Main entry point: takes the full document text and returns a list of
    chunk dictionaries, each with text + metadata, ready to be embedded
    and stored in ChromaDB.

    max_tokens: the maximum size of each chunk, in tokens
    overlap_tokens: how much token overlap to apply during fixed-size fallback splitting
    """
    # Step 1: split the document into heading-defined sections
    sections = split_by_headings(document_text)

    all_chunks = []
    chunk_counter = 0

    for section in sections:
        heading_path = section["heading_path"]
        section_text = section["text"]

        # Step 2: for each section, recursively split it further if it's
        # still bigger than our token budget
        pieces = recursive_split(section_text, max_tokens, overlap_tokens)

        for piece_text in pieces:
            chunk_id = f"chunk_{chunk_counter}"
            chunk_counter += 1

            # The last part of the heading path (e.g. "Prerequisites" out of
            # "Introduction > Setup > Prerequisites") is a convenient short
            # label for which specific section this chunk belongs to.
            section_label = heading_path.split(" > ")[-1]

            chunk = {
                "chunk_id": chunk_id,
                "text": piece_text,
                "metadata": {
                    "chunk_id": chunk_id,
                    "heading_path": heading_path,
                    "section": section_label,
                    "token_count": count_tokens(piece_text),
                    "source": source_name
                }
            }
            all_chunks.append(chunk)

    return all_chunks