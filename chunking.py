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


def merge_small_sections(sections, max_tokens):
    """
    Merges consecutive small heading-based sections together, up to
    max_tokens, so we don't end up with a huge number of tiny chunks
    (e.g. one short constitutional article per chunk, each only ~50 tokens).

    This runs BEFORE recursive_split(), so recursive_split only ever has
    to worry about sections that are too BIG — never too small.

    Merging is "greedy": we just keep adding the next section to the
    current group as long as it still fits under max_tokens, regardless
    of which heading level each section came from.
    """
    merged_sections = []

    # These track the group of sections we're currently accumulating
    current_texts = []
    current_heading_paths = []
    current_tokens = 0

    def flush_group():
        """Turns whatever we've accumulated so far into one merged section."""
        if not current_texts:
            return
        combined_text = "\n\n".join(current_texts)

        # Keep track of every distinct heading path folded into this chunk,
        # so we can still trace which original sections it covers.
        unique_paths = []
        for path in current_heading_paths:
            if path not in unique_paths:
                unique_paths.append(path)
        combined_heading_path = " | ".join(unique_paths)

        merged_sections.append({
            "heading_path": combined_heading_path,
            "text": combined_text
        })

    for section in sections:
        section_tokens = count_tokens(section["text"])

        if section_tokens > max_tokens:
            # This section is already big enough on its own — flush
            # whatever group we were building, then keep this section
            # standalone (recursive_split will break it down further later).
            flush_group()
            current_texts = []
            current_heading_paths = []
            current_tokens = 0
            merged_sections.append(section)
            continue

        if current_tokens + section_tokens > max_tokens:
            # Adding this section would push the group over the limit —
            # save the current group, start a new one with this section.
            flush_group()
            current_texts = [section["text"]]
            current_heading_paths = [section["heading_path"]]
            current_tokens = section_tokens
        else:
            # Still room — fold this section into the group being built.
            current_texts.append(section["text"])
            current_heading_paths.append(section["heading_path"])
            current_tokens += section_tokens

    # Don't forget the last group being built when the loop ends
    flush_group()

    return merged_sections

def merge_pieces(pieces, max_tokens):
    """
    Greedily re-merges a list of text pieces (e.g. paragraphs) back together
    up to max_tokens, joined by a blank line. This is the same idea as
    merge_small_sections, just applied one level down — otherwise every
    paragraph that individually fits under max_tokens becomes its own tiny
    chunk, which is what was causing the huge, non-uniform chunk count.
    """
    merged = []
    current = []
    current_tokens = 0

    for piece in pieces:
        piece_tokens = count_tokens(piece)

        if current and current_tokens + piece_tokens > max_tokens:
            merged.append("\n\n".join(current))
            current = [piece]
            current_tokens = piece_tokens
        else:
            current.append(piece)
            current_tokens += piece_tokens

    if current:
        merged.append("\n\n".join(current))

    return merged

def recursive_split(text, max_tokens, overlap_tokens):
    if count_tokens(text) <= max_tokens:
        return [text]

    paragraphs = split_by_paragraphs(text)

    if len(paragraphs) > 1:
        # First pass: break down any paragraph that's individually too big
        pieces = []
        for paragraph in paragraphs:
            if count_tokens(paragraph) <= max_tokens:
                pieces.append(paragraph)
            else:
                pieces.extend(recursive_split_sentences(paragraph, max_tokens, overlap_tokens))

        # Second pass: re-merge the small pieces back up toward max_tokens,
        # instead of emitting one chunk per tiny paragraph
        return merge_pieces(pieces, max_tokens)
    else:
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


def build_chunks(document_text, source_name, max_tokens=2000, overlap_tokens=400):
    """
    Main entry point: takes the full document text and returns a list of
    chunk dictionaries, each with text + metadata, ready to be embedded
    and stored in ChromaDB.

    max_tokens: the maximum size of each chunk, in tokens
    overlap_tokens: how much token overlap to apply during fixed-size fallback splitting
    """
    # Step 1: split the document into heading-defined sections
    sections = split_by_headings(document_text)

    # Step 2 (NEW): merge consecutive small sections together up to max_tokens
    sections = merge_small_sections(sections, max_tokens)

    all_chunks = []
    chunk_counter = 0

    for section in sections:
        heading_path = section["heading_path"]
        section_text = section["text"]

        # Step 3: for each section, recursively split it further if it's
        # still bigger than our token budget
        pieces = recursive_split(section_text, max_tokens, overlap_tokens)

        for piece_text in pieces:
            chunk_id = f"chunk_{chunk_counter}"
            chunk_counter += 1

            # CHANGED: heading_path may now contain multiple entries joined
            # by " | " (from merged sections), so we grab the LAST one's
            # LAST level as the short section label.
            last_path = heading_path.split(" | ")[-1]
            section_label = heading_path.split(" > ")[-1]

            chunk = {
                "chunk_id": chunk_id,
                "text": piece_text,
                "metadata": {
                    "chunk_id": chunk_id,
                    "heading_path": heading_path,
                    "section": section_label,
                    "short_section_label": last_path,
                    "token_count": count_tokens(piece_text),
                    "source": source_name
                }
            }
            all_chunks.append(chunk)

    return all_chunks