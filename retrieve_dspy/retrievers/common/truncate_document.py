def truncate_document(doc: str, max_words: int = 500) -> str:
    """
    Truncate a document to a maximum number of words.
    
    Args:
        doc: The document text to truncate
        max_words: Maximum number of words to keep (default: 500)
        
    Returns:
        Truncated document string
    """
    words = doc.split(" ")
    if len(words) <= max_words:
        return doc
    return " ".join(words[:max_words])