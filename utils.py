import re

def is_how_to_question(query):
    """Check if a query is a how-to question"""
    how_to_patterns = [
        r"^how (?:do|to|can) (?:i|you|we)",
        r"^how (?:does|is|are|would)",
        r"^what (?:is|are) the steps",
        r"steps to",
        r"guide for",
        r"tutorial for"
    ]
    
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in how_to_patterns)

def extract_cdp_names(query):
    """Extract CDP names mentioned in a query"""
    cdps = []
    query_lower = query.lower()
    
    cdp_variations = {
        "segment": ["segment"],
        "mparticle": ["mparticle", "m particle", "m-particle"],
        "lytics": ["lytics"],
        "zeotap": ["zeotap"]
    }
    
    for cdp, variations in cdp_variations.items():
        if any(variation in query_lower for variation in variations):
            cdps.append(cdp)
    
    return cdps

def is_comparison_question(query):
    """Check if a query is asking for a comparison"""
    comparison_keywords = ["compare", "comparison", "difference", "versus", "vs", "or"]
    query_lower = query.lower()
    
    # Check if query contains comparison keywords
    has_comparison_keyword = any(keyword in query_lower for keyword in comparison_keywords)
    
    # Check if query mentions multiple CDPs
    cdps = extract_cdp_names(query)
    
    return has_comparison_keyword and len(cdps) >= 2

def parse_long_question(query):
    """Parse and extract key information from a long question"""
    # If query is not too long, return as is
    if len(query) < 200:
        return query
    
    # Extract CDP names
    cdps = extract_cdp_names(query)
    
    # Try to extract the core question
    how_to_patterns = [
        r"(how (?:do|to|can) (?:i|you|we) .*?)(?:\?|\.|$)",
        r"(what (?:is|are) the steps to .*?)(?:\?|\.|$)",
        r"(steps to .*?)(?:\?|\.|$)",
        r"(guide for .*?)(?:\?|\.|$)"
    ]
    
    for pattern in how_to_patterns:
        match = re.search(pattern, query.lower())
        if match:
            core_question = match.group(1)
            
            # Add CDP name if not in core question
            if cdps and not any(cdp in core_question.lower() for cdp in cdps):
                core_question += f" in {cdps[0]}"
            
            return core_question + "?"
    
    # If no patterns match, return a truncated version
    return query[:150] + "..."

def format_answer(content, query, cdps=None):
    """Format retrieved content into a user-friendly answer"""
    # Determine if it's a how-to question
    if is_how_to_question(query):
        answer = "## How-to Guide\n\n"
    else:
        answer = "## Information\n\n"
    
    # If it's a comparison
    if cdps and len(cdps) >= 2:
        answer = f"## Comparison: {', '.join(cdps[:-1])} and {cdps[-1]}\n\n"
    
    # Add the main content
    answer += content
    
    return answer

def check_cdp_relevance(query):
    """Check if query is relevant to CDPs"""
    cdp_keywords = [
        "segment", "mparticle", "lytics", "zeotap", "cdp", "customer data",
        "platform", "tracking", "analytics", "audience", "user profile", 
        "integration", "data collection", "source", "destination"
    ]
    
    query_lower = query.lower()
    
    return any(keyword in query_lower for keyword in cdp_keywords)