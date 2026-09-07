"""
Tavily search wrapper. Returns a normalized list of results with URL,
title, and extracted page content (Tavily already returns cleaned text,
so no separate scraper is needed).
"""
from tavily import TavilyClient
from config import TAVILY_API_KEY, SEARCH_RESULTS_PER_QUERY

_client = TavilyClient(api_key=TAVILY_API_KEY)


def search(query: str, max_results: int = None, search_depth: str = "advanced", start_date: str = None, end_date: str = None) -> list:
    max_results = max_results or SEARCH_RESULTS_PER_QUERY
    try:
        kwargs = dict(query=query, search_depth=search_depth, max_results=max_results, include_raw_content=False)
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date
        resp = _client.search(**kwargs)
        # resp = _client.search(
        #     query=query,
        #     search_depth=search_depth,
        #     max_results=max_results,
        #     include_raw_content=False,
        # )
    except Exception as e:
        return [{"error": str(e)}]

    results = []
    for r in resp.get("results", []):
        results.append({
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "content": r.get("content", ""),   # Tavily's cleaned excerpt
            "score": r.get("score", 0.0),
            "published_date": r.get("published_date", "")
        })
    return results
