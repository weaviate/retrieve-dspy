import dspy
import os
import asyncio
from retrieve_dspy.signatures import SummarizeSearchRelevance

class QueryDocumentSummarizer(dspy.Module):
    def __init__(self, verbose: bool = False):
        self.lm = dspy.LM("openai/gpt-4.1-mini", api_key=os.getenv("OPENAI_API_KEY"))
        self.summarizer = dspy.ChainOfThought(SummarizeSearchRelevance)
        dspy.configure(lm=self.lm, track_usage=True)
        self.verbose = verbose

    def forward(self, query: str, document: str) -> str:
        summary = self.summarizer(query=query, passage=document).relevance_summary

        if self.verbose:
            print(f"Summary:\n{summary}")

        return summary
    
    async def aforward(self, query: str, document: str) -> str:
        summary_pred = await self.summarizer.acall(query=query, passage=document)
        summary = summary_pred.relevance_summary
        
        if self.verbose:
            print(f"Summary:\n{summary}")

        return summary
    
async def main():
    import os
    import dspy
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    lm = dspy.LM("openai/gpt-4.1-mini", api_key=openai_api_key)
    dspy.configure(lm=lm, track_usage=True)
    print(f"DSPy configured with: {lm}")

    test_summarizer = QueryDocumentSummarizer(verbose=False)
    
    test_query = "What number did David Ortiz wear when he played for the Boston Red Sox?"
    test_document = "David Ortiz wore the number 34 when he played for the Boston Red Sox. He was a designated hitter and first baseman who played for the team from 2003 to 2016. During his time with the Red Sox, Ortiz helped the team win three World Series championships in 2004, 2007, and 2013. The Red Sox retired his number 34 in 2017, making him the 11th player to receive this honor."
    
    # Test synchronous forward
    print("Testing synchronous forward:")
    summary = test_summarizer.forward(test_query, test_document)
    print(f"Result: {summary}")
    
    # Test asynchronous aforward
    print("\nTesting asynchronous aforward:")
    async_summary = await test_summarizer.aforward(test_query, test_document)
    print(f"Result: {async_summary}")
    
    # Test concurrent summarization with multiple candidate documents
    print("\nTesting concurrent summarization with multiple candidates:")
    candidate_documents = [
        "David Ortiz wore the number 34 when he played for the Boston Red Sox. He was a designated hitter and first baseman who played for the team from 2003 to 2016.",
        "Derek Jeter wore the number 2 for the New York Yankees throughout his career. He was the team's captain and played shortstop from 1995 to 2014.",
        "The Boston Red Sox retired David Ortiz's number 34 in 2017, making him the 11th player to receive this honor. Big Papi was inducted into the Baseball Hall of Fame in 2022.",
        "Ted Williams wore number 9 for the Boston Red Sox and is considered one of the greatest hitters in baseball history. His number was retired by the team.",
        "Manny Ramirez wore number 24 during his time with the Boston Red Sox from 2001 to 2008. He was known for his powerful hitting and helped the team win two World Series titles."
    ]
    
    # Use asyncio.gather to summarize all documents concurrently
    concurrent_tasks = [
        test_summarizer.aforward(test_query, doc) 
        for doc in candidate_documents
    ]
    
    concurrent_summaries = await asyncio.gather(*concurrent_tasks)
    
    print(f"Processed {len(candidate_documents)} documents concurrently:")
    for i, summary in enumerate(concurrent_summaries):
        print(f"Document {i+1} Summary:")
        print(f"\033[92m{summary}\033[0m")

if __name__ == "__main__":
    asyncio.run(main())