import query_agent_benchmarking

print("\033[92m" + f"query_agent_benchmarking version: {query_agent_benchmarking.__version__}" + "\033[0m")

query_agent_benchmarking.run_search_eval(
    dataset="bright/biology",
    agent_name="external_service",
    external_service_host="http://localhost:8000/search",
)