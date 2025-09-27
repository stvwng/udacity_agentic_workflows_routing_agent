import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables and initialize OpenAI client

client = OpenAI()
MODEL="gpt-4.1"

# --- Helper Function for API Calls ---
def call_openai(instructions, input, model=MODEL):
    """Simple wrapper for OpenAI API calls."""
    try:
        response = client.responses.create(
            model=model,
            instructions=instructions, # system prompt
            input=input # user prompt
        )
        return response.output_text
    except Exception as e:
        print(f"""Response failed: {e}""")


# --- Agents for Different Retail Tasks ---

def product_researcher_agent(query):
    """Product researcher agent gathers product information."""
    instructions = """You are a product research agent for a retail company. Your task is to provide 
    structured information about products, market trends, and competitor pricing."""
    
    input = f"Research this product thoroughly: {query}"
    return call_openai(instructions, input)


def customer_analyzer_agent(query):
    """Customer analyzer agent processes customer data and feedback."""
    instructions = """You are a customer analysis agent. Your task is to analyze customer feedback, 
    preferences, and purchasing patterns."""
    
    input = f"Analyze customer behavior for: {query}"
    return call_openai(instructions, input)


def pricing_strategist_agent(query, product_data=None, customer_data=None):
    """Pricing strategist agent recommends optimal pricing."""
    instructions = """You are a pricing strategist agent. Your task is to recommend optimal pricing 
    strategies based on product research and customer analysis."""
    
    input = f"""
        Original Pricing Query: {query}
        Product Research Data:
        {product_data}
        Customer Analysis Data:
        {customer_data}
        Based on all the above information, please provide a recommended pricing strategy, suggest an optimal price or price range, and explain your reasoning.
        """
    
    return call_openai(instructions, input)


# --- Routing Agent with LLM-Based Task Determination ---
def routing_agent(query, *args):
    """Routing agent that determines which agent to use based on the query."""
    
    instructions = """You are a helpful AI assistant that categorizes retail-related user queries. Based on the user's query, determine if it is primarily about:
                    * "product research" (e.g., asking for product specs, trends, competitor prices)
                    * "customer analysis" (e.g., asking about customer feedback, preferences, purchase patterns)
                    * "pricing strategy" (e.g., asking for optimal pricing for a product)
                    Respond only with one of these exact phrases: "product research", "customer analysis", or "pricing strategy".
                    """
                    
    input = f"""Classify this query: {query}"""
    
    return call_openai(instructions, input)


# --- Example Usage ---
if __name__ == "__main__":
    # Example queries
    queries = [
        "What are the specifications and current market trends for wireless earbuds?",
        "What do customers think about our premium coffee brand?",
        "What should be the optimal price for our new organic skincare line?"
    ]
    
    # Process each query
    for query in queries:
        print(f"\nQuery: {query}")
        print("\nProcessing...")
        
        classification = routing_agent(query)
        print(f'Classification: {classification}')
        print()
        if classification == "product research":
            print("Product research query")
            print(product_researcher_agent(query))
        elif classification == "customer analysis":
            print("Customer analysis query")
            print(customer_analyzer_agent(query))
        elif classification == "pricing strategy":
            print("Pricing strategy")
            product_data = product_researcher_agent(query)
            print(f'sample product data: {product_data[:20]}')
            customer_data = customer_analyzer_agent(query)
            print(f'sample customer data: {customer_data[:20]}')
            print()
            print(pricing_strategist_agent(query, product_data, customer_data))
        else:
            print(f'Unable to classify this query: {query}')
            
        print("---------------------")
        print()