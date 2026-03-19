import os

from google import genai
from google.genai import types
from tavily import TavilyClient

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# google search didn't work, so using tavily instead, which is a wrapper around google search and other search engines.
# It also provides a nice interface for getting the results in a structured format.
# LLM never runs this, only calls the tool, which then runs this function and returns the results to the LLM.

def websearch(query:str) -> str:
    results = []
    response = tavily.search(query=query, num_results=5) # num_results is the number of search results to return
    for r in response["results"]:
        results.append(f"Source: {r['title']} ({r['url']})\n{r['content']}\n---")
    return "\n".join(results)

# Gemini's answer are only as good as what Tavily fetches.

search_tool = types.Tool(
    function_declarations= [
		types.FunctionDeclaration(
			name="websearch", # python function name, used to call the tool from the LLM response
			description="Search the web for relevant information.", # uses this to decide when to call the tool, so make it descriptive and relevant to the task at hand.
			parameters= types.Schema(
       type = types.Type.OBJECT,
	   properties = {
        "query": types.Schema(
			type = types.Type.STRING,
			description = "The search query to find relevant information."
		)
	   },
    	   required = ["query"]
	)
		)
	]
)      

def main() -> None:
	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key:
		raise RuntimeError(
			"Missing GEMINI_API_KEY. Set it in your environment before running this script."
		)

	client = genai.Client(api_key=api_key)
	conversation_history = []
	print("Chatbot ready! Type 'quit' to exit.\n")
 
	try:
		system_prompt = f"""You are a helpful research assistant. When asked about current events or recent information,
		use the websearch tool to find relevant data. Always provide concise and accurate answers based on the information you have or can retrieve."""
		while True:
			user_input = input("You: ")
			if user_input.lower() == "quit":
				print("Exiting chat. Goodbye!")
				break
			conversation_history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
			while True:
				response = client.models.generate_content(
					model="gemini-2.5-flash",
					config = types.GenerateContentConfig(system_instruction= system_prompt, tools = [search_tool]),
					contents=conversation_history,
				)
				if not response.candidates:
					print("No response from model. Ending conversation.")
					break
				candidate = response.candidates[0]
				if candidate.content.parts[0].function_call:
					function_call = candidate.content.parts[0].function_call
					print(f"\nSearching: {function_call.args['query']}\n")
					search_results = websearch(function_call.args["query"])
					conversation_history.append(candidate.content)
					conversation_history.append(types.Content(
						role="user",
						parts=[
							types.Part(
            					function_response=types.FunctionResponse(
									name="web_search",
									response={"result": search_results},
								)
							)
						],
					))
				else:
					final_answer = candidate.content.parts[0].text
					conversation_history.append(types.Content(role="model", parts=[types.Part(text=final_answer)]))
					print(f"Assistant: {final_answer}\n")
					break
				
	except Exception as exc:
		message = str(exc)
		if "RESOURCE_EXHAUSTED" in message or "quota" in message.lower() or "429" in message:
			print("Gemini API quota exceeded (429). Check billing/plan or wait and retry.")
			print("Rate limits: https://ai.google.dev/gemini-api/docs/rate-limits")
			return
		raise

if __name__ == "__main__":
    main()