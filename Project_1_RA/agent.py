import os
import json

from google import genai
from google.genai import types
from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()

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
		system_prompt = f"""You are a helpful research assistant. When asked about any topic,
		use the websearch tool to find current information.
  Always respond in this exact JSON format and nothing else:
  {{
	  "summary" : "2-3 sentence overview of the topic",
	  "key_facts": ["fact 1", "fact 2", "fact 3"],
	  "sources": ["domain1.com", "domain2.com"],
	  "confidence": "high | medium | low"
  }}
  Do not include any text outside the JSON. No preamble, no explanations!
  """
		while True:
			user_input = input("You: ")
			if user_input.lower() == "quit":
				print("Exiting chat. Goodbye!")
				break
			conversation_history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
			while True:
				response = client.models.generate_content(
					model="gemini-2.5-flash-lite",
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
					final_answer = candidate.content.parts[0].text or ""
					conversation_history.append(types.Content(role="model", parts=[types.Part(text=final_answer)]))
					cleaned = final_answer.strip()
					if cleaned.startswith("```"):
						cleaned = cleaned.split("\n", 1)[1] 
						cleaned = cleaned.rsplit("```", 1)[0].strip()
    # We cleaned final answer to ensure it's just the JSON, in case the model included any formatting. 
    # This is common when models try to output code or structured data.
    
					try:
						parsed_answer = json.loads(cleaned)
						print("\n --- Research Results ---")
						print(f"\nSummary: {parsed_answer['summary']}")
						print(f"\nKey Facts:")
						for fact in parsed_answer["key_facts"]:
							print(f" - {fact}")
						print(f"\nSources: {', '.join(parsed_answer['sources'])}")
						print(f"Confidence: {parsed_answer['confidence']}\n")
					except json.JSONDecodeError:
						print("Failed to parse model response as JSON. Here's the raw response:")
						print(f"\nAgent: {final_answer}\n")
					break
 
	# The above try - except block is crucial because it ensures that if the model's 
    # response isn't perfectly formatted JSON (which can happen), we still handle it gracefully and 
    # provide feedback to the user, rather than crashing the program.
    
	except Exception as exc:
		message = str(exc)
		if "RESOURCE_EXHAUSTED" in message or "quota" in message.lower() or "429" in message:
			print("Gemini API quota exceeded (429). Check billing/plan or wait and retry.")
			print("Rate limits: https://ai.google.dev/gemini-api/docs/rate-limits")
			return
		raise

if __name__ == "__main__":
    main()