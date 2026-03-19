import os

from google import genai
from google.genai import types
from googlesearch import search


def websearch(query:str) -> str:
    results = []
    for url in search(query, num_results=5):
        results.append(url)
    return "\n".join(results)

search_tool = types.Tool(
    function_declarations= [
		types.FunctionDeclaration(
			name="websearch",
			description="Search the web for relevant information.",
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
		system_prompt = "You are a helpful research assistant that gives concise, structured answers."
		while True:
			user_input = input("You: ")
			if user_input.lower() == "quit":
				print("Exiting chat. Goodbye!")
				break
			conversation_history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
			response = client.models.generate_content(
				model="gemini-2.5-flash-lite",
				config = types.GenerateContentConfig(system_instruction= system_prompt),
				contents=conversation_history,
			)
			assistant_reply = response.text
			conversation_history.append(types.Content(role="model", parts=[types.Part(text=assistant_reply)]))
			print(f"Agent: {assistant_reply}\n")
	except Exception as exc:
		message = str(exc)
		if "RESOURCE_EXHAUSTED" in message or "quota" in message.lower() or "429" in message:
			print("Gemini API quota exceeded (429). Check billing/plan or wait and retry.")
			print("Rate limits: https://ai.google.dev/gemini-api/docs/rate-limits")
			return
		raise

if __name__ == "__main__":
    main()