from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import requests
import time
from bs4 import BeautifulSoup
import re

def clean_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text


def search(query: str):
    response = requests.get(f"https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query={query}")

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'lxml')
        
        main_pack = soup.find(id='main_pack')
        
        if main_pack:
            text_only = main_pack.get_text(separator='\n').strip()
            
            cleaned_text = clean_text(text_only)
            cleaned_text = cleaned_text
            with open('final_extracted_text.txt', 'w', encoding='utf-8') as file:
                file.write(cleaned_text)
            return cleaned_text

        else:
            print("Element with id 'main_pack' not found.")
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

def routing(agent: str):
    if agent == "chat":
        return agent
    if agent == "tool":
        return agent
    else:
        return None
    
TOOL_MAPPING = {
    "search": search,
    "routing": routing,
}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for a given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for."
                }
            },
            "required": [
                "query"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}]

ROUTING = [
    {
        "type": "function",
        "function": {
            "name": "routing",
            "description": "Route the user's request to the appropriate agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": ["chat", "tool"],
                        "description": "The agent to route the user's request to.",
                    }
                },
                "required": ["agent"],
                "additionalProperties": False,
            },
            "strict": True
        }
    }
]



class AIAgent:
    history = []
    
    @classmethod
    def add_history(cls, message):
        cls.history.append(message)
    
    @classmethod
    def get_history(cls):
        return cls.history
    
    def __init__(self, model: str = None, tools=False, endpoint: str = "http://192.168.0.17:9898/v1", system_prompt: str = None, **kwargs):
        load_dotenv()
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.client = OpenAI(base_url=endpoint, api_key=os.getenv("OPENROUTER_API_KEY"))
        AIAgent.add_history({"role": "system", "content": system_prompt})
        
   
        
    def text_response(self, user_prompt):
        AIAgent.add_history({"role": "user", "content": user_prompt})
        if self.tools:
            params = dict(
                model=self.model,
                messages=AIAgent.get_history(),
                tools=self.tools,
                stream=True,
            )
        else:
            params = dict(
                model=self.model,
                messages=AIAgent.get_history(),
                stream=True,
            )

        
        response = self.client.chat.completions.create(**params)


        final_response = ""
        function_name = ""
        function_args = ""
        for chunk in response:
            # if chunk.choices[0].delta.content is not None:
            #     final_response += chunk.choices[0].delta.content
            # print(chunk.choices[0].delta.tool_calls[0].function.name)
            # print(chunk.choices[0].delta.content)
            print(chunk.choices[0].delta.content, end="", flush=True)
        #         print(chunk.choices[0].delta.content, end="", flush=True)
        #     else:
        #         if chunk.choices[0].delta.tool_calls[0].function.name is not None:
        #             print(chunk.choices[0].delta.tool_calls[0].function.name, end="", flush=True)
        #             function_name += chunk.choices[0].delta.tool_calls[0].function.name
                    
        #         print(chunk.choices[0].delta.tool_calls[0].function.arguments, end="", flush=True)
        #         function_args += chunk.choices[0].delta.tool_calls[0].function.arguments
        # print()
        
        if final_response:
            AIAgent.add_history({"role": "assistant", "content": final_response})
            
        if function_name:
            return (function_name, function_args)
        else:
            return final_response

            
    def get_tool_response(self, *args):
        tool_name = args[0]
        tool_args = json.loads(args[1])

        tool_result = TOOL_MAPPING[tool_name](**tool_args)
        AIAgent.add_history({
            "role": "assistant",
            "content": tool_result,
        })
        return tool_result
    
if __name__ == "__main__":
    route_agent = AIAgent(model="devstral-small-2505", tools=TOOLS, system_prompt="You must only put the word 'chat' or 'tool'")
    chat_agent = AIAgent(model="devstral-small-2505", tools=None, system_prompt="You are a helpful assistant that can answer questions and help with tasks.")
    tool_agent = AIAgent(model="devstral-small-2505", tools=TOOLS, 
                     system_prompt=f"""You are a ReAct agent that solves problems through systematic reasoning and action.

    INSTRUCTIONS:
    1. Start with "Thought:" to analyze the user's request
    2. Use "Action:" to call appropriate tools
    3. Use "Observation:" to analyze tool results
    4. Repeat steps 1-3 if more information is needed
    5. End with "Final Answer:" when you have sufficient information

    EXAMPLE FORMAT:
    Thought: I need to understand what the user is asking for and determine if I need to search for information.
    Action: [Use search tool if needed]
    Observation: [Analyze the search results]
    Thought: [Decide next steps based on observation]
    Final Answer: [Provide comprehensive response]

    GUIDELINES:
    - Be explicit about your reasoning process
    - Use tools when you need current or specific information
    - Don't make assumptions - verify with searches when uncertain
    - Provide thorough, accurate answers

    Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}""")
        
    while True:
        user_prompt = input("You: ")
        
        if user_prompt == "exit":
            break   
        
        route = route_agent.text_response(user_prompt)
        
        print(f"\route: {route}\n")
        
        if route == "chat":
            chat_agent.text_response(user_prompt)
        elif route == "tool":
            tool_result = tool_agent.get_tool_response(*tool_agent.text_response(user_prompt))
            print(f"\ntool_result: {tool_result}\n")
            chat_agent.text_response(user_prompt)
        
        

