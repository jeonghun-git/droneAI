from agents.services.agent import AIAgent

refine_agent = AIAgent(model="deepseek/deepseek-r1-0528", system_prompt="You are a helpful assistant.")

plan_agent = AIAgent(model="deepseek/deepseek-r1-0528", system_prompt="You are a helpful assistant.")

user_agent = AIAgent(model="deepseek/deepseek-r1-0528", system_prompt="You are a helpful assistant.", tools=["get_current_location", "get_current_time"], mcp_use=True)

while True:
    # prompt initialize 
    raw_request = input("입력: ")
    refine_agent.history.append({"role": "user", "content": raw_request})

    # prompt refining
    while True:
        refined_response = refine_agent.text_response(raw_request)
        user_response = input("계획을 수정하시겠습니까? (y/n): ")

        if user_response == "y":
            user_request = input("수정 요청: ")
            refine_agent.history.append({"role": "user", "content": user_request})
            refined_response = refine_agent.text_response(user_request)
            refine_agent.history.append({"role": "assistant", "content": refined_response})

        if user_response == "n":
            final_plan = refined_response
            break
        
    plan_agent.history.append({"role": "user", "content": final_plan})
    plan_response = plan_agent.text_response(final_plan)
    
    user_agent.history.append({"role": "user", "content": plan_response})
    user_agent_response = user_agent.text_response(plan_response)
    
    
