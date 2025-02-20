import asyncio
import yaml
import os
from crewai import Crew, Agent, Task, Process
from dotenv import load_dotenv
from crewai_tools import SerperDevTool  # Import available tools

load_dotenv()  # Load environment variables
 
class Orchestrator:
    def __init__(self):
        """Initialize the Orchestrator by loading agent and task definitions."""
        package_dir = os.path.dirname(os.path.abspath(__file__))

        # Load agents from YAML
        with open(os.path.join(package_dir, 'config', 'agents.yaml'), 'r') as f:
            self.agents_config = yaml.safe_load(f)

        # Load tasks from YAML
        with open(os.path.join(package_dir, 'config', 'tasks.yaml'), 'r') as f:
            self.tasks_config = yaml.safe_load(f)

        # Dynamically register agents and tasks
        self.agents = self.create_agents()
        self.tasks = self.create_tasks()

    def create_agents(self):
        """Dynamically create agents from YAML definitions."""
        agents = {}
        for agent_name, agent_data in self.agents_config.items():
            tools = self.create_tools(agent_data.get('tools', []))

            agent = Agent(
                role=agent_data['role'],
                goal=agent_data['goal'],
                backstory=agent_data['backstory'],
                llm=agent_data.get('llm', "google/gemini-2.0-flash-exp:free"),  # Set Gemini as default
                tools=tools,  # Pass actual tool instances
                verbose=True
            )
            agents[agent_name] = agent  # Store agents in a dictionary
        return agents

    def create_tools(self, tool_list):
        """Convert YAML-defined tools into actual CrewAI tool instances."""
        tool_mapping = {
            "search": SerperDevTool(),  # Web search tool
        }

        tools = []
        for tool in tool_list:
            if tool["type"] in tool_mapping:
                tools.append(tool_mapping[tool["type"]])  # Convert name to instance
            else:
                print(f"⚠️ Warning: Tool {tool['type']} is not recognized and will be ignored.")
        return tools

    def create_tasks(self):
        """Dynamically create tasks from YAML definitions and assign them to appropriate agents."""
        tasks = []
        for task_name, task_data in self.tasks_config.items():
            agent_key = task_data.get('assigned_agent')  # Get the agent name from the YAML
            agent = self.agents.get(agent_key)  # Retrieve the corresponding agent

            if agent:
                task = Task(
                    description=task_data['description'],
                    expected_output=task_data.get('expected_output', ''),
                    human_input=task_data.get('human_input', False),
                    max_iterations=task_data.get('max_iterations', 3),
                    agent=agent  # Assign the agent to the task
                )
                tasks.append(task)
            else:
                print(f"⚠️ Warning: No valid agent assigned for task '{task_name}', skipping.")
        return tasks

    async def get_user_input(self):
        """Prompt the user for input."""
        print("\n🤖 Welcome to the AI Orchestrator!")
        return input("What would you like to do today? ")

    async def run(self):
        """Main function to get user input, plan, and execute dynamically selected agents."""
        while True:
            user_request = await self.get_user_input()
            if user_request.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break


            # Assemble the Crew using CrewAI’s built-in planning feature
            my_crew = Crew(
                agents=list(self.agents.values()),  # Convert dict to list
                tasks=self.tasks,
                process=Process.sequential,  # Switch to parallel if needed
                planning=True,
                planning_llm="openrouter/google/gemini-2.0-flash-exp:free"  # Set Gemini for planning
            )

            print("Tasks in my_crew:", my_crew.tasks)
            # Run the planned execution
            try:
                result = my_crew.kickoff()
            except Exception as e:
                print("Error during execution:", str(e))

            print("\n🎯 Execution Result:")
            print(result)


if __name__ == "__main__":
    orchestrator = Orchestrator()
    asyncio.run(orchestrator.run())
