from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from rich.console import Console
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from rich.panel import Panel
from rich.markdown import Markdown
from langgraph.graph import StateGraph
from typing import Annotated, Sequence
from langgraph.graph import add_messages
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)

load_dotenv()


class FileReadToolInput(BaseModel):
    file_path: str = Field(description="The absolute path to the file to read")


class FileReadTool(BaseTool):
    name: str = "file_read"
    description: str = (
        "Reads a file designated by the supplied absolute path and returns the content as a string"
    )
    args_schema: type[BaseModel] = FileReadToolInput

    def _run(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class SimpleAgent:
    def __init__(self):
        self.console = Console()
        self.console.print(Panel.fit("[bold green]Hello, world![/bold green]"))

        self.tools = [FileReadTool()]

        self.model = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.3,
        )
        self.model = self.model.bind_tools(self.tools)
        workflow = StateGraph(AgentState)
        workflow.add_node("user_input", self._get_user_input)
        workflow.add_node("model_response", self._get_model_response)
        workflow.add_node("tool_use", self._get_tool_use)

        workflow.set_entry_point("user_input")

        # workflow.add_edge("user_input", "model_response")
        workflow.add_conditional_edges(
            "user_input",
            self._route_user_input,
            {"model_response": "model_response", "end": "__end__"},
        )

        workflow.add_edge("tool_use", "model_response")

        # workflow.add_edge("model_response", "user_input")
        workflow.add_conditional_edges(
            "model_response",
            self._check_tool_use,
            {"tool_use": "tool_use", "user_input": "user_input"},
        )

        self.agent = workflow.compile()
        print(self.agent.get_graph().draw_mermaid)
        self.save_agent_graph(self.agent)

    def save_agent_graph(self, agent_executor, filename="agent_graph.png"):
        try:
            # This returns the image data as bytes
            png_data = agent_executor.get_graph().draw_mermaid_png()
            with open(filename, "wb") as f:
                f.write(png_data)
            print(f"Graph successfully saved to {filename}")
        except Exception as e:
            print(f"Could not save graph: {e}")

    def _check_tool_use(self, state: AgentState) -> AgentState:
        if state.messages[-1].tool_calls:
            return "tool_use"
        return "user_input"

    def _route_user_input(self, state: AgentState) -> str:
        last_message = state.messages[-1].content.lower().strip()
        if last_message in ["exit", "quit"]:
            return "end"
        return "model_response"

    def _get_user_input(self, state: AgentState) -> AgentState:
        self.console.print("[bold blue]User Input[/bold blue]")
        user_input = self.console.input("> ")
        return {"messages": [HumanMessage(content=user_input)]}

    def _get_model_response(self, state: AgentState) -> AgentState:
        messages = [
            SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": "You are a helpful assistant that can read files and answer questions about them.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            ),
            HumanMessage(content=f"Working directory:{os.getcwd()}"),
        ] + state.messages
        response = self.model.invoke(messages)
        print(response)
        if isinstance(response.content, list):
            for item in response.content:
                if item["type"] == "text":
                    self.console.print(
                        Panel.fit(Markdown(item["text"]), title="Assistant")
                    )
                elif item["type"] == "tool_use":
                    self.console.print(
                        Panel.fit(Markdown(item["name"]), title="Tool Use")
                    )
        else:
            self.console.print(Panel.fit(Markdown(response.content), title="Assistant"))

        return {"messages": [response]}

    def _get_tool_use(self, state: AgentState) -> AgentState:
        tools_by_name = {tool.name: tool for tool in self.tools}
        response = []
        for tool_call in state.messages[-1].tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool = tools_by_name[tool_name]
            try:
                tool_result = tool._run(**tool_args)
                response.append(
                    ToolMessage(
                        content=tool_result,
                        name=tool_name,
                        tool_call_id=tool_call["id"],
                    )
                )
                self.console.print(
                    Panel.fit(
                        Markdown("'''" + tool_result + "'''"), title="Tool Result"
                    )
                )
            except Exception as e:
                response.append(
                    ToolMessage(
                        content="Error" + str(e),
                        name=tool_name,
                        tool_call_id=tool_call["id"],
                    )
                )
                self.console.print(
                    Panel.fit(Markdown(str(e)), title="Tool Error", border_style="red")
                )
        return {"messages": response}

    def run(self) -> str:
        self.agent.invoke(
            {"messages": [AIMessage(content="what can I do for you today?")]}
        )


if __name__ == "__main__":
    agent = SimpleAgent()
    agent.run()
