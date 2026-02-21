from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from rich.console import Console
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from rich.panel import Panel
from rich.markdown import Markdown
from langgraph.graph import StateGraph
from typing import Annotated, Sequence, TypedDict, Literal
from langgraph.graph import add_messages
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)
import base64
import music_helper

load_dotenv()


from pydantic import BaseModel, Field


class XMLPathInput(BaseModel):
    xml_path: str = Field(description="The path to the MusicXML file.")


class VisionTranscribeInput(BaseModel):
    image_path: str = Field(
        description="Path to the music image to be visually transcribed by the model's vision."
    )


class UpdateXMLInput(BaseModel):
    xml_path: str = Field(description="The path to the file to update.")
    corrected_notes_text: str = Field(
        description="The corrected notes in format 'G4:quarter, A4:half'."
    )


class ListFilesInput(BaseModel):
    pattern: str = Field(
        default="*",
        description="Glob pattern to filter files (e.g., '*.musicxml' or '*.png').",
    )


class ClearHistoryInput(BaseModel):
    confirm: bool = Field(description="Must be True to clear the conversation history.")


class VisionTranscribeTool(BaseTool):
    name: str = "vision_literal_transcription"
    description: str = (
        "Analyzes a music image, generates a literal transcription, and saves it as a MusicXML file."
    )
    args_schema: type[BaseModel] = VisionTranscribeInput

    # We'll pass the model in so the tool can use its 'eyes'
    model: any = None

    def _run(self, image_path: str):
        error_log = ""

        # --- TIER 1: OEMR (Traditional OMR) ---
        try:
            # Attempting the specialized music recognition first
            return music_helper.run_oemer_with_updates(image_path)
        except Exception as e:
            error_log = f"OMR Error: {str(e)}"

        # --- TIER 2: CLAUDE VISION FALLBACK ---
        try:
            if self.model is None:
                return "Error: Vision model not properly initialized in tool."

            with open(image_path, "rb") as f:
                base64_img = base64.b64encode(f.read()).decode("utf-8")

            prompt = (
                "Transcribe this music staff. Output ONLY the notes in the format: "
                "Note:Duration, Note:Duration (e.g., G4:quarter, A4:half). "
                "Do not include any other text."
            )

            # Using the model's native invoke for multimodal
            vision_msg = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_img}"},
                    },
                ]
            )

            response = self.model.invoke([vision_msg])
            notes_text = response.content.strip()

            xml_path = image_path.rsplit(".", 1)[0] + ".musicxml"
            result_log = music_helper.update_musicxml(xml_path, notes_text)

            return f"OMR failed: {error_log}, but Vision Fallback succeeded!\nLog: {result_log}"

        except Exception as e:
            error_log += f"Vision Fallback Error: {str(e)}"

        # --- TIER 3: FINAL CUMULATIVE FAILURE ---
        return "Total Transcription Failure. Details:\n" + error_log


class Review_Process_and_Play(BaseTool):
    name: str = "Review_Process_and_Play"
    description: str = (
        " Review the transcribed music sheet by:"
        " Creating a PNG visual representation of the MusicXML,"
        " Playing the music using violin mp3 library,"
        " Returning a text list of all notes found in a MusicXML file."
    )
    args_schema: type[BaseModel] = XMLPathInput

    def _run(self, xml_path: str):
        return music_helper.process_and_play(xml_path)


class UpdateXMLTool(BaseTool):
    name: str = "update_musicxml_from_text"
    description: str = "Updates the MusicXML file with corrected note data."
    args_schema: type[BaseModel] = UpdateXMLInput

    def _run(self, xml_path: str, corrected_notes_text: str):
        return music_helper.update_musicxml(xml_path, corrected_notes_text)


class ListFilesTool(BaseTool):
    name: str = "list_workspace_files"
    description: str = "Lists music-related files in the current working directory."
    args_schema: type[BaseModel] = ListFilesInput

    def _run(self, pattern: str = "*"):
        import glob

        files = glob.glob(pattern)
        if not files:
            return "No files found matching that pattern."
        return "\n".join(files)


class ClearHistoryTool(BaseTool):
    name: str = "clear_conversation_history"
    description: str = (
        "Clears the current conversation history to start fresh. Use this if the context is cluttered."
    )
    args_schema: type[BaseModel] = ClearHistoryInput

    def _run(self, confirm: bool = False):
        if confirm:
            return "History cleared. How can I help you start fresh?"
        return "Clear history cancelled."


class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class MusicalAgent:
    def __init__(self):
        self.console = Console()
        self.console.print(
            Panel.fit("[bold green]Hello, How Can I help you today![/bold green]")
        )

        self.model = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.3,
        )

        self.tools = [
            VisionTranscribeTool(model=self.model),
            Review_Process_and_Play(),
            UpdateXMLTool(),
            ListFilesTool(),
            ClearHistoryTool(),
        ]

        self.model = self.model.bind_tools(self.tools)
        workflow = StateGraph(AgentState)
        workflow.add_node("user_input", self._get_user_input)
        workflow.add_node("model_response", self._get_model_response)
        workflow.add_node("transcribe", self._get_transcribe)
        workflow.add_node("update", self._update_musicxml)
        workflow.add_node("review", self._review)
        workflow.add_node("list_files", self._list_files)
        workflow.add_node("clear", self._clear_history)

        workflow.set_entry_point("user_input")

        # workflow.add_edge("user_input", "model_response")
        workflow.add_conditional_edges(
            "user_input",
            self._route_user_input,
            {"model_response": "model_response", "end": "__end__"},
        )

        workflow.add_conditional_edges(
            "model_response",
            self._check_tool_use,  # Update this to return "transcribe", "update", or "review"
            {
                "transcribe": "transcribe",
                "update": "update",
                "review": "review",
                "list_files": "list_files",
                "user_input": "user_input",
                "clear": "clear",
            },
        )

        workflow.add_edge("transcribe", "model_response")
        workflow.add_edge("update", "model_response")
        workflow.add_edge("review", "model_response")
        workflow.add_edge("list_files", "model_response")
        workflow.add_edge("clear", "user_input")

        self.agent = workflow.compile()
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

    def _check_tool_use(self, state: AgentState) -> str:
        last_msg = state.messages[-1]
        if not last_msg.tool_calls:
            return "user_input"
        tool_name = last_msg.tool_calls[0]["name"]
        mapping = {
            "vision_literal_transcription": "transcribe",
            "update_musicxml_from_text": "update",
            "Review_Process_and_Play": "review",
            "list_workspace_files": "list_files",
            "clear_conversation_history": "clear",
        }
        return mapping.get(tool_name, "user_input")

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

        for msg in state.messages:
            if isinstance(msg, ToolMessage):
                # We only want to print the latest tool results to avoid duplicates
                # You can check if it's the very last message or just print the content
                self.console.print(
                    Panel.fit(
                        Markdown(f"**Tool Output ({msg.name}):**\n\n{msg.content}"),
                        title="System Execution",
                        border_style="yellow",
                    )
                )

        messages = [
            SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": "You are a musical agent that help the user to create and correct trasncribed music sheets images by: "
                        "Showing the user the transcription visualization and playing the music, then ask user for corrections"
                        "Update the transcription based on user input, then again visualize and play."
                        "Repeat until user said this is correct."
                        "show the file names in the workspace to the user if they asked",
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            ),
            HumanMessage(content=f"Working directory:{os.getcwd()}"),
        ] + state.messages
        response = self.model.invoke(messages)
        # print(response)
        if isinstance(response.content, list):
            for item in response.content:
                if item["type"] == "text":
                    self.console.print(
                        Panel.fit(Markdown(item["text"]), title="Assistant")
                    )
                elif item["type"] == "tool_use":
                    self.console.print(
                        Panel.fit(
                            Markdown(item["name"] + " " + str(item["input"])),
                            title="Tool Use",
                        )
                    )
        else:
            self.console.print(Panel.fit(Markdown(response.content), title="Assistant"))

        return {"messages": [response]}

    # --- New Nodes ---

    def _get_transcribe(self, state: AgentState) -> dict:
        tool_call = state.messages[-1].tool_calls[0]
        image_path = tool_call["args"]["image_path"]

        # Look up the tool from the list you initialized in __init__
        # This tool ALREADY has self.model attached to it
        tools_by_name = {tool.name: tool for tool in self.tools}
        tool = tools_by_name["vision_literal_transcription"]

        result = tool._run(image_path)
        return {
            "messages": [
                ToolMessage(
                    content=result, tool_call_id=tool_call["id"], name=tool.name
                )
            ]
        }

    def _update_musicxml(self, state: AgentState) -> dict:
        """Node for updating MusicXML from text."""
        tool_call = state.messages[-1].tool_calls[0]
        args = tool_call["args"]

        tool = UpdateXMLTool()
        result = tool._run(
            xml_path=args["xml_path"], corrected_notes_text=args["corrected_notes_text"]
        )

        return {
            "messages": [
                ToolMessage(
                    content=result, tool_call_id=tool_call["id"], name=tool.name
                )
            ]
        }

    def _review(self, state: AgentState) -> dict:
        """Node for visualization and playback."""
        # This handles cases from the model OR following a transcribe/update
        # Find the last AI message that actually contained tool calls
        last_ai_msg = next(
            (
                m
                for m in reversed(state.messages)
                if isinstance(m, AIMessage) and m.tool_calls
            ),
            None,
        )

        # print(str(last_ai_msg))
        if last_ai_msg:
            # Get the path from the tool call arguments
            # If we came from 'update', the path is in 'update_musicxml_from_text'
            # If we came from 'transcribe', it might be different.
            # Let's just grab the path from whatever the latest call was.
            xml_path = last_ai_msg.tool_calls[0]["args"].get("xml_path")

            # Fallback if image_path was used instead of xml_path (for transcription)
            if not xml_path and "image_path" in last_ai_msg.tool_calls[0]["args"]:
                img_path = last_ai_msg.tool_calls[0]["args"]["image_path"]
                xml_path = img_path.rsplit(".", 1)[0] + ".musicxml"
        else:
            return {
                "messages": [
                    ToolMessage(
                        content="Error: No file path found to review.",
                        tool_call_id="manual",
                        name="review",
                    )
                ]
            }

        # tool_call = state.messages[-1].tool_calls[0]
        # xml_path = tool_call["args"]["xml_path"]

        tools_by_name = {tool.name: tool for tool in self.tools}
        tool = tools_by_name["Review_Process_and_Play"]

        result = tool._run(xml_path)

        return {
            "messages": [
                ToolMessage(
                    content=result,
                    tool_call_id=last_ai_msg.tool_calls[0]["id"],
                    name=tool.name,
                )
            ]
        }

    def _list_files(self, state: AgentState) -> dict:
        tool_call = state.messages[-1].tool_calls[0]
        pattern = tool_call["args"].get("pattern", "*")

        tool = ListFilesTool()
        result = tool._run(pattern)

        return {
            "messages": [
                ToolMessage(
                    content=result, tool_call_id=tool_call["id"], name=tool.name
                )
            ]
        }

    def _clear_history(self, state: AgentState) -> dict:
        # We return a state that effectively resets the 'messages' list
        # In LangGraph, if you want to wipe history, you usually send a
        # specific command or filter the state.
        return {
            "messages": [SystemMessage(content="Conversation reset by user request.")]
        }

    def run(self) -> str:
        self.agent.invoke(
            {"messages": [AIMessage(content="what can I do for you today?")]}
        )


if __name__ == "__main__":
    agent = MusicalAgent()
    agent.run()
