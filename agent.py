import ast
import inspect
import os
import re
from string import Template
from typing import List, Callable, Tuple

import click
from dotenv import load_dotenv
from openai import OpenAI
from langchain_ollama import ChatOllama

import platform

from prompt_template import react_system_prompt_template

load_dotenv()

class ReActAgent:
    def __init__(self, tools: List[Callable], model: str, project_directory: str):
        self.tools = { func.__name__: func for func in tools }
        self.model = model
        self.project_directory = project_directory

        model_source = os.getenv('model_source', 'gpt-4o')
        if model_source == "gpt-4o":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=ReActAgent.get_api_key(),
            )
        else:
            print(f"\n\n💭 TChatOllama\n")
            self.llm = ChatOllama(
                base_url="http://localhost:11434",
                model="mistral:latest",
            )
            



    def run(self, user_input: str):
        messages = [
            {"role": "system", "content": self.render_system_prompt(react_system_prompt_template)},
            {"role": "user", "content": f"<question>{user_input}</question>"}
        ]
        
        while True:
            # 请求模型
            content = self.call_model(messages)
            
            # 提取所有thought-action对
            thought_actions = re.findall(r"<thought>(.*?)</thought>\s*<action>(.*?)</action>", content, re.DOTALL)
            
            # 处理每个thought-action对
            for thought, action in thought_actions:
                print(f"\n\n💭 Thought: {thought.strip()}")
                
                tool_name, args = self.parse_action(action.strip())
                print(f"\n\n🔧 Action: {tool_name}({', '.join(args)})")
                
                # 询问用户确认
                should_continue = input("\n\n是否继续？（Y/N）") if tool_name == "run_terminal_command" else "y"
                if should_continue.lower() != 'y':
                    print("\n\n操作已取消。")
                    return "操作被用户取消"
                
                try:
                    observation = self.tools[tool_name](*args)
                except Exception as e:
                    observation = f"工具执行错误：{str(e)}"
                
                print(f"\n\n🔍 Observation：{observation}")
                messages.append({"role": "user", "content": f"<observation>{observation}</observation>"})
            
            # 检查是否有final_answer
            final_match = re.search(r"<thought>(.*?)</thought>\s*<final_answer>(.*?)</final_answer>", content, re.DOTALL)
            if final_match:
                thought = final_match.group(1)
                final_answer = final_match.group(2)
                print(f"\n\n💭 Thought: {thought.strip()}")
                return final_answer.strip()
            
            # 如果没有找到任何thought-action或final_answer，报错
            if not thought_actions and not final_match:
                raise RuntimeError("模型输出格式不正确，缺少<thought>-<action>对或<final_answer>")


    def get_tool_list(self) -> str:
        """生成工具列表字符串，包含函数签名和简要说明"""
        tool_descriptions = []
        for func in self.tools.values():
            name = func.__name__
            signature = str(inspect.signature(func))
            doc = inspect.getdoc(func)
            tool_descriptions.append(f"- {name}{signature}: {doc}")
        return "\n".join(tool_descriptions)

    def render_system_prompt(self, system_prompt_template: str) -> str:
        """渲染系统提示模板，替换变量"""
        tool_list = self.get_tool_list()
        file_list = ", ".join(
            os.path.abspath(os.path.join(self.project_directory, f))
            for f in os.listdir(self.project_directory)
        )
        return Template(system_prompt_template).substitute(
            operating_system=self.get_operating_system_name(),
            tool_list=tool_list,
            file_list=file_list
        )

    @staticmethod
    def get_api_key() -> str:
        """Load the API key from an environment variable."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("未找到 OPENROUTER_API_KEY 环境变量，请在 .env 文件中设置。")
        return api_key
    
    def convert_messages(self,messages, to_dict=True):
        """
        Convert between message formats:
        - Dict format: [{"role": x, "content": y}, ...]
        - Tuple format: [(x, y), ...]
        
        Args:
            messages: List of messages in either format
            to_dict: If True, converts to dict format, else to tuple format
        
        Returns:
            List of messages in the converted format
        """
        converted = []
        
        for msg in messages:
            if to_dict:
                # Convert from tuple to dict format
                if isinstance(msg, tuple):
                    role, content = msg
                    converted.append({"role": role, "content": content})
                else:
                    converted.append(msg)  # already in dict format
            else:
                # Convert from dict to tuple format
                if isinstance(msg, dict):
                    converted.append((msg["role"], msg["content"]))
                else:
                    converted.append(msg)  # already in tuple format
                    
        return converted

    
    def call_model(self, messages):
        print("\n\n正在请求模型，请稍等...")

        model_source = os.getenv('model_source', 'gpt-4o')
        if model_source == "gpt-4o":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            content = response.choices[0].message.content
            messages.append({"role": "assistant", "content": content})
            return content   
        else:
            tup_messages = self.convert_messages(messages,to_dict=False)
            response = self.llm.invoke(tup_messages)
           
            # Handle response.content based on type
            if isinstance(response.content, str):
                content = response.content
            elif isinstance(response.content, list):
                # Concatenate all string parts (ignore dicts for now)
                content = ''.join(
                    part if isinstance(part, str) else str(part)
                    for part in response.content
                )
            else:
                raise ValueError("Unsupported content type returned from model")

            # Append model reply to the message history
            messages.append({"role": "assistant", "content": content})

            return content  # return just the model's response text



    def parse_action(self, code_str: str) -> Tuple[str, List[str]]:
        match = re.match(r'(\w+)\((.*)\)', code_str, re.DOTALL)
        if not match:
            raise ValueError("Invalid function call syntax")

        func_name = match.group(1)
        args_str = match.group(2).strip()

        # 手动解析参数，特别处理包含多行内容的字符串
        args = []
        current_arg = ""
        in_string = False
        string_char = None
        i = 0
        paren_depth = 0
        
        while i < len(args_str):
            char = args_str[i]
            
            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    string_char = char
                    current_arg += char
                elif char == '(':
                    paren_depth += 1
                    current_arg += char
                elif char == ')':
                    paren_depth -= 1
                    current_arg += char
                elif char == ',' and paren_depth == 0:
                    # 遇到顶层逗号，结束当前参数
                    args.append(self._parse_single_arg(current_arg.strip()))
                    current_arg = ""
                else:
                    current_arg += char
            else:
                current_arg += char
                if char == string_char and (i == 0 or args_str[i-1] != '\\'):
                    in_string = False
                    string_char = None
            
            i += 1
        
        # 添加最后一个参数
        if current_arg.strip():
            args.append(self._parse_single_arg(current_arg.strip()))
        
        return func_name, args
    
    def _parse_single_arg(self, arg_str: str):
        """解析单个参数"""
        arg_str = arg_str.strip()
        
        # 如果是字符串字面量
        if (arg_str.startswith('"') and arg_str.endswith('"')) or \
           (arg_str.startswith("'") and arg_str.endswith("'")):
            # 移除外层引号并处理转义字符
            inner_str = arg_str[1:-1]
            # 处理常见的转义字符
            inner_str = inner_str.replace('\\"', '"').replace("\\'", "'")
            inner_str = inner_str.replace('\\n', '\n').replace('\\t', '\t')
            inner_str = inner_str.replace('\\r', '\r').replace('\\\\', '\\')
            return inner_str
        
        # 尝试使用 ast.literal_eval 解析其他类型
        try:
            return ast.literal_eval(arg_str)
        except (SyntaxError, ValueError):
            # 如果解析失败，返回原始字符串
            return arg_str

    def get_operating_system_name(self):
        os_map = {
            "Darwin": "macOS",
            "Windows": "Windows",
            "Linux": "Linux"
        }

        return os_map.get(platform.system(), "Unknown")
        
# react_agent.py
def react_agent_process(user_input: str) -> str:
    # Simulate intelligent response
    return f"你说的是：{user_input}，这是我根据你的语音做出的回应。"



def read_file(file_path):
    """用于读取文件内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def write_to_file(file_path, content):
    """将指定内容写入指定文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content.replace("\\n", "\n"))
    return "写入成功"

def run_terminal_command(command):
    """用于执行终端命令"""
    import subprocess
    run_result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return "执行成功" if run_result.returncode == 0 else run_result.stderr

@click.command()
@click.argument('project_directory',
                type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('task', required=False)  # Make task optional
def main(project_directory, task):
    project_dir = os.path.abspath(project_directory)

    tools = [read_file, write_to_file, run_terminal_command]
    model_name=os.getenv("TOOL_LLM_NAME", "mistral:latest")
    agent = ReActAgent(tools=tools, model=model_name, project_directory=project_dir)

    # If task is provided via command line, use it; otherwise prompt for input
    if not task:
        task = input("请输入任务：")
    final_answer = agent.run(task)
    
    print(f"\n\n✅ Final Answer：{final_answer}")

if __name__ == "__main__":
    main()
