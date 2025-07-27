# 运行方法

首先请确保你已经安装了 uv，如果没有的话，请按以下页面的要求安装：

https://docs.astral.sh/uv/guides/install-python/

然后在当前目录下，新建一个叫做 .env 的文件，输入以下内容：

```
model_source=ollama
TOOL_LLM_NAME=mistral:latest
OPENROUTER_API_KEY=xxxx
```

xxxx 就是你在 OpenRouter 上配好的 API Key。如果你不用 OpenRouter，那直接改下代码，换个别的 baseUrl 就行了。

确保 uv 已经安装成功后，进入到当前文件所在目录，然后执行以下命令即可启动：

下面的地址“E:\ReAct\tmp”，用你自己的决定地址替换，以后agent帮我们写的代码都放在该目录下，要先确保它的存在。

本项目的目的：验证ReAct格式(thought-act-observation)提示词，在控制大模型行为上的一次试验.本实验通过单独的
题词模板文件(prompt_template.py)提供给大模型。

```bash
uv run agent.py E:\ReAct\tmp
```