import inspect

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from zhipuai import ZhipuAI
import util
import configs.model_config
from langchain_openai import ChatOpenAI

@util.timer_decorator
def ChatUseZhipuAISKD(content):
    client = ZhipuAI(api_key=configs.model_config.API_KEY)  # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": content},
        ],
    )
    # json_data=jsonpickle.encode(response)
    respContent=response.choices[0].message.content
    res=f"[ChatUseZhipuAISKD]\n content:{content}\n response: {respContent}"
    print(res)

@util.timer_decorator
def ChatUseLangChainSDKV0(content):
    # llm = OpenAI(api_key=configs.model_config.API_KEY)
    chat40 = ChatOpenAI(
        model_name="glm-4",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_key=configs.model_config.API_KEY,
    )


    response = chat40.invoke(content)

    respContent = response.content
    res = f"[ChatUseLangChainSDKV0]\n content:{content}\n response: {respContent}"

    print(res)

    # res=f"[ChatUseLangChainSDK]\n content:{content}\n response: {str(response)}"
    # print(res)


@util.timer_decorator
def ChatUseLangChainSDKV1(question):
    # 创建LLM
    llm = ChatOpenAI(
        model_name="glm-4",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_key=configs.model_config.API_KEY,
    )

    # 创建Prompt
    prompt = ChatPromptTemplate.from_template("{question}")

    # 创建输出解析器
    output_parser = StrOutputParser()

    # 调用LLM
    message = prompt.invoke({'question': question})
    response = llm.invoke(message)
    answer = output_parser.invoke(response)

    res = f"[ChatUseLangChainSDKV0]\n question:{question}\n answer: {answer}"
    print(res)

#with chain
@util.timer_decorator
def ChatUseLangChainSDKV2(question):
    # 创建LLM
    llm = ChatOpenAI(
        model_name="glm-4",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_key=configs.model_config.API_KEY,
    )

    # 创建Prompt
    prompt = ChatPromptTemplate.from_template("{question}")

    # 创建输出解析器
    output_parser = StrOutputParser()

    # 创建Chain
    chain = prompt | llm | output_parser

    # 调用Chain
    answer = chain.invoke({'question': question})

    res = f"[ChatUseLangChainSDKV0]\n question:{question}\n answer: {answer}"
    print(res)

#with RunnablePassthrough
@util.timer_decorator
def ChatUseLangChainSDKV3(question):
    # 创建LLM
    llm = ChatOpenAI(
        model_name="glm-4",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_key=configs.model_config.API_KEY,
    )

    # 创建Prompt
    prompt = ChatPromptTemplate.from_template("{question}")

    # 创建输出解析器
    output_parser = StrOutputParser()

    # 创建Chain
    chain = {"question": RunnablePassthrough()}|prompt | llm | output_parser


    # 调用Chain
    answer = chain.invoke(question)

    res = f"[{inspect.currentframe().f_code.co_name}]\n question:{question}\n answer: {answer}"
    print(res)

#with DAG I think no need
@util.timer_decorator
def ChatUseLangChainSDKV4(question):
    # 创建LLM
    llm = ChatOpenAI(
        model_name="glm-4",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_key=configs.model_config.API_KEY,
    )

    # 创建Prompt
    prompt = ChatPromptTemplate.from_template("{question}")

    # 创建输出解析器
    output_parser = StrOutputParser()

    # 创建Chain
    chain = {"question": RunnablePassthrough()}|prompt | llm | output_parser


    # 调用Chain
    answer = chain.invoke(question)

    res = f"[{inspect.currentframe().f_code.co_name}]\n question:{question}\n answer: {answer}"
    print(res)


if __name__ == '__main__':
    ChatUseLangChainSDKV3("我想知道你的基本信息")
