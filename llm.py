import inspect
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from zhipuai import ZhipuAI
import util
import configs.model_config
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
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

# with memory
@util.timer_decorator
def ChatUseLangChainSDKV5(question,question_2):
    # 创建LLM
    llm = ChatOpenAI(
        model_name="glm-4",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_key=configs.model_config.API_KEY,
    )

    # 创建Prompt
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ])

    # 创建输出解析器
    output_parser = StrOutputParser()

    # 创建Chain
    chain = prompt | llm | output_parser

    # 添加History
    history = ChatMessageHistory()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    # 调用Chain
    answer = chain_with_history.invoke({'question': question},
                                config={"configurable": {"session_id": None}})

    res = f"[{inspect.currentframe().f_code.co_name}]\n question:{question}\n answer: {answer}"
    print(res)

    answer = chain_with_history.invoke({'question': question_2},
                                config={"configurable": {"session_id": None}})

    res = f"[{inspect.currentframe().f_code.co_name}]\n question_2:{question_2}\n answer: {answer}"
    print(res)

# with rag
@util.timer_decorator
def ChatUseLangChainSDKV6(question):
    # 创建LLM
    llm = ChatOpenAI(
        model_name="glm-4",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_key=configs.model_config.API_KEY,
    )

    # 创建Prompt
    prompt = ChatPromptTemplate.from_template('基于上下文：{context}\n回答：{input}')

    # 创建输出解析器
    output_parser = StrOutputParser()

    # 模拟文档
    docs = [Document(page_content="LMG:LIVE Monetization Governance 直播商业治理，指的是当主播利用直播营收相关功能在平台产生了影响品牌形象/有损正向体验/破坏社区诚信的内容或者行为时，平台对用户的营收能力进行相应限制的治理方式。")]

    embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh")
    # 文档嵌入
    splits = RecursiveCharacterTextSplitter().split_documents(docs)
    vector_store = FAISS.from_documents(splits, embeddings_model)
    retriever = vector_store.as_retriever()

    # 创建Chain
    chain_no_context = RunnablePassthrough() | llm | output_parser
    chain = (
            {"context": retriever, "input": RunnablePassthrough()}
            | prompt | llm | output_parser
    )

    # 调用Chain
    chain_no_context_res=chain_no_context.invoke(question)
    res = f"[{inspect.currentframe().f_code.co_name}]\n question:{question}\n chain_no_context_res: {chain_no_context_res}"
    print(res)

    chain_res=chain.invoke(question)
    res = f"[{inspect.currentframe().f_code.co_name}]\n question:{question}\n chain_res: {chain_res}"
    print(res)



if __name__ == '__main__':
    ChatUseLangChainSDKV6('LMG是什么？')
