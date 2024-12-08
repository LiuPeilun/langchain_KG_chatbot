import os.path

from utils import *
from prompt import *
from config import *

from langchain.chains.llm import LLMChain
from langchain.chains import LLMRequestsChain
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.agents import ZeroShotAgent, AgentExecutor, Tool
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


class Agent():
    def __init__(self):
        self.db = Chroma(
            embedding_function=get_embedding_model(),
            persist_directory=os.path.join(os.path.dirname(__file__), 'data\\db\\')
        )


    # 定义Tool函数
    # --大模型本身
    def generic_func(self, query):
        prompt = PromptTemplate.from_template(
            template=GENERIC_PROMPT_TPL
        )
        llm_chain = LLMChain(
            llm=get_llm_model(),
            prompt=prompt,
            verbose=os.getenv('VERBOSE')
        )

        return llm_chain.invoke(query)['text']

    # --大模型+向量数据库检索
    def retrival_func(self, query):
        # 召回
        documents = self.db.similarity_search_with_relevance_scores(query, k=3)
        # 过滤低相似度召回文本 doc[0]召回文本内容  doc[1]召回文本的相似度分数
        query_result = [doc[0].page_content for doc in documents if doc[1] > 0.7]
        # 构建提示词
        prompt = PromptTemplate.from_template(
            RETRIVAL_PROMPT_TPL
        )
        llm_chain = LLMChain(
            llm=get_llm_model(),
            prompt=prompt,
            verbose=os.getenv('VERBOSE')
        )

        inputs = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if len(query_result) else '没有查到',
        }

        return llm_chain.invoke(inputs)['text']

    # 图数据库检索
    def graph_func(self, x, query):
        # 命名实体识别
        response_schemas = [
            ResponseSchema(type='list', name='disease', description='疾病名称实体'),
            ResponseSchema(type='list', name='symptom', description='疾病症状实体'),
            ResponseSchema(type='list', name='drug', description='药物名称实体'),
        ]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = structured_output_parser(response_schemas)

        prompt = PromptTemplate(
            template=NER_PROMPT_TPL,
            partial_variables={'format_instructions': format_instructions},
            input_variables=['query']
        )

        llm_chain = LLMChain(
            llm=get_llm_model(),
            prompt=prompt,
            verbose=os.getenv('VERBOSE')
        )

        # 完成命名实体识别
        result = llm_chain.invoke(query)['text']
        result = output_parser.parse(result)

        # 要将命名实体识别出的实体，添加到模板中，而后才能根据模板中新组成的问题进行数据库检索及后续问答
        graph_templates = []
        for key, template in GRAPH_TEMPLATE.items():
            slot = template['slots'][0]
            slot_values = result[slot]
            for value in slot_values:
                graph_templates.append({
                    'question': replace_token_in_string(template['question'], [[slot, value]]),
                    'cypher': replace_token_in_string(template['cypher'], [[slot, value]]),
                    'answer': replace_token_in_string(template['answer'], [[slot, value]])
                })
        if not graph_templates:
            return

        # 模板已经全部填充完，此时需要与用户的问题做相似度计算，而后根据相似度筛选出真正有用的问题，过滤掉不相关的问题
        graph_documents = [
            Document(page_content=template['question'], metadata=template)
            for template in graph_templates
        ]
        db = FAISS.from_documents(graph_documents, get_embedding_model())
        graph_documents_filter = db.similarity_search_with_relevance_scores(query=query, k=3)
        print(graph_documents_filter)

        # 执行CQL,拿到结果
        query_result = []
        neo4j_conn = get_neo4j_conn()
        # 把筛选好的问题中的question，cypher，answer等字段提取出来，通过cypher查询数据库
        for document in graph_documents_filter:
            question = document[0].page_content
            cypher = document[0].metadata['cypher']
            answer = document[0].metadata['answer']
            try:
                result = neo4j_conn.run(cypher).data()
                # print(question)
                # print(result)
                # exit()
                # 检索到的result不为空，且'RES'对应的value不为空，代表成功检索到答案，
                # 对模板中answer字段对应文本中的%RES%替换(替换为检索到的答案)
                if result and any(value for value in result[0].values()):
                    answer_str = replace_token_in_string(answer, list(result[0].items()))
                    # print('result[0]: ', result[0])
                    # print('result[0].items: ', result[0].items)
                    # exit()
                    query_result.append(f"问题：{question}\n答案：{answer_str}")
            except:
                pass

        # print('---------------------------------------------------------------', query_result)
        # 最终根据过滤好的问题，查询数据库，得到了答案，之后需要将得到的答案给到大模型，让大模型进行总结输出
        prompt = PromptTemplate.from_template(GRAPH_PROMPT_TPL)
        graph_chain = LLMChain(
            llm=get_llm_model(),
            prompt=prompt,
            verbose=os.getenv('VERBOSE')
        )
        inputs = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if len(query_result) else '没有查到'
        }

        return graph_chain.invoke(inputs)['text']

    # 搜索函数
    def search_func(self, query):
        prompt = PromptTemplate.from_template(SEARCH_PROMPT_TPL)
        llm_chain = LLMChain(
            llm=get_llm_model(),
            prompt=prompt,
            verbose=os.getenv('VERBOSE')
        )
        llm_request_chain = LLMRequestsChain(
            llm_chain=llm_chain,
            requests_key='query_result',
        )
        inputs = {
            'query': query,
            'url': 'https://www.so.com/s?q='+query.replace(" ", "+")
        }
        return llm_request_chain.invoke(inputs)

    # 问题入口，在这里定义多种tool以供选择
    def query(self, query):
        tools = [
            Tool.from_function(
                name='generic_func',
                func=lambda x: self.generic_func(x, query),
                description='可以解答通用领域的知识，例如打招呼，问你是谁等问题'
            ),
            Tool.from_function(
                name='retrival_func',
                func=lambda x: self.retrival_func(x, query),
                description='用于回答寻医问药网相关消息'
            ),
            Tool.from_function(
                name='graph_func',
                func=lambda x: self.graph_func(x, query),
                description='用于回答疾病、症状、药物等医疗相关问题'
            ),
            Tool.from_function(
                name='search_func',
                func=self.search_func,
                description='其他工具没有正确答案时，通过搜索引擎，回答通用类问题'
            ),
        ]

        # 构建agent
        prefix = """请用中文，尽你所能回答以下问题。你可以使用以下工具："""
        suffix = """Begin!
        
        History:{chat_history}
        Question:{input}
        Thought:{agent_scratchpad}
        """
        agent_prompt = ZeroShotAgent.create_prompt(
            tools=tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=['chat_history', 'input', 'agent_scratchpad']
        )

        llm_chain = LLMChain(
            llm=get_llm_model(),
            prompt=agent_prompt
        )

        agent = ZeroShotAgent(llm_chain=llm_chain)

        memory = ConversationBufferMemory(memory_key='chat_history')

        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=os.getenv('VERBOSE')
        )
        return agent_chain.invoke({'input': query})


# agent = Agent()
# print(agent.query('你好'))
# print(agent.query('寻医问药网获得过哪些投资？'))
# print(agent.query('鼻炎和感冒是并发症吗？'))