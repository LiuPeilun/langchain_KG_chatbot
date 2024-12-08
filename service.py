import os

from prompt import *
from utils import *
from agent import *

class Service():
    def __init__(self):
        self.agent = Agent()

    def summary_message(self, message, history):
        prompt = PromptTemplate.from_template(SUMMARY_PROMPT_TPL)

        llm_chain = LLMChain(
            llm=get_llm_model(),
            prompt=prompt,
            verbose=os.getenv('VERBOSE')
        )
        # 只关注之前的两轮对话
        chat_history = ''
        for q, a in history[-2:]:
            chat_history += f'问题:{q}, 答案:{a}\n'

        inputs = {
            'query': message,
            'chat_history': chat_history
        }

        result = llm_chain.invoke(inputs)['text']
        return result

    def answer(self, message, history):
        if history:
            message = self.summary_message(message, history)
        return self.agent.query(message)


if __name__ == '__main__':
    service = Service()
    print(service.answer(
        '得了鼻炎怎么办', [['你好', '你好，有什么可以帮助你的吗']]
    ))