import os
from dotenv import load_dotenv

from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from file_history_store import get_history

load_dotenv(override=True)


class RagService():

    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )
        self.retriever = self.vector_service.get_vector_store()
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "以我提供的已知参考资料为主，"
                       "简洁专业回答我的用户问题，参考资料：{context}。"),
            ("system", "并且我提供用户的历史对话记录，如下："),
            MessagesPlaceholder("history"),
            ("user", "请回答用户的提问：{input}"),
        ])
        self.chat_model = ChatOpenAI(
            model=config.model_name,
            base_url=config.chat_base_url,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        self.chain = self.__get_chain()

    @staticmethod
    def _format_document(docs):
        return "无参考资料" if not docs else "".join(
            f"文档片段：{doc.page_content}\n文档原数据：{doc.metadata}\n\n" for doc in docs
        )

    @staticmethod
    def _format_sources(docs):
        """去重后的文档来源，供 API 返回引用"""
        sources = {}
        for doc in docs:
            sources.setdefault(doc.metadata.get("source", "未知来源"), doc.page_content[:200])
        return [{"file": file, "snippet": snippet} for file, snippet in sources.items()]

    def retrieve(self, question):
        """返回 (拼装好的参考资料, 原始文档列表)，评估时可直接拿到上下文"""
        docs = self.retriever.invoke(question)
        return self._format_document(docs), docs

    def _ensure_context(self, inputs):
        """未显式传入 context 时自动检索，兼容原有 chain.invoke({"input": ...}) 用法"""
        if inputs.get("context"):
            return inputs
        return {**inputs, "context": self.retrieve(inputs["input"])[0]}

    def __get_chain(self):
        """获取最终的执行链"""
        chain = (
            RunnableLambda(self._ensure_context)
            | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )
        return RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def answer(self, question, session_id="user_001"):
        """同步问答，返回回答与引用来源"""
        context, docs = self.retrieve(question)
        result = self.chain.invoke(
            {"input": question, "context": context},
            config={"configurable": {"session_id": session_id}},
        )
        return {
            "answer": result,
            "sources": self._format_sources(docs),
            "contexts": [
                f"文档来源：{doc.metadata.get('source', '未知来源')}\n{doc.page_content}"
                for doc in docs
            ],
        }

    async def astream_answer(self, question, session_id="user_001"):
        """流式问答：逐个 token 产出 {"type": "token", ...}，最后产出 {"type": "sources", ...}"""
        context, docs = self.retrieve(question)
        async for chunk in self.chain.astream(
            {"input": question, "context": context},
            config={"configurable": {"session_id": session_id}},
        ):
            if chunk:
                yield {"type": "token", "content": chunk}
        yield {"type": "sources", "sources": self._format_sources(docs)}


if __name__ == "__main__":
    result = RagService().answer("我身高190cm,尺码推荐", "user_001")
    print(result["answer"])
