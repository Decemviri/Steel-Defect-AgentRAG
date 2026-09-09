from abc import abstractmethod, ABC
from typing import Optional, Union, Any
from codes.config_handler import agent_conf
from codes.logger_handler import logger

try:
    from langchain_community.chat_models import ChatTongyi
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_core.language_models import BaseChatModel
except ImportError:
    try:
        from langchain.chat_models import ChatTongyi
        from langchain.embeddings import DashScopeEmbeddings
        from langchain.schema import BaseChatModel
    except ImportError:
        ChatTongyi = None
        DashScopeEmbeddings = None
        BaseChatModel = Any


class BaseModelFactory(ABC):
    @abstractmethod
    def generate(self) -> Optional[Any]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generate(self) -> Optional[Any]:
        api_key = agent_conf.get("api_key") or agent_conf.get("dashscope_api_key")
        model_name = agent_conf.get("chat_model_name", "qwen3-max")
        if ChatTongyi is not None and api_key:
            try:
                return ChatTongyi(model=model_name, api_key=api_key)
            except Exception as e:
                logger.warning(f"[Model Factory] ChatTongyi 初始化异常: {e}")
        return None


class EmbeddingsFactory(BaseModelFactory):
    def generate(self) -> Optional[Any]:
        api_key = agent_conf.get("api_key") or agent_conf.get("dashscope_api_key")
        emb_model = agent_conf.get("embedding_model_name", "text-embedding-v4")
        if DashScopeEmbeddings is not None and api_key:
            try:
                return DashScopeEmbeddings(model=emb_model, dashscope_api_key=api_key)
            except Exception as e:
                logger.warning(f"[Model Factory] DashScopeEmbeddings 初始化异常: {e}")
        return None


chat_model = ChatModelFactory().generate()
embeddings_model = EmbeddingsFactory().generate()
