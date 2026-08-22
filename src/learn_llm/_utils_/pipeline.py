from abc import ABC, abstractmethod

class PipelineNode(ABC):
    """
    一套处理框架，把复杂的事情拆成串行的流水线，每一个节点做一件事情，给一个统一的设计：
    - 启动和结束
    - 断点续传
    - 日志记录
    """
    name: str = 'empty'
    @abstractmethod
    def __call__(self, x: Any) -> Any:
        ...

    # 更多功能后面再说
    # def save(self, x: Any) -> Any:
    #     pass
    # def resume(self, x: Any) -> Any:
    #     pass