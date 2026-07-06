from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Iterable, List, Optional

import hanlp
from wdd.file_utils import logging


class HanlpService:
    """
    HanLP 服务类。
    负责统一管理所有 HanLP 模型。
    """

    def __init__(self) -> None:
        """初始化 Service，不加载模型。"""

        self._tokenizer = None
        self._pos = None
        self._ner = None
        self._dependency = None

        # 修改 tokenizer.dict_force 时需要加锁
        self._lock = Lock()

    def load(self) -> None:
        """
        加载所有 HanLP 模型。
        建议在 FastAPI lifespan 中调用一次。
        """

        logging.info("Loading HanLP models...")

        try:
            # 分词
            self._tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
            # 词性标注
            self._pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
            # 命名实体识别
            self._ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
            # 依存句法
            self._dependency = hanlp.load(hanlp.pretrained.dep.CTB9_DEP_ELECTRA_SMALL)

            logging.info("HanLP models loaded successfully.")
        except Exception:
            logging.exception("Load HanLP models failed.")
            raise

    @property
    def loaded(self) -> bool:
        """
        判断模型是否已加载。
        """
        return (
            self._tokenizer is not None
            and self._pos is not None
            and self._ner is not None
            and self._dependency is not None
        )

    def _check_loaded(self) -> None:
        """
        检查模型是否已加载。
        """

        if not self.loaded:
            raise RuntimeError("HanLP model has not been loaded.")

    def tokenize(
        self,
        text: str,
        dict_force: Optional[Iterable[str]] = None,
    ) -> List[str]:
        """
        中文分词。

        :param text: 输入文本
        :param dict_force: 强制词典
        :return: 分词结果
        """

        self._check_loaded()

        if not text:
            raise ValueError("text is empty.")

        with self._lock:
            if dict_force:
                self._tokenizer.dict_force = dict_force
            else:
                self._tokenizer.dict_force = None

            return self._tokenizer(text)

    def pos(self, words: List[str]) -> List[str]:
        """
        词性标注。

        :param words: 分词后的结果
        :return: 每个词对应的词性
        """

        self._check_loaded()

        return self._pos(words)

    def ner(self, words: List[str], pos: Optional[List[str]] = None) -> Any:
        """
        命名实体识别。

        :param words: 分词结果
        :param pos: 可选词性
        :return: HanLP 返回结果
        """

        self._check_loaded()

        if pos is None:
            return self._ner(words)

        return self._ner(words, pos)

    def dependency(self, words: List[str]) -> Any:
        """
        依存句法分析。

        :param words: 分词结果
        :return: HanLP Dependency 结果
        """

        self._check_loaded()

        return self._dependency(words)

    def analyze(
        self,
        text: str,
        dict_force: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """
        一次完成 NLP 全流程。
        流程：文本->Tokenize->POS->NER->Dependency

        :param text: 输入文本
        :param dict_force: 强制词典
        :return: 全部分析结果。
        """

        words = self.tokenize(text, dict_force)
        pos = self.pos(words)
        ner = self.ner(words, pos)
        dep = self.dependency(words)

        return {
            "tokens": words,
            "pos": pos,
            "ner": ner,
            "dependency": dep,
        }

    def status(self) -> Dict[str, bool]:
        """
        返回模型状态。
        """

        return {
            "tokenizer": self._tokenizer is not None,
            "pos": self._pos is not None,
            "ner": self._ner is not None,
            "dependency": self._dependency is not None,
        }
