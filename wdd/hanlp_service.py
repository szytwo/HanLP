from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Iterable, Optional

import hanlp
from wdd.file_utils import logging


class HanlpService:
    """
    HanLP 服务类。
    负责统一管理所有 HanLP 模型。
    """

    def __init__(self) -> None:
        """初始化 Service，不加载模型。"""

        self._eos = None
        self._tok = None
        self._pos = None
        self._ner = None
        self._dep = None

        # 修改 tokenizer.dict_force 时需要加锁
        self._lock = Lock()

    def load(self) -> None:
        """
        加载所有 HanLP 模型。
        建议在 FastAPI lifespan 中调用一次。
        """

        logging.info("Loading HanLP models...")

        try:
            # 分句
            self._eos = hanlp.load(hanlp.pretrained.eos.UD_CTB_EOS_MUL)
            # 分词
            self._tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
            # 词性标注
            self._pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
            # 命名实体识别
            self._ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
            # 依存句法
            self._dep = hanlp.load(hanlp.pretrained.dep.CTB9_DEP_ELECTRA_SMALL)

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
            self._eos is not None
            and self._tok is not None
            and self._pos is not None
            and self._ner is not None
            and self._dep is not None
        )

    def _check_loaded(self) -> None:
        """
        检查模型是否已加载。
        """

        if not self.loaded:
            raise RuntimeError("HanLP model has not been loaded.")

    def eos(self, text: str | list[str]) -> list[str]:
        """
        中文分句。

        :param text: 输入文本
        :return: 分句结果
        """

        self._check_loaded()

        if not text:
            raise ValueError("text is empty.")

        return self._eos(text)

    def tokenize(
        self, text: str | list[str], dict_force: Optional[Iterable[str]] = None
    ) -> list[str]:
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
                self._tok.dict_force = dict_force
            else:
                self._tok.dict_force = None

            return self._tok(text)

    def pos(
        self, text: str | list[str], dict_force: Optional[Iterable[str]] = None
    ) -> Any:
        """
        词性标注。

        :param text: 输入文本
        :param dict_force: 强制词典
        :return: 每个词对应的词性
        """

        self._check_loaded()

        if not text:
            raise ValueError("text is empty.")

        with self._lock:
            if dict_force:
                self._tok.dict_force = dict_force
            else:
                self._tok.dict_force = None

            HanLP = (
                hanlp.pipeline()
                .append(self._tok, output_key="tok")
                .append(self._pos, output_key="pos")
            )

            return HanLP(text)

    def ner(
        self, text: str | list[str], dict_force: Optional[Iterable[str]] = None
    ) -> Any:
        """
        命名实体识别。

        :param text: 输入文本
        :param dict_force: 强制词典
        :return: HanLP 返回结果
        """

        self._check_loaded()

        if not text:
            raise ValueError("text is empty.")

        with self._lock:
            if dict_force:
                self._tok.dict_force = dict_force
            else:
                self._tok.dict_force = None

            HanLP = (
                hanlp.pipeline()
                .append(self._tok, output_key="tok")
                .append(self._pos, output_key="pos")
                .append(self._ner, output_key="ner", input_key="tok")
            )

            return HanLP(text)

    def dependency(
        self, text: str | list[str], dict_force: Optional[Iterable[str]] = None
    ) -> Any:
        """
        依存句法分析。

        :param text: 输入文本
        :param dict_force: 强制词典
        :return: HanLP Dependency 结果
        """

        self._check_loaded()

        if not text:
            raise ValueError("text is empty.")

        with self._lock:
            if dict_force:
                self._tok.dict_force = dict_force
            else:
                self._tok.dict_force = None

            HanLP = (
                hanlp.pipeline()
                .append(self._tok, output_key="tok")
                .append(self._pos, output_key="pos")
                .append(self._dep, output_key="dep", input_key="tok")
            )

            return HanLP(text)

    def analyze(
        self, text: str | list[str], dict_force: Optional[Iterable[str]] = None
    ) -> Any:
        """
        一次完成 NLP 全流程。
        流程：文本->Tokenize->POS->NER->Dependency

        :param text: 输入文本
        :param dict_force: 强制词典
        :return: 全部分析结果。
        """

        self._check_loaded()

        if not text:
            raise ValueError("text is empty.")

        with self._lock:
            if dict_force:
                self._tok.dict_force = dict_force
            else:
                self._tok.dict_force = None

            HanLP = (
                hanlp.pipeline()
                .append(self._tok, output_key="tok")
                .append(self._pos, output_key="pos")
                .append(self._ner, output_key="ner", input_key="tok")
                .append(self._dep, output_key="dep", input_key="tok")
            )

            return HanLP(text)

    def status(self) -> Dict[str, bool]:
        """
        返回模型状态。
        """

        return {
            "eos": self._eos is not None,
            "tok": self._tok is not None,
            "pos": self._pos is not None,
            "ner": self._ner is not None,
            "dep": self._dep is not None,
        }
