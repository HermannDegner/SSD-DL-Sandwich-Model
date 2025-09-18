# -*- coding: utf-8 -*-
"""
SSDサンドイッチモデルに必要な、簡易的なSSD概念ラベル付けエンジン。
キーワードマッチングに基づき、テキストからSSDの概念を抽出します。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict

class SSLayerType(Enum):
    """SSDの四層構造"""
    PHYSICAL = "物理"
    BASE = "基層"
    CORE = "中核"
    UPPER = "上層"

class SSConcept(Enum):
    """SSDの主要概念"""
    MEANING_PRESSURE = "意味圧"
    ALIGNMENT = "整合"
    JUMP = "跳躍"
    BOREDOM = "退屈"
    ALIGNMENT_INERTIA = "整合慣性"
    UNPROCESSED_PRESSURE = "未処理圧"
    STRUCTURE_OBSERVATION = "構造観照"

@dataclass
class SSDLabel:
    """テキストから抽出された単一のSSDラベル"""
    concept: SSConcept
    layer: SSLayerType
    confidence: float
    evidence_text: str

class SimpleSSDLabelingEngine:
    """
    キーワードベースのシンプルなSSDラベル付けエンジン。
    """
    def __init__(self):
        self.keyword_map: Dict[str, SSDLabel] = self._build_keyword_map()

    def _build_keyword_map(self) -> Dict[str, SSDLabel]:
        """キーワードとSSDラベルを紐付ける辞書を構築"""
        mapping = {}
        
        # 意味圧
        pressure_words = ["プレッシャー", "ストレス", "要求", "期待", "負荷", "圧力"]
        for word in pressure_words:
            mapping[word] = SSDLabel(SSConcept.MEANING_PRESSURE, SSLayerType.BASE, 0.8, word)

        # 整合
        alignment_words = ["うまくいく", "調和", "バランス", "安定", "快適", "納得", "解決"]
        for word in alignment_words:
            mapping[word] = SSDLabel(SSConcept.ALIGNMENT, SSLayerType.CORE, 0.85, word)

        # 跳躍
        jump_words = ["閃いた", "アイデア", "発想", "革新", "変革", "突然", "ひらめき"]
        for word in jump_words:
            mapping[word] = SSDLabel(SSConcept.JUMP, SSLayerType.UPPER, 0.9, word)
            
        # 退屈
        boredom_words = ["退屈", "つまらない", "飽きた", "マンネリ", "ルーティン"]
        for word in boredom_words:
            mapping[word] = SSDLabel(SSConcept.BOREDOM, SSLayerType.BASE, 0.8, word)

        # 上層の概念
        upper_words = ["価値観", "信念", "物語", "意味", "理念"]
        for word in upper_words:
            mapping[word] = SSDLabel(SSConcept.STRUCTURE_OBSERVATION, SSLayerType.UPPER, 0.7, word)
            
        # 中核の概念
        core_words = ["ルール", "制度", "法律", "社会"]
        for word in core_words:
            mapping[word] = SSDLabel(SSConcept.ALIGNMENT_INERTIA, SSLayerType.CORE, 0.7, word)
            
        return mapping

    def label_text(self, text: str) -> List[SSDLabel]:
        """
        テキストを解析し、含まれるSSD概念のラベルリストを返す。
        """
        found_labels = []
        for keyword, label_template in self.keyword_map.items():
            if keyword in text:
                # テンプレートをコピーして具体的な証拠テキストを設定
                found_labels.append(SSDLabel(
                    concept=label_template.concept,
                    layer=label_template.layer,
                    confidence=label_template.confidence,
                    evidence_text=keyword
                ))
        
        # ラベルが見つからない場合、デフォルトのラベルを返す
        if not found_labels:
            found_labels.append(SSDLabel(
                SSConcept.MEANING_PRESSURE, SSLayerType.CORE, 0.5, text
            ))
            
        return found_labels
