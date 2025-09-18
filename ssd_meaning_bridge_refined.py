# -*- coding: utf-8 -*-
"""
SSD意味ブリッジ：ラベル付けエンジン統合版（改良版）
— DL埋め込み ⇔ SSDコア間の意味的変換層 —

このコードは、SSD-DLサンドイッチモデルの心臓部である「意味のブリッジ」を実装します。
主な機能：
1.  人間の言語（テキスト）を、SSDコアが理解できる多層的な「意味圧ベクトル」に変換（エンコード）。
2.  SSDコアの思考結果である「反応ベクトル」を、DL生成モデルが利用できる埋め込みやプロンプトに変換（デコード）。
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

# 外部ファイルからラベル付けエンジンをインポート
from simple_ssd_labeling_engine import SimpleSSDLabelingEngine, SSDLabel, SSLayerType, SSConcept

# --- データ構造定義 ---

@dataclass
class MeaningPressureVector:
    """SSDコアへの入力形式。四層構造に対応したベクトルを持つ。"""
    physical: np.ndarray    # 物理層への意味圧ベクトル
    base: np.ndarray        # 基層への意味圧ベクトル
    core: np.ndarray        # 中核層への意味圧ベクトル
    upper: np.ndarray       # 上層への意味圧ベクトル
    metadata: Dict          # ラベル情報など、計算以外の付随データ

@dataclass
class ReactionVector:
    """SSDコアからの出力形式。思考結果を要約する。"""
    concept_activations: np.ndarray     # 各概念ノードの最終的な活性度
    layer_responses: Dict[str, float]   # 四層それぞれの反応強度
    emergent_keywords: List[str]        # 「跳躍」によって生まれた新しいキーワード
    confidence: float                   # 思考結果に対する全体的な確信度

# --- メインクラス ---

class SSDBridge:
    """
    SSD意味ブリッジ：DLの次元空間とSSDの力学モデルを繋ぐ変換器。
    """
    
    def __init__(self, 
                 embedding_dim: int = 512,
                 concept_nodes: int = 1000,
                 layer_dim: int = 256):
        """
        ブリッジの初期化
        Args:
            embedding_dim (int): DLモデルが扱う埋め込みベクトルの次元数
            concept_nodes (int): SSDコアが持つ内部概念ノードの数
            layer_dim (int): SSDの各層を表現するベクトルの基本次元数
        """
        self.embedding_dim = embedding_dim
        self.concept_nodes = concept_nodes
        self.layer_dim = layer_dim
        
        # 1. 簡易的なキーワードベースのラベル付けエンジンを初期化
        self.labeler = SimpleSSDLabelingEngine()
        
        # 2. DL埋め込み → SSD意味圧ベクトル への変換ニューラルネットワークを構築
        self.dl_to_ssd_net = self._build_dl_to_ssd_bridge()
        
        # 3. SSD反応ベクトル → DL生成用埋め込み への変換ニューラルネットワークを構築
        self.ssd_to_dl_net = self._build_ssd_to_dl_bridge()
        
        # 4. SSDの主要概念ラベルと、内部ノードIDのマッピング辞書を作成
        self.concept_mapping = self._init_concept_mapping()

    # --- ブリッジの構築 ---

    def _build_dl_to_ssd_bridge(self) -> nn.ModuleDict:
        """DL埋め込み → SSD意味圧ベクトル 変換器を構築"""
        # 各層ごとに異なる射影ネットワーク（プロジェクター）を用意する
        # これにより、単一のDL埋め込みから多層的な意味を抽出する
        return nn.ModuleDict({
            'physical_projector': nn.Sequential(
                nn.Linear(self.embedding_dim, self.layer_dim),
                nn.ReLU(),
                nn.Linear(self.layer_dim, self.layer_dim)
            ),
            'base_projector': nn.Sequential(
                nn.Linear(self.embedding_dim, self.layer_dim * 2),
                nn.ReLU(), 
                nn.Linear(self.layer_dim * 2, self.layer_dim * 2)
            ),
            'core_projector': nn.Sequential(
                nn.Linear(self.embedding_dim, self.layer_dim * 2),
                nn.ReLU(),
                nn.Linear(self.layer_dim * 2, self.layer_dim * 2)
            ),
            'upper_projector': nn.Sequential(
                nn.Linear(self.embedding_dim, self.layer_dim),
                nn.ReLU(),
                nn.Linear(self.layer_dim, self.layer_dim)
            ),
        })

    def _build_ssd_to_dl_bridge(self) -> nn.Module:
        """SSD反応ベクトル → DL生成用埋め込み 変換器を構築"""
        # SSDの思考結果（概念活性度＋層の反応）をフラットなベクトルとして受け取る
        total_input_dim = self.concept_nodes + len(SSLayerType)
        
        return nn.Sequential(
            nn.Linear(total_input_dim, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim) # 出力を安定化させる
        )

    def _init_concept_mapping(self) -> Dict[str, int]:
        """概念ラベル → SSDコアのノードIDのマッピングを初期化"""
        mapping = {}
        node_id = 0
        
        # SSDの主要概念を固定IDにマッピング
        for concept in SSConcept:
            mapping[concept.value] = node_id
            node_id += 1
            
        # 四層構造もノードとしてマッピング
        for layer in SSLayerType:
            mapping[f"layer_{layer.value.lower()}"] = node_id
            node_id += 1

        # 一般的なキーワードも追加
        common_keywords = [
            "プレッシャー", "ストレス", "要求", "期待", "負荷",
            "うまくいく", "調和", "バランス", "安定", "快適",
            "閃き", "アイデア", "発想", "革新", "変革",
            "退屈", "つまらない", "飽きた", "マンネリ",
            "本能", "感情", "欲求", "ルール", "制度",
            "理念", "価値観", "信念", "物語", "意味"
        ]
        
        for keyword in common_keywords:
            if keyword not in mapping:
                mapping[keyword] = node_id
                node_id += 1
                if node_id >= self.concept_nodes: break # ノード数上限
        return mapping

    # --- エンコード/デコード処理 ---

    def encode_to_ssd(self, 
                      text: str, 
                      dl_embedding: Optional[torch.Tensor] = None) -> MeaningPressureVector:
        """
        テキスト(+DL埋め込み)からSSD意味圧ベクトルへエンコードする。
        """
        
        # 1. ラベル付けエンジンでテキストからSSD概念をキーワードベースで抽出
        labels = self.labeler.label_text(text)
        
        # 2. 抽出したラベルに基づき、各層への初期意味圧ベクトルを生成
        layer_pressures = self._labels_to_layer_pressures(labels)
        
        # 3. 外部DLモデルからの埋め込みがあれば、それを統合して意味圧を強化
        if dl_embedding is not None:
            enhanced_pressures = self._integrate_dl_embedding(
                layer_pressures, dl_embedding
            )
        else:
            enhanced_pressures = layer_pressures
            
        # 4. SSDコアに渡すためのメタデータを作成
        metadata = {
            'original_text': text,
            'detected_labels': [
                {
                    'concept': label.concept.value,
                    'layer': label.layer.value, 
                    'confidence': label.confidence,
                    'evidence': label.evidence_text
                }
                for label in labels
            ],
            'layer_activation_summary': {
                layer.value: sum(1 for l in labels if l.layer == layer)
                for layer in SSLayerType
            }
        }
        
        return MeaningPressureVector(
            physical=enhanced_pressures['physical'],
            base=enhanced_pressures['base'],
            core=enhanced_pressures['core'],
            upper=enhanced_pressures['upper'],
            metadata=metadata
        )

    def decode_from_ssd(self, reaction: ReactionVector) -> torch.Tensor:
        """
        SSD反応ベクトルからDL生成モデル用の埋め込みへデコードする。
        """
        
        # 1. SSDの反応ベクトル（概念活性度、層の反応）をPyTorchテンソルに変換
        concept_features = torch.FloatTensor(reaction.concept_activations)
        layer_features = torch.FloatTensor([
            reaction.layer_responses.get('physical', 0.0),
            reaction.layer_responses.get('base', 0.0), 
            reaction.layer_responses.get('core', 0.0),
            reaction.layer_responses.get('upper', 0.0)
        ])
        
        # 2. 特徴量を結合して単一のベクトルにする
        combined_features = torch.cat([concept_features, layer_features])
        
        # 3. 変換ネットワークを通して、DLモデルが理解できる埋め込みベクトルに変換
        dl_embedding = self.ssd_to_dl_net(combined_features)
        
        # 4. 「跳躍」で生まれたキーワードがあれば、それを埋め込みに反映させる
        if reaction.emergent_keywords:
            keyword_adjustment = self._compute_keyword_adjustment(
                reaction.emergent_keywords
            )
            # ベクトルを微調整して、創発的なニュアンスを加える
            dl_embedding = dl_embedding + keyword_adjustment
            
        return dl_embedding

    # --- 補助メソッド ---

    def _labels_to_layer_pressures(self, labels: List[SSDLabel]) -> Dict[str, np.ndarray]:
        """抽出したラベル情報から、各層への意味圧ベクトルを生成する"""
        layer_pressures = {
            'physical': np.zeros(self.layer_dim),
            'base': np.zeros(self.layer_dim * 2),
            'core': np.zeros(self.layer_dim * 2), 
            'upper': np.zeros(self.layer_dim)
        }
        
        for label in labels:
            layer_key = label.layer.value.lower()
            if layer_key not in layer_pressures: continue
                
            # 概念の種類に応じて、ベクトルの特定の位置を活性化させる
            concept_dim_index = self._concept_to_dimension(label.concept)
            activation_strength = label.confidence
            
            target_vector = layer_pressures[layer_key]
            if concept_dim_index < len(target_vector):
                target_vector[concept_dim_index] += activation_strength
        
        return layer_pressures

    def _concept_to_dimension(self, concept: SSConcept) -> int:
        """概念Enumをベクトルの次元インデックスにマッピングする（簡易版）"""
        concept_hash = hash(concept.value)
        # 簡易的にハッシュ値を使っているが、本来は学習可能なマッピングが望ましい
        return concept_hash % (self.layer_dim * 2) # 最大の次元数で割る

    def _integrate_dl_embedding(self, 
                                layer_pressures: Dict[str, np.ndarray],
                                dl_embedding: torch.Tensor) -> Dict[str, np.ndarray]:
        """キーワードベースの圧力と、DL埋め込みからの射影を統合する"""
        with torch.no_grad():
            # DL埋め込みを各層の次元空間に射影
            projections = {
                'physical': self.dl_to_ssd_net['physical_projector'](dl_embedding).numpy(),
                'base': self.dl_to_ssd_net['base_projector'](dl_embedding).numpy(),
                'core': self.dl_to_ssd_net['core_projector'](dl_embedding).numpy(),
                'upper': self.dl_to_ssd_net['upper_projector'](dl_embedding).numpy()
            }
            
            # ラベルベースの圧力とDL埋め込みからの圧力を重み付きで加算
            alpha = 0.6  # ラベルベースの信頼度
            beta = 0.4   # DL埋め込みの信頼度
            
            enhanced = {
                layer: alpha * layer_pressures[layer] + beta * projections[layer]
                for layer in layer_pressures
            }
        return enhanced

    def _compute_keyword_adjustment(self, keywords: List[str]) -> torch.Tensor:
        """創発キーワードから埋め込みの調整ベクトルを計算する（簡易版）"""
        adjustment = torch.zeros(self.embedding_dim)
        for keyword in keywords:
            # キーワードのハッシュを使い、決定論的だが予測不能な調整を加える
            hash_val = hash(keyword)
            torch.manual_seed(hash_val)
            adjustment += torch.randn(self.embedding_dim) * 0.05
        return adjustment

    def create_prompt_from_reaction(self, reaction: ReactionVector) -> str:
        """SSD反応から、デバッグやLLMへの指示に使えるプロンプトを生成する"""
        
        # 最も活性化した概念を特定
        top_concepts = []
        if len(reaction.concept_activations) > 0:
            # 概念マッピングから逆引き
            reverse_mapping = {v: k for k, v in self.concept_mapping.items()}
            top_indices = np.argsort(reaction.concept_activations)[-3:] # 上位3つ
            top_concepts = [reverse_mapping.get(i, f"unknown_concept_{i}") for i in top_indices]

        # 最も反応が強かった層を特定
        dominant_layer = max(reaction.layer_responses.items(), key=lambda item: item[1])[0]

        # プロンプトを組み立てる
        prompt = f"思考結果の要約:\n"
        prompt += f"- 主要な視点: {dominant_layer.upper()}層からのアプローチ\n"
        if top_concepts:
            prompt += f"- 核心コンセプト: {', '.join(reversed(top_concepts))}\n"
        if reaction.emergent_keywords:
            prompt += f"- 創発キーワード: {', '.join(reaction.emergent_keywords)} (これが跳躍の結果です)\n"
        prompt += f"- 確信度: {reaction.confidence:.0%}\n"
        prompt += "上記を基に、創造的かつ一貫性のある応答を生成してください。"
        
        return prompt

# --- デモ用関数 ---

def dummy_ssd_core(pressure: MeaningPressureVector) -> ReactionVector:
    """
    SSDコアの動作を模擬するダミー関数。
    入力された意味圧に応じて、もっともらしい反応ベクトルを生成する。
    """
    # どの層が最も強く刺激されたかを計算
    layer_means = {
        'physical': np.mean(np.abs(pressure.physical)),
        'base': np.mean(np.abs(pressure.base)),
        'core': np.mean(np.abs(pressure.core)),
        'upper': np.mean(np.abs(pressure.upper))
    }
    dominant_layer = max(layer_means.items(), key=lambda item: item[1])[0]

    # ダミーの反応を生成
    concept_activations = np.random.rand(1000) # 1000ノード
    if dominant_layer == 'upper':
        # 上層が刺激されたら「跳躍」や「価値観」に関連するノードを活性化
        concept_activations[hash(SSConcept.JUMP.value) % 1000] = 0.9
        concept_activations[hash("価値観") % 1000] = 0.85
    elif dominant_layer == 'base':
        # 基層なら「意味圧」「感情」
        concept_activations[hash(SSConcept.MEANING_PRESSURE.value) % 1000] = 0.9
        concept_activations[hash("感情") % 1000] = 0.8
        
    return ReactionVector(
        concept_activations=concept_activations,
        layer_responses=layer_means,
        emergent_keywords=["発見", "再構築"] if dominant_layer == 'upper' else [],
        confidence=np.random.uniform(0.6, 0.9)
    )


def demo_bridge():
    """ブリッジの動作デモ"""
    bridge = SSDBridge(concept_nodes=len(SimpleSSDLabelingEngine()._build_keyword_map()) + 20) # マッピングに合わせて調整
    
    test_cases = [
        "上司からのプレッシャーが強くて、もう限界だ。ストレスで何も手につかない。",
        "毎日同じことの繰り返しで飽きた。何か新しい挑戦がしたい。",
        "突然、全てが繋がるような最高のアイデアが閃いた！これは革命になる！",
        "会社のルールと、自分が大切にしている価値観がどうしても合わない。"
    ]
    
    print("=== SSD意味ブリッジ デモ ===\n")
    
    for i, text in enumerate(test_cases, 1):
        print(f"--- テストケース {i} ---")
        print(f"👤 入力テキスト: 「{text}」")
        
        # 1. エンコード (テキスト → SSD意味圧)
        # 外部DLモデルがないため、dl_embeddingはNoneとする
        meaning_pressure = bridge.encode_to_ssd(text, dl_embedding=None)
        
        print(f"\n📥 [ブリッジ/入力層] テキストをSSD意味圧ベクトルに変換しました:")
        print(f"  - 検出された主要概念: {[l['concept'] for l in meaning_pressure.metadata['detected_labels']]}")
        print(f"  - 最も関連する層: {max(meaning_pressure.metadata['layer_activation_summary'].items(), key=lambda x:x[1])[0]}")

        # 2. 模擬SSDコアでの処理
        reaction = dummy_ssd_core(meaning_pressure)
        print("\n🧠 [模擬SSDコア] 意味圧を処理し、反応ベクトルを生成しました。")

        # 3. デコード (SSD反応 → プロンプト)
        prompt_for_llm = bridge.create_prompt_from_reaction(reaction)
        
        print(f"\n📤 [ブリッジ/出力層] 反応ベクトルから生成AI用のプロンプトを作成しました:")
        print("--------------------------------------------------")
        print(prompt_for_llm)
        print("--------------------------------------------------")

        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    demo_bridge()
