# Search_Engine.py

import numpy as np
import re
import random
from typing import List, Dict, Any

class SovereignNavigator:
    """
    [الملاح السيادي]: العقل المسؤول عن الملاحة الدلالية، البحث في المتجهات،
    وربط السياق التقني بالذاكرة المخبئية.
    """
    def __init__(self, network, embedder, logger):
        self.network = network
        self.embedder = embedder
        self.logger = logger
        self.seen_content = set() # ذاكرة قصيرة لمنع التكرار في نفس الجلسة

    def _extract_technical_tokens(self, text: str) -> List[str]:
        """
        [التشريح السيادي]: استخراج الرموز الهندسية والكلمات التقنية بدقة عالية.
        تم تطويره ليشمل المعادلات والرموز الرياضية المعقدة.
        """
        # استهداف: الكلمات التقنية، الرموز اليونانية، والمصطلحات المكتوبة بـ CamelCase أو ALL_CAPS
        patterns = [
            r'\b[A-Z][a-z]{3,}\b',                 # كلمات تبدأ بحرف كبير (Jacobian)
            r'[θφαβγδεζηικλμνξοπρστυφχψω]',       # الرموز اليونانية كاملة
            r'\b[A-Z]{2,}\b',                      # الاختصارات التقنية (DOF, PTP)
            r'[A-Za-z]\d+',                        # المتغيرات المرقمة (L1, q2, T3)
            r'[\+\-\*\/=\^√Σ∫]'                    # العمليات الحسابية الأساسية
        ]

        combined_pattern = "|".join(patterns)
        tokens = re.findall(combined_pattern, text)

        # تنظيف وتصفية الرموز (إزالة المكرر والحفاظ على الترتيب الأهم)
        unique_tokens = []
        for t in tokens:
            if t not in unique_tokens:
                unique_tokens.append(t)

        return unique_tokens[:15] # العودة بأهم 15 رمزاً فقط لعدم إثقال السياق

    def aggregate_context(self, matches: List[Dict[str, Any]]) -> str:
        """
        [تجميع السيادة]: دمج الكتل النصية مع وسمها تقنياً لبناء صورة ذهنية كاملة للمحرك.
        """
        context_blocks = []
        for match in matches:
            content = match.get('content', '')
            metadata = match.get('metadata', {})
            page = metadata.get('page', '??')

            # استخراج الأوسمة التقنية لهذا المقطع تحديداً
            tokens = self._extract_technical_tokens(content)

            # بناء كتلة سياقية غنية بالبيانات الوصفية
            block_header = f"📍 [ENTRY_POINT: PAGE {page}]"
            token_line = f"🏷️ [TECHNICAL_SIGNATURE: {', '.join(tokens[:8])}]"

            full_block = f"{block_header}\n{token_line}\n{content.strip()}"
            context_blocks.append(full_block)

        return "\n\n" + "—" * 30 + "\n\n".join(context_blocks)

    def infer_navigation_path(self, query: str) -> List[int]:
        """
        [تحديد المسار السيادي]: تحليل الاستفسار لتوقع الصفحات ذات الصلة
        باستخدام الروابط الدلالية والأوزان التكرارية.
        """
        target_pages = []
        page_scores = {} # حساب وزن كل صفحة بناءً على تكرار الرموز

        # 1. استخراج الرموز التقنية من الاستعلام
        query_tokens = self._extract_technical_tokens(query)

        if not query_tokens:
            self.logger.warning("⚠️ [NAVIGATOR]: No technical tokens found in query to infer path.")
            return []

        # 2. المسح المتقاطع في الرسم البياني للمعرفة (Knowledge Graph)
        for token in query_tokens:
            # البحث عن الرمز مباشرة أو عن رموز قريبة منه (تجاهل حالة الأحرف)
            for registered_token, pages in self.network.knowledge_graph.items():
                if token.lower() in registered_token.lower():
                    for p in pages:
                        # زيادة وزن الصفحة كلما احتوت على رموز أكثر من الاستعلام
                        page_scores[p] = page_scores.get(p, 0) + 1

        # 3. ترتيب الصفحات بناءً على الوزن (الأكثر صلة أولاً)
        sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        target_pages = [p[0] for p in sorted_pages]

        if target_pages:
            self.logger.info(f"📍 [NAVIGATOR]: Path inferred through tokens. High-priority pages: {target_pages[:5]}")

        return target_pages

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        البوصلة الدلالية المطورة (الإصلاح الذي اعتمدناه سابقاً).
        """
        query_vector = self.embedder.encode_text(query)
        # إضافة تذبذب بسيط (Stochastic Exploration) لكسر الجمود
        noise = np.random.normal(0, 0.01, query_vector.shape)
        query_vector = query_vector + noise

        results = []

        for page_num, data in self.network.cache.items():
            content_vector = data.get('embedding')
            content = data.get('content', '')

            if content_vector is not None:
                similarity = np.dot(query_vector, content_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(content_vector)
                )

                # عقوبة التكرار
                if content in self.seen_content:
                    similarity -= 0.25

                results.append({
                    "score": similarity,
                    "content": content,
                    "metadata": data.get('metadata', {})
                })

        results.sort(key=lambda x: x['score'], reverse=True)

        # اختيار متنوع من أفضل 10 مرشحين
        top_candidates = results[:10]
        final_selection = random.sample(top_candidates, min(len(top_candidates), top_k))
        final_selection.sort(key=lambda x: x['score'], reverse=True)

        for res in final_selection:
            self.seen_content.add(res['content'])

        self.logger.info(f"🎯 [NAVIGATOR]: Context synthesized from {len(final_selection)} nodes.")
        return final_selection
