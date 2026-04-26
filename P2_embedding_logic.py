# P2_embedding_logic.py

import numpy as np
import warnings
from typing import List, Union, TYPE_CHECKING # حذفنا Optional لأنها غير مستخدمة

# إيقاف التحذيرات المزعجة
warnings.filterwarnings("ignore", category=FutureWarning)

# --- الحل الصحيح لمنع "دورة الاستيراد" وإسكات Pylance ---
if TYPE_CHECKING:
    from P1_sovereign_utils import LoggerProtocol
else:
    # تعريف وهمي بسيط لتجنب خطأ NameError أثناء التشغيل إذا لم يتم استيراده
    LoggerProtocol = any
# -------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class EmbeddingManager:
    """
    [مدير التشفير المتجهي V2.0]:
    المسؤول عن تحويل النصوص الهندسية إلى متجهات رقمية.
    """
    def __init__(self, logger: 'LoggerProtocol', model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.logger = logger
        self.dimension = 384

        if HAS_TRANSFORMERS:
            try:
                self.logger.info(f"🧬 [EMBEDDING]: تحميل النموذج السيادي ({model_name})...")
                self._model = SentenceTransformer(model_name)
            except Exception as e:
                self.logger.error(f"⚠️ [EMBEDDING_ERROR]: فشل تحميل النموذج: {e}")
        else:
            self.logger.warning("❌ [CRITICAL]: sentence-transformers غير مثبتة.")

    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        تحويل النص إلى متجهات رقمية.
        في حالة الفشل، يتم إنتاج متجهات عشوائية لضمان عدم توقف "المهمة السيادية".
        """
        try:
            if self._model:
                # توليد المتجه الحقيقي
                vectors = self._model.encode(text, convert_to_numpy=True)
                return vectors.astype('float32')

            # وضع الطوارئ: إنشاء متجه وهمي متناسق مع الأبعاد المطلوبة
            if isinstance(text, list):
                count = len(text)
                return np.random.uniform(-1, 1, (count, self.dimension)).astype('float32')
            return np.random.uniform(-1, 1, self.dimension).astype('float32')

        except Exception as e:
            self.logger.error(f"❌ [ENCODE_ERROR]: حدث خطأ أثناء التشفير: {str(e)}")
            # إرجاع متجه صفري لتجنب انهيار العمليات اللاحقة
            return np.zeros(self.dimension).astype('float32')

    def calculate_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """حساب درجة التشابه الجيبي (Cosine Similarity) بين فكرتين تقنيتين."""
        try:
            # التأكد من أن المتجهات ليست صفرية
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0

            dot_product = np.dot(v1, v2)
            return float(dot_product / (norm_v1 * norm_v2))
        except Exception as e:
            self.logger.debug(f"Similarity Calculation Issue: {e}")
            return 0.0
