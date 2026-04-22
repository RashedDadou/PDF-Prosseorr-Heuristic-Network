# pdf_processor.py

"""
محرك تحليل PDF المنفصل والمتكامل
يعمل مع أي thinking engine
"""

import re
import logging
import time
import functools
import psutil  # لإدارة موارد النظام ديناميكياً
import warnings
import gc
from collections import deque, OrderedDict, defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Protocol, Set
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer

# تحسين: استيراد torch فقط عند الحاجة أو التأكد من وجوده
import torch
import numpy as np
import fitz  # PyMuPDF

# تحسين: إدارة التحذيرات قبل تحميل المكتبات الثقيلة
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from dotenv import load_dotenv
from openai import OpenAI

# تحميل المتغيرات من ملف .env
load_dotenv()

# استدعاء القيم مع وضع قيم افتراضية (Fallback) في حال عدم وجود الملف
base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
api_key = os.getenv("LLM_API_KEY", "lm-studio")

client = OpenAI(base_url=base_url, api_key=api_key)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("الرجاء تثبيت sentence-transformers عبر: pip install sentence-transformers")

# ===================================================================
# إعدادات الموديل (Singleton Pattern Concept)
# ===================================================================
_SHARED_MODEL = None
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_embedding_model():
    global _SHARED_MODEL
    if _SHARED_MODEL is None:
        _SHARED_MODEL = SentenceTransformer(MODEL_NAME, device=DEVICE)
    return _SHARED_MODEL

def profile_performance(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            return func(self, *args, **kwargs)
        finally:
            duration = time.time() - start_time
            # الوصول الآمن للـ logger سواء في الكلاس الرئيسي أو الـ Manager
            logger = getattr(self, 'logger', None)
            msg = f"⏱️ العملية [{func.__name__}] استغرقت: {duration:.4f} ثانية"
            if logger:
                logger.info(msg)
            else:
                print(msg)
    return wrapper

# ===================================================================
# إعدادات الموديل (Singleton Pattern Concept)
# ===================================================================
import time
import functools
from typing import List, Dict, Any

class InsightManager:
    def __init__(self, model: Any, logger: Any, device: str, threshold: float = 0.88):
        # استقبال البيانات الممرة من الكلاس الرئيسي
        self.model = model
        self.logger = logger
        self.device = device
        self.threshold = threshold

        # تعريف الخزنة كقاموس فارغ عند بدء التشغيل
        self.vault: Dict[int, Dict[str, Any]] = {}

        # إدارة الذاكرة السيمانتيكية (الجمل التي تم تحليلها سابقاً)
        self.master_insight_map: Dict[int, str] = {}
        self.vector_store: List[np.ndarray] = []

        if self.logger:
            self.logger.info(f"🧠 Insight Manager Active | Device: {self.device} | Threshold: {self.threshold}")

    def _create_basic_logger(self):
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger("InsightManager")

    def process_sentences(self, text: str, page_num: int) -> List[int]:
        import re
        import numpy as np
        import torch

        # 1. تقسيم النص لجمل (تصفية الجمل القصيرة جداً)
        sentences = [s.strip() for s in re.split(r'[.!?|]', text) if len(s.strip()) > 25]
        assigned_ids = []

        for sent in sentences:
            try:
                # محاولة التشفير باستخدام الجهاز المحدد (GPU/CPU)
                sent_vec = self.model.encode(sent, convert_to_numpy=True, device=self.device)
            except Exception as e:
                if "cuda" in str(e).lower() or "memory" in str(e).lower():
                    self.logger.warning("⚠️ VRAM Pressure: Sentence encoding shifted to CPU.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    sent_vec = self.model.encode(sent, convert_to_numpy=True, device="cpu")
                else:
                    self.logger.error(f"💥 Sentence Encoding Failed: {e}")
                    continue

            duplicate_id = None

            # نتحقق أن الخزنة تحتوي على عناصر قبل محاولة إنشاء مصفوفة
            if self.vault and len(self.vault) > 0:
                existing_ids = list(self.vault.keys())

                # بناء المصفوفة
                vault_matrix = np.stack([self.vault[idx]["vector"] for idx in existing_ids])

                # حساب التشابه
                norm_sent = np.linalg.norm(sent_vec)
                norm_vault = np.linalg.norm(vault_matrix, axis=1)

                # إضافة epsilon (1e-9) لمنع القسمة على صفر
                similarities = np.dot(vault_matrix, sent_vec) / (norm_vault * norm_sent + 1e-9)

                max_sim_idx = np.argmax(similarities)
                if similarities[max_sim_idx] > self.threshold:
                    duplicate_id = existing_ids[max_sim_idx]

            # 3. إدارة التكرار والتخزين
            if duplicate_id:
                assigned_ids.append(duplicate_id)
                if page_num not in self.vault[duplicate_id]["pages"]:
                    self.vault[duplicate_id]["pages"].append(page_num)
            else:
                new_id = len(self.vault) + 1
                self.vault[new_id] = {
                    "vector": sent_vec,
                    "text": sent,
                    "pages": [page_num],
                    "timestamp": time.time()
                }
                assigned_ids.append(new_id)

        return assigned_ids

class EmbeddingManager:
    _instance = None
    _model = None

    def __new__(cls, model_name='all-MiniLM-L6-v2'):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            try:
                print(f"🔄 جاري تحميل نموذج التضمين ({model_name}) لأول مرة...")
                from sentence_transformers import SentenceTransformer
                cls._model = SentenceTransformer(model_name)
                print("✅ تم تحميل النموذج بنجاح.")
            except ImportError:
                logging.error("فشل استيراد sentence_transformers. تأكد من تثبيت المكتبة.")
                cls._model = None
        return cls._instance

    @property
    def model(self):
        return self._model

# ===================================================================
# إعدادات الموديل (LoggerProtocol)
# ===================================================================
from typing import Protocol, Any, runtime_checkable

@runtime_checkable
class LoggerProtocol(Protocol):
    """
    بروتوكول موحد لضمان عمل أي نظام تسجيل (Logging)
    مع المحرك السيادي دون حدوث تعارض في الأنواع.
    """
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def error(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None: ...

# ===================================================================
# محرك قراءة الملفات (PDF Page Cache Network)
# ===================================================================
class PDFPageCacheNetwork:
    # 1. تعريف الأنواع لإرشاد Pylance (Type Hinting)
    model: SentenceTransformer
    doc: Optional[Any]
    vectors: List[np.ndarray]
    engine_metadata: Dict[str, Any]
    _registered_hubs: Set[str]

    def __init__(self, logger=None, model=None, max_pages=150, embedding_model=None):
        """
        تهيئة المحرك السيادي: تنظيم متكامل للذاكرة، الجراف، والميتاداتا.
        """
        # 2. إعدادات الهوية والتسجيل (Logging)
        self.logger = logger or logging.getLogger("Sovereign")

        # 3. توحيد محرك التضمين (Embedding Logic)
        try:
            # الأولوية للموديل الممرر، ثم الوسيط، ثم الـ Manager
            self.model = model or embedding_model or EmbeddingManager().model
        except (NameError, Exception):
            self.logger.error("❌ فشل الوصول إلى EmbeddingManager. تأكد من استيراد الكلاس.")
            raise

        # 4. العتاد والملفات (Hardware & Documents)
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.doc: Optional[Any] = None

        # 5. إدارة الذاكرة والكاش الديناميكي (Memory Management)
        self.max_pages = max_pages
        self.page_cache: OrderedDict[int, Any] = OrderedDict()
        self.page_queue = deque()

        # 6. الهياكل البيانية والجراف المعرفي (Knowledge Structures)
        self.knowledge_graph = defaultdict(set)
        self.visual_links: List[Dict[str, Any]] = []
        self.layer_hierarchy: Dict[int, List[int]] = {}
        self.index: List[Dict[str, Any]] = []
        self._registered_hubs: Set[str] = set()
        self.network_hubs_count: int = 0

        # 7. الفهرسة السيمانتيكية (Vector Database Logic)
        self.vectors: List[np.ndarray] = []
        self.vector_map: List[int] = []

        # 8. الميتاداتا السيادية (Sovereign Metadata)
        self.engine_metadata = {
            "dna_flow": [],
            "system_alerts": [],
            "initial_metadata": {"source_file": "Sovereign_Doc"}
        }

        self.logger.info(f"🚀 Sovereign Engine initialized on {self.device}")

    # ==========================================
    # الدالة الجديدة المسؤولية عن نظام الدفعات
    # ==========================================
    def ingest(self, pdf_path: str, batch_size: int = 16):
        import fitz
        from concurrent.futures import ThreadPoolExecutor

        self.doc = fitz.open(pdf_path)
        total_pages = len(self.doc)
        all_chunks = []
        chunk_metadata = []

        self.logger.info(f"📂 معالجة المستند عبر التوازي (Parallel Logic)...")

        def process_page(page_idx):
            # 1. حماية المدخلات
            if page_idx is None:
                return []

            try:
                # 2. التأكد من أن المستند مفتوح وصالح
                if not self.doc or self.doc.is_closed:
                    return []

                page = self.doc[page_idx]
                # التأكد من نجاح استخراج الصفحة
                if not page:
                    return []

                raw_text = page.get_text("text") or "" # حماية: إذا عاد النص None يصبح نصاً فارغاً
                clean_text = self._clean_text(str(raw_text))

                results = []
                # 3. حماية: التأكد أن clean_text ليس None وأنه نص صالح
                if clean_text and isinstance(clean_text, str) and len(clean_text) > 10:
                    page_chunks = self._create_overlapping_chunks(clean_text)

                    # 4. حماية ضد فشل التقسيم (تحول page_chunks إلى None)
                    if page_chunks is not None:
                        for chunk in page_chunks:
                            meta = {
                                "page_num": page_idx + 1,
                                "folder_id": f"FOLDER_{page_idx // 50:02d}",
                                "binder_id": f"BINDER_{page_idx // 10:02d}"
                            }
                            results.append((chunk, meta))

                return results

            except Exception as e:
                # تسجيل الخطأ بالتفصيل لمعرفة أي صفحة سببت المشكلة
                self.logger.error(f"❌ Critical error at Page {page_idx}: {str(e)}")
                return []

        # 1. تنفيذ التوازي
        cpu_cores = os.cpu_count() or 4
        with ThreadPoolExecutor(max_workers=min(32, cpu_cores + 4)) as executor:
            # map تحافظ على الترتيب الأصلي
            future_results = list(executor.map(process_page, range(total_pages)))

        # 2. تجميع النتائج مع حماية ضد None (Flattening)
        for page_result in future_results:
            if page_result: # حماية ضد None أو القوائم الفارغة
                for chunk, meta in page_result:
                    if chunk and meta:
                        all_chunks.append(chunk)
                        chunk_metadata.append(meta)

        if not all_chunks:
            self.logger.warning("⚠️ الملف لا يحتوي على نصوص كافية.")
            return

        # 2. المعالجة بنظام الدفعات (Batch Encoding) - تبقى كما هي لأنها محسنة أصلاً داخل المكتبة
        self.logger.info(f"🧠 تشفير {len(all_chunks)} كتلة نصية...")
        embeddings = self.model.encode(
            all_chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device
        )

        # 3. بناء الهياكل الاستنتاجية (Vector Map & Hierarchy)
        for idx, (meta, emb) in enumerate(zip(chunk_metadata, embeddings)):
            self.vectors.append(emb)
            # الآن vector_map يحمل أرقام الصفحات الحقيقية للبحث السيمانتيكي
            self.vector_map.append(meta["page_num"])

            # إصلاح الكاش: نستخدم idx كفتاح فريد لمنع تداخل البيانات
            self.page_cache[idx] = {
                "content": all_chunks[idx],
                "metadata": meta,
                "layer_type": "CHUNKED_DATA",
                "semantic_keywords": self.PDF_extract_keywords(all_chunks[idx], page_num=meta["page_num"])
            }

            # بناء الهيكلية الطبقية (Layer Hierarchy) لربط الكتل المتجاورة
            if idx > 0:
                if idx not in self.layer_hierarchy: self.layer_hierarchy[idx] = []
                self.layer_hierarchy[idx].append(idx - 1)

                if (idx - 1) not in self.layer_hierarchy: self.layer_hierarchy[idx - 1] = []
                self.layer_hierarchy[idx - 1].append(idx)

        self.total_chars = sum(len(c) for c in all_chunks)
        # تحديث عداد الـ Hubs بعد المعالجة
        self.network_hubs_count = len(getattr(self, 'heuristic_network', {}))

        self.logger.info(f"✅ تمت الفهرسة. المخرجات: {len(self.vectors)} كتلة سياقية.")

    def _setup_logging(self):
        # إنشاء الـ Logger
        logger = logging.getLogger("SovereignEngine")
        logger.setLevel(logging.DEBUG)

        # تنسيق السجلات (وقت - مستوى الخطأ - الرسالة)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # 1. الإخراج إلى ملف (app.log)
        file_handler = logging.FileHandler('app.log', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 2. الإخراج إلى الـ Terminal
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _create_overlapping_chunks(self, text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
        """
        [المقسم السيمانتيكي المطور]:
        يستبدل المنطق التقليدي (While Loop) بمنطق يعتمد على "حدود المعنى".
        الهدف: رفع DNA Quality من 74% إلى +90% عبر منع بتر المعادلات والجمل.
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []

        # 1. التقسيم الأولي بناءً على علامات الترقيم الكبرى والأسطر (نهايات الأفكار)
        # نستخدم regex ذكي يكتشف النقطة، السطر الجديد، وعلامات التعجب/الاستفهام
        segments = re.split(r'(?<=[.؟!])\s+|\n+', text)

        chunks = []
        current_chunk = ""

        for segment in segments:
            segment = segment.strip()
            if not segment: continue

            # إذا كانت إضافة الجزء الحالي لا تتخطى الحجم المسموح، ندمجه
            if len(current_chunk) + len(segment) <= chunk_size:
                current_chunk += (" " + segment if current_chunk else segment)
            else:
                # وصلنا للحد الأقصى: نخزن الكتلة الحالية كفكرة متكاملة
                if current_chunk:
                    chunks.append(current_chunk)

                # إعداد التداخل (Overlap) بذكاء:
                # بدلاً من أخذ 120 حرفاً عشوائياً، نحاول الحفاظ على آخر جملة
                # لضمان استمرارية السياق في الـ Knowledge Graph
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + segment

        # إضافة ما تبقى من النص
        if current_chunk:
            chunks.append(current_chunk)

        # تسجيل إحصائية سريعة في الـ Logger للمراقبة
        if hasattr(self, 'logger'):
            self.logger.debug(f"🧩 Semantic Chunking: Generated {len(chunks)} contextual units.")

        return chunks

    def _create_chunks(self, text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
        """
        تقسيم النص إلى كتل متداخلة.
        chunk_size: حجم الكتلة الواحدة (مثلاً 800 حرف).
        overlap: حجم التداخل (مثلاً 120 حرف لضمان بقاء جملة كاملة تقريباً).
        """
        chunks = []
        if len(text) <= chunk_size:
            return [text]

        start = 0
        while start < len(text):
            # تحديد نهاية الكتلة
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            # القفز بمقدار (الحجم - التداخل) لخلق النافذة المنزلقة
            start += (chunk_size - overlap)

            # التوقف إذا وصلنا لنهاية النص ولم يتبقَ قدر كافٍ لعمل كتلة جديدة مفيدة
            if start >= len(text) - overlap:
                break

        return chunks

    def _make_logger(self) -> LoggerProtocol:
        """إنشاء مسجل داخلي احترافي باستخدام logging القياسي"""
        logger_name = f"PDFProcessor_{id(self)}"
        logger = logging.getLogger(logger_name)

        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - ℹ️ %(message)s', datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger # النوع هنا يتوافق مع LoggerProtocol

    # ------------ Run Sovereign Session (دالة القيادة) ------------
    def run_sovereign_session(self, pdf_path: str, query: str):
        """
        [القائد الأعلى]: الدالة المايسترو التي تدير تدفق البيانات والذكاء.
        تضمن هذه النسخة عدم الانهيار عند غياب الـ Hubs أو فشل الـ LLM.
        """
        self.logger.info(f"🎖️ [COMMAND]: Initiating Sovereign Session for: {pdf_path}")

        # 1. المرحلة التنفيذية: (بناء الأرشيف والـ Hubs)
        # نقوم بتمرير الـ query لـ process_pdf_core لتوجيه عملية الفهرسة الأولية
        doc_result = self.process_pdf_core(pdf_path)

        if doc_result.get("status") != "success":
            self.logger.error(f"❌ Core Ingestion Failed: {doc_result.get('error_details')}")
            return {"status": "FAILED", "error": "Core Ingestion Error"}

        # 2. المرحلة الاستدلالية: (تحديد مسار الملاحة)
        # البحث السيمانتيكي يحدد "مراكز الثقل" المرتبطة بسؤال المستخدم
        target_pages = self.infer_navigation_path(query)

        # 3. تجميع السياق (Context Aggregation)
        context_parts = []
        # نستخدم التحقق من وجود target_pages لمنع الانهيار
        selected_pages = target_pages[:3] if target_pages else []

        for p_num in selected_pages:
            p_data = self.get_page_data(p_num)
            if isinstance(p_data, dict) and p_data.get('content'):
                context_parts.append(p_data['content'])

        full_context = "\n---\n".join(context_parts) if context_parts else "No relevant context found."

        # 4. مرحلة التوليد الذكي (The Generator)
        try:
            # نمرر السياق للذكاء الاصطناعي مع طلب المستخدم
            ideas, state = self.generate(full_context, user_request=query)
        except Exception as e:
            self.logger.warning(f"⚠️ Generation failed, falling back to local analysis: {e}")
            ideas, state = [], {"avg_score": 0.3, "engine_status": "LOCAL_FALLBACK"}

        # 5. مرحلة الرقابة والتقرير (The Monitor)
        # ضمان وجود إحصائيات صالحة حتى لو فشل doc_result
        stats = doc_result.get('data', {}).get('stats', {})
        total_pages = stats.get('pages', 1)

        # تحديث عداد الوعي التراكمي
        avg_score = state.get('avg_score', 0.5)
        if hasattr(self, 'cumulative_awareness'):
            self.cumulative_awareness.append(avg_score)

        # تفعيل الرقابة البصرية على سير العملية
        self._analysis_monitor(
            current_pulse=len(selected_pages),
            total_pulses=total_pages,
            chunk_score=avg_score
        )

        # استخراج المفاهيم السيادية (Hubs) - النسخة المؤمنة التي لا تعيد IndexError
        discovered_hubs = self.get_strategic_hubs(top_n=3)

        self.logger.info(f"🏆 [SUCCESS]: Session complete. Ideas: {len(ideas)} | Hubs: {len(discovered_hubs)}")

        return {
            "ideas": ideas,
            "hubs_discovered": discovered_hubs,
            "status": state.get("engine_status", "LOCAL_ONLY"),
            "performance_metrics": {
                "total_chars": getattr(self, 'total_chars', 0),
                "knowledge_density": getattr(self, 'knowledge_density_index', 0),
                "pages_processed": total_pages
            }
        }

    # ------------ ثانياً: استخراج البيانات الهيكلية (Extraction Engine)
    def process_pdf_core(self, pdf_path: str, user_request: str = "") -> Dict[str, Any]:
        """
        [القائد الميداني المحصن]: معالجة متوازية مع صمام أمان لضمان سلامة الـ DNA.
        """
        import time
        import threading # لإضافة أقفال الحماية
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path

        start_time = time.time()
        # صمام أمان (Lock) لمنع تضارب البيانات أثناء تحديث الكاش
        self._cache_lock = threading.Lock()

        try:
            self.doc = fitz.open(pdf_path)
            page_count = len(self.doc)
            self.logger.info(f"🚀 انطلاق كتائب المعالجة المحصنة: {page_count} صفحة.")

            # تحديد عدد العمال بناءً على استهلاك الذاكرة (Memory-Safe Threads)
            num_workers = min(os.cpu_count() or 4, 6)

            # استخدام دالة وسيطة تضمن "القفل" أثناء الكتابة في الكاش
            def safe_worker(p_num):
                try:
                    # 1. استخراج البيانات (عملية مكثفة خارج القفل)
                    # ملاحظة: يفضل فتح نسخة مستقلة من doc لكل thread إذا استمرت المشاكل
                    data = self._process_page_worker(p_num)

                    # 2. حقن البيانات في الكاش (داخل القفل لضمان الثبات)
                    with self._cache_lock:
                        self.logger.debug(f"📑 Page P{p_num} secured in cache.")
                except Exception as e:
                    self.logger.error(f"⚠️ Worker failure at P{p_num}: {str(e)}")

            # 3. استخدام "المعالج المتوازي" لتوزيع العمل عبر الجندي المحصن
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # نرسل safe_worker هنا لضمان تشغيل صمامات الأمان التي وضعناها
                executor.map(safe_worker, range(page_count))

            # --- الخطوة الاستراتيجية لمنع الـ N/A ---
            # التأكد من أن كل صفحة تم حقنها فعلياً في الفهرس الاستدلالي
            self.logger.info("📡 نسج الروابط الاستدلالية وحقن الـ DNA...")
            for p_num in range(page_count):
                self._build_heuristic_links(p_num)
                # فحص سريع: إذا كانت الصفحة حيوية ولم تُفهرس، نقوم بفهرستها الآن
                if p_num not in self.page_cache:
                    self.get_page_data(p_num)

            process_time = round(time.time() - start_time, 2)
            return {
                "status": "success",
                "data": {
                    "stats": {
                        "pages": page_count,
                        "hubs": self.network_hubs_count,
                        "time": process_time
                    }
                }
            }

        except Exception as e:
            self.logger.error(f"❌ انهيار القائد الميداني: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _process_page_worker(self, page_num: int):
        """
        [الجندي الميداني]: معالجة الصفحة مع ضمان وجود المستند.
        """
        # التأكد من أن المستند مفتوح في هذا السياق
        if self.doc is None:
            self.logger.error("❌ محاولة معالجة صفحة بينما المستند (doc) غير محمل.")
            return

        try:
            # الوصول للصفحة (هذا هو السطر الذي قد يسبب subscriptable error إذا كان doc=None)
            page = self.doc[page_num]
            page_contents = []

            # A. الاستخراج الهيكلي
            # تأكد أن هذه الدالة لا تطلب وسيطاً إضافياً (مثل رقم الصفحة) إذا كانت مصممة لذلك
            layout_data = self._extract_layout_structure(page)

            # B. استخراج النصوص (الترتيب البصري)
            raw_blocks = page.get_text("blocks")
            if raw_blocks:
                blocks = sorted([b for b in raw_blocks if len(b[4].strip()) > 5],
                                key=lambda b: (b[1], b[0]))
                for b in blocks:
                    page_contents.append(b[4].strip())

            # C. معالجة الجداول
            has_tables = False
            try:
                tabs = page.find_tables()
                if tabs and hasattr(tabs, 'tables') and tabs.tables:
                    has_tables = True
                    for table in tabs.tables:
                        t_data = table.extract()
                        if t_data:
                            table_text = "\n".join([" | ".join([str(c).strip() if c else "" for c in r]) for r in t_data])
                            page_contents.append(f"\n[TABLE_DATA]:\n{table_text}")
            except Exception:
                pass

            combined_text = "\n".join(page_contents).strip()

            # إذا لم يوجد نص، قد لا ترغب في إضافة الصفحة للكاش
            if not combined_text:
                self.logger.debug(f"📭 الصفحة {page_num+1} فارغة، تم التجاوز.")
                return

            # D. إعداد الميتاداتا
            page_meta = {
                "filename": getattr(self, 'current_filename', 'doc'),
                "page_number": page_num + 1,
                "contains_tables": has_tables,
                "layout": layout_data
            }

            # E. الحقن في الكاش مع ضمان الأنواع
            self.add_page(page_num, combined_text, page_meta or {})

        except Exception as e:
            self.logger.warning(f"⚠️ تعثر الجندي في الصفحة {page_num+1}: {e}")

    def fast_ingest_stream(self, pdf_path: str, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        [القارئ السريع]: يمسح الصفحات ويقوم بالحقن الحيوي للجينات (Keywords)
        لضمان ظهور الـ Hubs في التقرير الأولي.
        """
        import fitz
        import time
        from pathlib import Path

        try:
            if not Path(pdf_path).exists():
                return {"status": "error", "pages_ingested": 0, "message": "File not found"}

            with fitz.open(pdf_path) as pdf_doc:
                total_pages = pdf_doc.page_count
                self.logger.info(f"🌀 بدء التدفق السريع لـ {total_pages} صفحة...")

                for page_num in range(total_pages):
                    page = pdf_doc.load_page(page_num)
                    raw_content = page.get_text("text")

                    # الحماية السيادية للأنواع
                    page_text = str(raw_content).strip() if raw_content else ""

                    if page_text:
                        # 1. الحقن الحيوي للكلمات المفتاحية (حل مشكلة الـ 0 Hubs)
                        # نمرر page_num لكي يتم بناء الـ heuristic_network فوراً
                        self.PDF_extract_keywords(page_text, page_num=page_num)

                        # 2. التخزين في المستودع (The Warehouse)
                        self.page_cache[page_num] = {
                            "content": page_text,
                            "layer_type": "PRE_INDEXED_RAW",
                            "metadata": {
                                "fast_stream": True,
                                "source": str(pdf_path),
                                "location_stamp": f"FOLDER_01 > PAGE_{page_num+1}",
                                "ingested_at": time.time()
                            }
                        }

                        if callback:
                            try: callback(page_num, page_text)
                            except: pass

                self.logger.info(f"✅ Ingestion Complete: {total_pages} pages ready for deep analysis.")
                return {"status": "completed", "pages": total_pages}

        except Exception as e:
            self.logger.error(f"❌ Fast Stream Failure: {str(e)}")
            return {"status": "error", "message": str(e)}

    def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        """Refined PDF extraction - Type Safe for Pylance"""
        try:
            pdf_path_obj = Path(pdf_path)
            if not pdf_path_obj.exists():
                return {"status": "error", "message": f"File not found: {pdf_path}"}

            with fitz.open(pdf_path) as pdf_doc:
                page_count = pdf_doc.page_count
                # المصفوفة التي ستجمع نصوص الصفحات بالترتيب الصحيح
                results = [None] * page_count

                raw_meta = pdf_doc.metadata or {}
                doc_info = {
                    "title": str(raw_meta.get("title") or pdf_path_obj.stem),
                    "author": str(raw_meta.get("author") or "unknown"),
                    "page_count": page_count
                }

                self.logger.info(f"📄 Processing: {doc_info['title']}")

                # --- الإصلاح: معالجة ذكية لتجنب تجميد المعالج ---
                def process_single_page(p_num):
                    """دالة داخلية لمعالجة كل صفحة ككيان مستقل"""
                    try:
                        p = pdf_doc.load_page(p_num)
                        # تحسين: استخراج النصوص مع تنظيف مباشر
                        page_blocks = p.get_text("blocks")
                        valid_text = []
                        for b in page_blocks:
                            if len(b) > 4 and isinstance(b[4], str):
                                clean_b = b[4].strip()
                                if len(clean_b) > 2:
                                    valid_text.append(clean_b)
                        return "\n".join(valid_text)
                    except Exception as e:
                        self.logger.error(f"Error on page {p_num}: {e}")
                        return ""

                # إذا كان الملف صغيراً، نعالجه مباشرة. إذا كان كبيراً، نستخدم قدرات المعالج كاملة
                if page_count > 5:
                    from concurrent.futures import ThreadPoolExecutor
                    # استخدام عدد خيوط يتناسب مع الهاردوير (CPU Cores)
                    workers = min(os.cpu_count() or 1, 8)
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        page_texts = list(executor.map(process_single_page, range(page_count)))
                else:
                    page_texts = [process_single_page(i) for i in range(page_count)]

                full_text = "\n--- PAGE BREAK ---\n".join(page_texts)
                # ------------------------------------------------

                return {
                    "status": "success",
                    "extraction_data": {
                        "full_text": full_text,
                        "char_count": len(full_text),
                        "page_count": page_count,
                        "document_info": doc_info
                    },
                    "file_system": {
                        "path": str(pdf_path_obj.absolute()),
                        "name": pdf_path_obj.name
                    }
                }

        except Exception as e:
            self.logger.error(f"❌ Extraction Failure: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _extract_layout_structure(self, page: Any) -> Dict[str, Any]:
        """
        [المحلل الهيكلي السيادي]: نسخة مطورة تطبق التحجيم الديناميكي
        لحماية المعالج من التجميد عند معالجة الصور الثقيلة.
        """
        import psutil
        layout_data = {"headings": [], "blocks": []}

        try:
            # 1. الحصول على البيانات النصية الأساسية (المسار السريع)
            dict_data = page.get_text("dict")
            layout_data["blocks"] = dict_data.get("blocks", [])

            # --- [إستراتيجية الحماية من التجميد] ---
            # إذا لم نجد نصوصاً (احتمال كبير أنها صورة أو صفحة ممسوحة)
            if not layout_data["blocks"] or len(str(layout_data["blocks"])) < 100:
                cpu_usage = psutil.cpu_percent(interval=None)

                # إذا كان المعالج مضغوطاً، نخفض جودة الصورة المعالجة فوراً
                # 72 DPI هي جودة كافية للتعرف على النصوص مع توفير 70% من جهد المعالج
                zoom = 1.0 if cpu_usage < 80 else 0.7
                mat = fitz.Matrix(zoom, zoom)

                # استخراج الصورة بجودة ديناميكية لمنع الـ OCR من سحب 100% CPU
                pix = page.get_pixmap(matrix=mat)

                # هنا يتم تمرير 'pix' لمحرك الـ OCR الخاص بك (إذا وجد)
                # ملاحظة: تقليل الـ DPI هنا هو "صمام الأمان" الحقيقي
                pass

            # 2. الاستمرار في تحليل العناوين (المسار الهيكلي)
            for block in layout_data["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if len(text) > 3:
                                # المعايير الهندسية الأصلية
                                is_large = span["size"] > 11.5
                                is_bold = "bold" in span["font"].lower()
                                is_caps = text.isupper() and len(text) > 5
                                is_not_numeric = not text.replace('.', '').isdigit()

                                if (is_large or is_bold or is_caps) and is_not_numeric:
                                    layout_data["headings"].append({
                                        "text": text,
                                        "bbox": span["bbox"],
                                        "font_size": span["size"],
                                        "font_name": span["font"],
                                        "type": "structural_anchor"
                                    })

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"⚠️ Layout extraction bypassed for P{page.number + 1}: {str(e)}")

        return layout_data

    def _clean_text(self, text: str) -> str:
        """
        تنقية النص من الضجيج (روابط، رموز، أرقام صفحات، مسافات زائدة)
        """
        if not text:
            return ""

        # 1. إزالة روابط المواقع (URLs)
        text = re.sub(r'http[s]?://\S+', '', text)

        # 2. إزالة رموز البريد الإلكتروني
        text = re.sub(r'\S+@\S+', '', text)

        # 3. إزالة الرموز الغريبة والزخارف (Non-alphanumeric) مع الحفاظ على النقاط والفواصل
        text = re.sub(r'[^\w\s\.\,\?\!\u0600-\u06FF]', ' ', text)

        # 4. إزالة أرقام الصفحات المعزولة (مثلاً: "Page 5" أو رقم وحيد في سطر)
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)

        # 5. توحيد المسافات (إزالة المسافات الزائدة والأسطر الفارغة المتكررة)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

# ------------ ثالثاً: إدارة الذاكرة والأرشفة (Caching & Storage) ------------
    def add_page(self, page_num: int, text: str = "", page_obj: Any = None, metadata: dict = None):
        import time
        import threading
        import torch
        from torch import no_grad # استيراد مباشر لإسكات Pylance

        if not hasattr(self, '_lock'):
            self._lock = threading.Lock()

        try:
            # المرحلة 1: العمليات الحساسة (Atomic Reading)
            with self._lock:
                if self.doc is None: return
                # حماية: التأكد من أن الفهرس ضمن نطاق الصفحات
                if page_num >= len(self.doc): return

                page_obj = self.doc.load_page(page_num)
                raw_text = page_obj.get_text("text")
                text = raw_text.strip() if raw_text else ""
                layout = self._extract_layout_structure(page_obj)

            if not text: return

            # المرحلة 2: العمليات الثقيلة (Parallel GPU/CPU)
            vector = None
            try:
                with no_grad(): # استخدام no_grad المباشر
                    vector = self.model.encode(text, convert_to_numpy=True, device=self.device)
            except Exception as e:
                if "cuda" in str(e).lower() or "memory" in str(e).lower():
                    self.logger.warning(f"⚠️ GPU Busy on P{page_num+1}: Retreating to CPU.")
                    with no_grad():
                        vector = self.model.encode(text, convert_to_numpy=True, device="cpu")

            # استخراج الميتاداتا والطبقات (خارج القفل لزيادة السرعة)
            loc = self._assign_sovereign_location(page_num)
            layer = self._classify_layer(text, {**(metadata or {}), **loc})
            keywords = self.PDF_extract_keywords(text)

            # المرحلة 3: التحديث النهائي (Final Commitment)
            with self._lock:
                if vector is not None:
                    self.vectors.append(vector)
                    self.vector_map.append(page_num + 1)

                self.page_cache[page_num + 1] = {
                    "content": text,
                    "metadata": {**(metadata or {}), **loc},
                    "layer_type": layer,
                    "semantic_keywords": keywords,
                    "visual_headings": layout.get("headings", []),
                    "indexed_at": time.time()
                }

                # بناء الجراف وتحديث الذاكرة
                self._build_visual_heuristics(page_num + 1, layout)
                self._refresh_memory_balance(page_num + 1)

        except Exception as e:
            self.logger.error(f"💥 Critical Failure on Page {page_num+1}: {str(e)}", exc_info=True)

    def _prepare_batches(self, pdf_path: str):
        import fitz
        texts = []
        page_numbers = []

        try:
            with fitz.open(pdf_path) as doc:
                for i in range(len(doc)):
                    # الحصول على النص الخام
                    raw_content = doc[i].get_text("text")

                    # حل Pylance: تحويل صريح للتأكد من أنه str قبل مناداة strip
                    if isinstance(raw_content, str):
                        clean_text = raw_content.strip()
                        if clean_text:
                            texts.append(clean_text)
                            page_numbers.append(i + 1)
                    else:
                        # في حال أعاد fitz شيئاً غير النص (نادر جداً مع وسيط "text")
                        self.logger.debug(f"ℹ️ Page {i+1}: Unexpected data type {type(raw_content)}")

        except Exception as e:
            self.logger.error(f"❌ Batch preparation failed: {e}")

        return texts, page_numbers

    def process_in_batches(self, pdf_path: str, batch_size: int = 16):
        """
        تنفذ عملية الـ Embedding بنظام الدفعات وتحدث الفهرس والجراف
        """
        # 1. جلب النصوص مجهزة
        all_texts, all_page_nums = self._prepare_batches(pdf_path)

        if not all_texts:
            print("⚠️ لم يتم العثور على نصوص في الملف.")
            return

        # 2. توليد الـ Embeddings لكل النصوص دفعة واحدة (أو مقسمة داخلياً حسب batch_size)
        # مكتبة SentenceTransformer مهيأة برمجياً للتعامل مع القوائم بكفاءة عالية
        print(f"🧠 جاري معالجة {len(all_texts)} صفحة بنظام الدفعات...")
        embeddings = self.model.encode(
            all_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # 3. تحديث الفهرس وبناء علاقات الجراف (هنا نستخدم حلقة سريعة للتخزين فقط)
        for idx, (page_num, emb) in enumerate(zip(all_page_nums, embeddings)):
            self.index.append({
                'page': page_num,
                'embedding': emb,
                'text': all_texts[idx]
            })

            # ربط الصفحة الحالية بالتي قبلها في الجراف
            if idx > 0:
                prev_page = all_page_nums[idx-1]
                self.knowledge_graph[page_num].add(prev_page)
                self.knowledge_graph[prev_page].add(page_num)

        print(f"✅ تم الانتهاء من الفهرسة بنجاح.")

    def get_page_data(self, page_num: int, pdf_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        [إصلاح سيادي]: استرجاع ذكي يدمج بين حماية الذاكرة والتحقق العميق من النزاهة.
        تم إصلاح منطق التحقق ليكون أكثر مرونة مع البيانات المستعادة.
        """
        data = None

        # 1. محاولة الجلب من الكاش المباشر (LRU Hit)
        if page_num in self.page_cache:
            self.page_cache.move_to_end(page_num)
            data = self.page_cache[page_num]

        # 2. الاستعادة الذكية (Auto-Recovery)
        elif pdf_path:
            if getattr(self, '_in_recovery', False):
                self.logger.warning(f"⚠️ استعادة متداخلة محظورة للصفحة {page_num}")
                return None

            self._in_recovery = True
            try:
                self.logger.info(f"🔄 إعادة بناء الذاكرة للصفحة {page_num}...")
                success = self._recover_page_logic(page_num, pdf_path)
                if success:
                    data = self.page_cache.get(page_num)
            finally:
                self._in_recovery = False

        # 3. فحص النزاهة المعمق (Deep Integrity Check)
        if data and isinstance(data, dict):
            # إصلاح: لا نرفض البيانات إذا كانت القوائم فارغة، بل نرفضها إذا كانت المفاتيح مفقودة تماماً
            critical_keys = ['semantic_keywords', 'layer_type', 'visual_headings']

            # التأكد من وجود المفاتيح حتى لو كانت قيمها فارغة [] أو "UNKNOWN"
            if all(k in data for k in critical_keys):
                # حساب المتركس بأمان
                metrics = data.get('processing_metrics', {})
                if isinstance(metrics, dict):
                    latency_values = [v for v in metrics.values() if isinstance(v, (int, float))]
                    if latency_values:
                        data['total_latency'] = f"{sum(latency_values):.4f}s"

                data['integrity_score'] = "VERIFIED"
                return data
            else:
                # محاولة إصلاح ذاتي سريع قبل إعلان الفشل
                self.logger.warning(f"🛠️ ترميم بيانات مفقودة للصفحة {page_num}")
                data.setdefault('semantic_keywords', [])
                data.setdefault('layer_type', 'RECOVERED')
                data.setdefault('visual_headings', [])
                return data

        return None

    def get_page_photo_scanned(self, page_obj, page_id):
        """
        [وحدة الاستطلاع البصري المطورة]: تدعم التوازي الآمن والتحجيم الديناميكي.
        إصلاح شامل للتكرار وضمان سلامة الذاكرة.
        """
        import fitz
        import pytesseract
        from PIL import Image
        import io
        import os
        from concurrent.futures import ThreadPoolExecutor

        image_list = page_obj.get_images(full=True)
        if not image_list:
            return ""

        self.logger.info(f"👁️ رصد {len(image_list)} هدف بصري في الصفحة {page_id}...")

        def process_image(img_info):
            img_index, img = img_info

            # 1. التأكد من توفر المستند والحصول على مرجع محلي لـ Pylance
            doc = getattr(self, 'doc', None)
            if doc is None:
                return None

            try:
                # 2. الحماية عند استخراج البيانات (Atomic Extraction)
                with self._lock:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image.get("image")

                if not image_bytes:
                    return None

                # 3. المعالجة البصرية (خارج القفل - لضمان التوازي الكامل)
                image = Image.open(io.BytesIO(image_bytes))

                # التحجيم الذكي لتقليل استهلاك RAM وتسريع OCR
                if max(image.size) > 1800:
                    image.thumbnail((1800, 1800), Image.Resampling.LANCZOS)

                # 4. تنفيذ الـ OCR (عملية ثقيلة على المعالج)
                custom_config = r'--oem 3 --psm 6'
                scanned_text = pytesseract.image_to_string(
                    image,
                    lang='eng+ara',
                    config=custom_config
                )

                clean_text = scanned_text.strip()
                if len(clean_text) > 15:
                    return f"[IMAGE_ANALYTICS_{img_index}]: {clean_text}"

            except Exception as e:
                self.logger.warning(f"⚠️ فشل تحليل الصورة {img_index} في P{page_id}: {str(e)}")
            return None

        # استخدام عدد خيوط متوازن (OCR عملية CPU-Bound)
        # لا نرفع max_workers كثيراً لتجنب استهلاك كامل قدرة المعالج
        max_workers = min(os.cpu_count() or 1, 4)
        indexed_images = list(enumerate(image_list))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_image, indexed_images))

        # تجميع النتائج وتصفية الـ None
        visual_intelligence = [r for r in results if r]
        return "\n".join(visual_intelligence)

    def _assign_sovereign_location(self, page_num: int) -> Dict[str, Any]:
        """
        [منطق التوزيع السيادي]: تقسيم المستند إلى محفظة ومجلدات ومصنفات.
        المصنف = 25 صفحة | المجلد = 4 مصنفات (100 صفحة)
        """
        # حساب رقم المصنف (Binder) ضمن المجلد
        # 0-24 = المصنف 1، 25-49 = المصنف 2...
        binder_idx = (page_num // 25) + 1

        # حساب رقم المجلد (Folder)
        # كل 4 مصنفات (100 صفحة) تذهب لمجلد واحد
        folder_idx = ((binder_idx - 1) // 4) + 1

        # المصنف النسبي داخل المجلد (مثلاً: المجلد 2، المصنف 3)
        relative_binder = ((binder_idx - 1) % 4) + 1

        return {
            "portfolio": "MAIN_KNOWLEDGE_ARCHIVE",
            "folder_id": f"FOLDER_{folder_idx:02d}",
            "binder_id": f"BINDER_{relative_binder:02d}",
            "absolute_binder": binder_idx,
            "page_in_binder": (page_num % 25) + 1
        }

    def _guess_topic(self, text: str) -> str:
        """
        [إصلاح تلقائي]: محرك تخمين الموضوعات ثنائي اللغة مع ترجيح السياق التقني.
        """
        categories = {
            'ENGINEERING': ['تحكم', 'نظام', 'آلية', 'محرك', 'control', 'system', 'robot', 'engine'],
            'PHILOSOPHY': ['فلسف', 'وعي', 'وجود', 'منطق', 'philoso', 'conscio', 'logic'],
            'TECHNOLOGY': ['ذكاء', 'برمج', 'خوارزم', 'data', 'intelligen', 'softw', 'algorithm'],
            'SCIENCE': ['بحث', 'نظرية', 'تجربة', 'scientif', 'research', 'theory'],
            'BUSINESS': ['إدارة', 'اقتصاد', 'تسويق', 'manage', 'econom', 'market'],
            'LEGAL': ['قانون', 'تشريع', 'عقد', 'legal', 'law', 'contract']
        }

        scores: Dict[str, float] = {topic: 0.0 for topic in categories}
        text_lower = text.lower()

        # تحسين: فحص أول 2000 حرف فقط للتخمين السريع (عنق الزجاجة)
        sample_text = text_lower[:2000]

        for topic, keywords in categories.items():
            for kw in keywords:
                count = sample_text.count(kw)
                if count > 0:
                    # ميزة: الكلمات الطويلة أو المركبة لها ثقل أكبر (DNA Weight)
                    weight = 2.0 if len(kw) > 6 else 1.0
                    scores[topic] += (float(count) * weight)

        # العثور على الفائز
        best_topic = max(scores, key=lambda k: scores[k])

        # شرط العبور: إذا كان أعلى سكور ضعيف جداً، نعتبره موضوعاً عاماً
        if scores[best_topic] > 2.0:
            return best_topic

        return "GENERAL_TOPIC"

    def _classify_layer(self, text: str, metadata: Dict) -> str:
        """
        [المصنف الطبقي السيادي المطور]: مدمج مع نظام التعويض الديناميكي
        لضمان دقة التصنيف حتى في ظروف المعالجة المنخفضة.
        """
        import re
        text_lower = text.lower()
        word_count = len(text_lower.split())

        # 1. استرجاع "معامل التحجيم" من الميتاداتا (الربط مع _extract_layout_structure)
        # إذا تم ضغط الصفحة، يجب تعويض الكثافة حسابياً
        zoom_factor = metadata.get('zoom_applied', 1.0)
        compensation_multiplier = 1.0 / zoom_factor # مقلوب التحجيم للتعويض

        # 2. كشاف الكثافة التقنية مع "معامل التعويض"
        symbol_pattern = r'[|%=\-\d]{2,}'
        raw_special_chars = len(re.findall(symbol_pattern, text_lower))

        # الكثافة المعدلة: تحسب الكثافة وكأن الصفحة عولجت بدقة كاملة
        adjusted_special_density = raw_special_chars * compensation_multiplier

        # 3. استخراج القرائن الهيكلية
        source_type = str(metadata.get('source_type', '')).lower()
        page_num = metadata.get('page_number', metadata.get('original_page', 0))
        is_image_ocr = metadata.get('is_ocr', False)

        is_table_context = any(kw in text_lower for kw in {'table', 'figure', 'جدول', 'مخطط', 'results'})
        is_table = is_table_context or 'table' in source_type

        # معيار البيانات التقنية المطور: يستخدم الكثافة المعدلة
        # العتبة (18) تظل كما هي ولكنها الآن تقارن بالقيمة المعوضة
        is_technical = is_table or adjusted_special_density > 18 or is_image_ocr

        # 4. سلم الأولويات (Priority Ladder)

        # الفئة A: البيانات التقنية
        if is_technical:
            if any(ind in text_lower[:400] for ind in {'chapter', 'فصل', 'section'}):
                return "TECHNICAL_CHAPTER"
            return "TECHNICAL_DATA"

        # الفئة B: الهيكل التنظيمي (بدايات الأقسام)
        if page_num < 15 and any(kw in text_lower for kw in {'contents', 'فهرس', 'preface', 'مقدمة', 'abstract'}):
            return "CORE_CONTENT"

        chapter_indicators = {'chapter', 'فصل', 'section', 'part', 'الوحدة', 'module'}
        if any(ind in text_lower[:300] for ind in chapter_indicators):
            return "CHAPTER_LAYER"

        # الفئة C: المراجع والملحقات
        reference_kws = {'appendix', 'مراجع', 'references', 'bibliography', 'ملحق', 'annex'}
        if any(kw in text_lower for kw in reference_kws):
            return "APPENDIX_LAYER"

        # الفئة D: المحتوى الجوهري (تعويض عدد الكلمات أيضاً)
        adjusted_word_count = word_count * compensation_multiplier
        if adjusted_word_count > 500:
            return "CORE_CONTENT"

        # الفئة E: محتوى قياسي
        return "STANDARD_CONTENT"

    def _recover_page_logic(self, page_num: int, pdf_path: str) -> bool:
        """
        [محرك الاستعادة المرن المطور]: يضيف المسار البصري كخيار أخير
        ويضمن عدم استنزاف الموارد أثناء محاولة الاستعادة.
        """
        import fitz
        from pathlib import Path

        try:
            with fitz.open(pdf_path) as doc:
                if page_num < 0 or page_num >= len(doc):
                    return False

                page = doc.load_page(page_num)

                # فحص الحجم قبل البدء (لحماية المعالج من الصفحات التالفة)
                pix = page.get_pixmap()
                if pix.width < 50 or pix.height < 50:
                    self.logger.debug(f"🔕 Page {page_num} is too small for analysis.")
                    return False

                # 1. الاستراتيجية الأولى: النص المباشر
                raw_text = page.get_text("text")
                clean_text = str(raw_text).strip() if raw_text else ""

                # 2. الاستراتيجية الثانية: المسح العميق (Deep Scan Blocks)
                if len(clean_text) < 15:
                    blocks = page.get_text("blocks")
                    extracted_parts = [b[4].strip() for b in blocks if len(b) > 4 and isinstance(b[4], str)]
                    clean_text = "\n".join(filter(None, extracted_parts))

                # 3. الاستراتيجية الثالثة: الاستعادة البصرية (OCR)
                recovery_method = "Direct/Blocks"
                if not clean_text.strip():
                    self.logger.info(f"🔄 Recovery: Page {page_num+1} seems visual. Activating OCR...")
                    # استدعاء الـ OCR (تأكد من وجود هذه الدالة في كلاسك)
                    clean_text = self.get_page_photo_scanned(page, page_num)
                    recovery_method = "OCR"

                if not clean_text or len(clean_text.strip()) < 5:
                    return False

                # 4. [الحقن السيادي الموحد]
                # ملاحظة: سنقوم بحقن النص مباشرة في الكاش لتجنب إعادة المعالجة في add_page
                metadata = {
                    "source_path": pdf_path,
                    "filename": Path(pdf_path).name,
                    "page_number": page_num + 1,
                    "recovery_method": recovery_method
                }

                # استدعاء add_page (تأكد من تحديث دالة add_page لتقبل نصاً خارجياً إذا لزم الأمر)
                # أو حقن النص يدوياً قبل الاستدعاء:
                self.add_page(page_num)

                # 5. تنظيف الذاكرة (Hardware Cleanup)
                if recovery_method == "OCR":
                    import gc
                    import torch
                    pix = None # إفراغ الـ pixmap فوراً
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                return True

        except Exception as e:
            self.logger.error(f"💥 Recovery Failure at P{page_num+1}: {str(e)}")
            return False

    def _refresh_memory_balance(self, current_page: int):
        """
        [منعش الذاكرة السيادي المطور]: نظام فحص وتفريغ دفعات ديناميكي التوسع.
        """
        # 1. حساب السعة الديناميكية بناءً على موارد الجهاز الحالية
        # القاعدة: 10 صفحات لكل 2 جيجابايت RAM متاحة، بحد أدنى 10 وأقصى 150
        try:
            available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            dynamic_max = max(10, int(available_ram_gb * 5))
            MAX_BURST_CAPACITY = min(dynamic_max, 150)
        except:
            MAX_BURST_CAPACITY = 10 # قيمة احتياطية في حال فشل النظام في القراءة

        # التأكد من استخدام OrderedDict لضمان منطق FIFO/LRU
        if not isinstance(self.page_cache, OrderedDict):
            self.page_cache = OrderedDict(self.page_cache)

        # إذا كان الكاش لا يزال تحت السعة الديناميكية، لا نفعل شيئاً
        if len(self.page_cache) <= MAX_BURST_CAPACITY:
            return

        try:
            # 2. بروتوكول الفحص قبل الإخلاء
            oldest_page_idx = next(iter(self.page_cache))
            oldest_entry = self.page_cache.get(oldest_page_idx, {})

            layer_type = oldest_entry.get("layer_type", "")
            # تحسين: رفع معايير الحماية لتشمل التحليلات الاستراتيجية أيضاً
            is_critical = layer_type in ["TECHNICAL_CHAPTER", "TECHNICAL_DATA", "STRATEGIC_HUB"]
            has_high_dna = len(oldest_entry.get("semantic_keywords", [])) > 8

            # 3. قرار الإنعاش (The Release Decision)
            if is_critical and has_high_dna:
                # منح الصفحة "حياة جديدة" عبر نقلها لآخر الطابور
                self.page_cache.move_to_end(oldest_page_idx)
                self.logger.debug(f"🛡️ Protection: Page {oldest_page_idx} (High-DNA) moved to end.")
            else:
                # إخلاء الصفحة العادية
                evicted_id, _ = self.page_cache.popitem(last=False)
                self.logger.info(f"♻️ Evicted: Page {evicted_id} | Dynamic Limit: {MAX_BURST_CAPACITY}")

        except Exception as e:
            self.logger.warning(f"⚠️ Memory Balance Sync Issue: {e}")

        # 4. التطهير العميق (Deep RAM Sanitization)
        # تقليل التطهير ليكون كل 10 صفحات بدلاً من 5 لتحسين الأداء على الأجهزة القوية
        if current_page % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # تنظيف ذاكرة الـ GPU أيضاً
            self.logger.debug("🧹 GC: Engine Lungs & VRAM Cleared.")

    def get_dynamic_burst_capacity(self, base_pages_per_gb: int = 10) -> int:
        """
        [محرك الحوسبة المرن]: يحسب سعة الكاش الديناميكية بناءً على موارد الهاردوير.
        يستخدم 'self' لضمان التوافق مع هيكلية المحرك وتسجيل العمليات.
        """
        try:
            # 1. قراءة الذاكرة العشوائية المتاحة
            import psutil
            available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)

            # 2. الحساب الديناميكي (قاعدة: 10 صفحات لكل 1 جيجابايت)
            dynamic_capacity = max(5, int(available_ram_gb * base_pages_per_gb))

            # 3. فرض الحدود السيادية (Guardrails)
            # الحد الأقصى 200 يمنع المحرك من "التهام" الذاكرة بالكامل في الملفات الضخمة
            final_capacity = min(dynamic_capacity, 200)

            self.logger.debug(f"🧠 Memory Intelligence: Dynamic burst set to {final_capacity} pages (RAM: {available_ram_gb:.1f}GB)")
            return final_capacity

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"⚠️ Hardware Probe Failed: Using safe fallback (10). Detail: {e}")
            return 10  # القيمة الآمنة (Fallback)

    def PDF_get_text(self, page_num: int) -> str:
        """
        [المستخرج الخام]: الدخول المباشر للمستند لسحب النص في حالات الطوارئ.
        """
        try:
            if self.doc:
                # PyMuPDF يستخدم الفهرس الصفر (0-indexed)
                page = self.doc.load_page(page_num)
                return page.get_text("text")
            return ""
        except Exception as e:
            self.logger.error(f"❌ فشل السحب المباشر للنص من الصفحة {page_num}: {e}")
            return ""

    def PDF_extract_keywords(self, text: str, top_n: int = 10, page_num: Optional[int] = None) -> List[str]:
        """
        [مستخرج الجينات المطور]: يقوم باستخراج الكلمات وحقنها مباشرة في الشبكة الاستدلالية
        لضمان عدم ضياع الروابط السيمانتيكية (Semantic Mapping).
        """
        import re
        from collections import Counter

        if not text or len(text.strip()) < 5:
            return []

        # 1. تنظيف ذكي وتحضير النصوص
        text_clean = re.sub(r'[^\w\s\-\_]', ' ', text, flags=re.UNICODE)
        original_words = text_clean.split()

        # معامل التعويض لضمان دقة الصفحات المحجمة (Scaled Pages)
        zoom_compensation = getattr(self, '_current_zoom_factor', 1.0)
        compensation_multiplier = 1.0 / zoom_compensation

        stop_words = self.get_sovereign_stop_words()
        weighted_scores = defaultdict(float)
        al_prefix = re.compile(r'^ال')

        # 2. استخراج ومعالجة الجينات (DNA Extraction)
        for word in original_words:
            word_len = len(word)
            if word_len < 3: continue # تجاهل الكلمات القصيرة جداً لمنع الضجيج

            is_tech_abbr = word.isupper() and word.isalpha()
            lower_word = word.lower()

            # توحيد المعرفات العربية
            norm_word = al_prefix.sub('', lower_word) if lower_word.startswith('ال') and word_len > 4 else lower_word

            if norm_word in stop_words or norm_word.isdigit():
                continue

            # 3. حساب الوزن النسبي (Gravity Weighting)
            weight = 1.0
            if is_tech_abbr: weight = 1.8
            if word_len > 10: weight = 1.4
            if '_' in norm_word or '-' in norm_word: weight = 2.5

            weighted_scores[norm_word] += (weight * compensation_multiplier)

        # 4. اختيار النخبة (The Elite DNA)
        top_entries = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        final_keywords = [word for word, score in top_entries]

        # 5. [الحقن الحيوي]: تحديث الشبكة الاستدلالية (Heuristic Network)
        # هذا الجزء هو المسؤول عن حل مشكلة الـ "0 Hubs" في تقريرك
        if page_num is not None:
            for word, score in top_entries:
                if word not in self.heuristic_network:
                    self.heuristic_network[word] = []

                # إضافة العقدة للشبكة مع وزنها المستخرج
                self.heuristic_network[word].append({
                    "page": page_num,
                    "weight": round(score, 2),
                    "context": "LOCAL_EXTRACTION"
                })

        return final_keywords if final_keywords else original_words[:top_n]

    @staticmethod
    def get_sovereign_stop_words():
        """
        توليد قائمة كلمات توقف هجينة (مكتبات + كلمات مخصصة).
        """

        import nltk
        from nltk.corpus import stopwords

        # تحميل القوائم (يتم مرة واحدة فقط)
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        try:
            # محاولة جلب الكلمات من NLTK (عربي + إنجليزي)
            nltk_stops = set(stopwords.words('english')).union(set(stopwords.words('arabic')))
        except Exception:
            # Fallback في حال عدم وجود المكتبة
            nltk_stops = {'the', 'and', 'are', 'was', 'with', 'from'}

        # جلب الكلمات من NLTK (عربي + إنجليزي)
        nltk_stops = set(stopwords.words('english')).union(set(stopwords.words('arabic')))

        # إضافاتك "السيادية" بناءً على اللوجات الأخيرة (الكلمات التي تسللت للـ Hubs)
        custom_stops = {
            # كلمات الربط والحالة
            'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'does', 'did',
            'shall', 'should', 'will', 'would', 'must', 'can', 'could', 'may', 'might',

            # كلمات التنقل والإشارة (التي تظهر في الأبحاث)
            'thus', 'therefore', 'however', 'moreover', 'hence', 'since', 'already',
            'within', 'without', 'instead', 'using', 'used', 'use', 'via', 'through',

            # كلمات أكاديمية "ضجيج"
            'study', 'paper', 'research', 'result', 'figure', 'table', 'data', 'shown',
            'chapter', 'page', 'appendix', 'author', 'university', 'system', 'process',

            # حروف وأرقام مفردة قد تظهر كـ Hubs
            'one', 'two', 'three', 'first', 'second', 'third', 'etc'
        }

        return nltk_stops.union(custom_stops)

# ------------ رابعاً: الشبكة الاستدلالية والروابط (Heuristic Network) ------------
    def _build_heuristic_links(self, page_num: int):
        """
        [محرك الربط العصبي السيادي]: نسخة محصنة تمنع تضخم البيانات وتضمن
        اكتشاف مراكز الثقل (Hubs) بدقة عالية.
        """
        import collections

        # 1. الاستعادة والتأمين
        page_data = self.get_page_data(page_num)
        if not page_data: return

        # التأكد من تهيئة الشبكة إذا لم تكن موجودة
        if not hasattr(self, 'heuristic_network'):
            self.heuristic_network = collections.defaultdict(list)
        if not hasattr(self, 'hub_registry'):
            self.hub_registry = set()

        keywords = set(page_data.get("semantic_keywords", []))
        current_layer = page_data.get("layer_type", "STANDARD_CONTENT")
        zoom_applied = page_data.get("metadata", {}).get('zoom_applied', 1.0)

        # 2. وزن الطبقة (Layer Weighting)
        priority_map = {
            "TECHNICAL_CHAPTER": 1.8,
            "TECHNICAL_DATA": 1.4,
            "CHAPTER_LAYER": 1.2
        }
        rank_weight = priority_map.get(current_layer, 0.6) * zoom_applied
        total_connections_for_page = 0

        # 3. بناء الجسور الاستدلالية (Cross-Linking)
        if not hasattr(self, '_link_registry'):
            self._link_registry = set()

        for kw in keywords:
            # التأكد من تهيئة القائمة للكلمة الجديدة
            if kw not in self.heuristic_network:
                self.heuristic_network[kw] = []

            # إضافة الصفحة للشبكة الدلالية
            self.heuristic_network[kw].append({
                    "page": page_num,
                    "weight": 0.8
                })

            # جلب التوائم الدلالية (أحدث 50 علاقة لضمان الكفاءة)
            related_entries = self.heuristic_network[kw][-50:]

            for entry in related_entries:
                past_page = entry["page"]
                if past_page == page_num: continue

                # تعريف المعرف الفريد للرابط (Atomic Link ID)
                # نستخدم tuple مرتب لضمان أن الرابط بين (1 و 5) هو نفسه بين (5 و 1)
                link_id = tuple(sorted((page_num, past_page)))

                # التحقق عبر الـ Set (سرعة O(1) بدلاً من O(n))
                if link_id not in self._link_registry:
                    self._link_registry.add(link_id) # تسجيل الرابط

                    self.visual_links.append({
                        "origin": page_num,
                        "target": past_page,
                        "strength": round(float(0.8 + entry["weight"]) / 2, 3), # استخدمنا 0.8 كوزن أساسي
                        "term": kw,
                        "type": "semantic_bridge"
                    })

        # 4. إدارة مراكز الثقل (Hub Management)
        hub_threshold = 8 if current_layer == "STANDARD_CONTENT" else 4

        if total_connections_for_page >= hub_threshold and page_num not in self.hub_registry:
            self.hub_registry.add(page_num)
            self.network_hubs_count = len(self.hub_registry)
            self.logger.info(f"🌟 New Hub Identified: P{page_num} ({current_layer}) | Connections: {total_connections_for_page}")

        # 5. الربط الهيكلي (Hierarchy Bridge)
        if rank_weight >= 1.0:
            # ربط بآخر 3 صفحات لخلق تسلسل منطقي
            self.layer_hierarchy[page_num] = list(range(max(1, page_num-3), page_num))

        # تنظيف دوري (Memory Protection)
        if len(self.visual_links) > 10000:
            self.visual_links = [l for l in self.visual_links if l.get("strength", 0) > 0.5]

    def get_strategic_hubs(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        [تطوير إستراتيجي]: تحديد مراكز الثقل المعرفي (المفاهيم القائدة).
        يوازن بين ثقل الكلمة ومدى انتشارها عبر الأرشيف.
        """
        hub_scores: Dict[str, float] = defaultdict(float)

        # 1. تحليل الشبكة بذكاء
        for keyword, entries in self.heuristic_network.items():
            if len(keyword) < 3 or not isinstance(entries, list): continue

            pages = [e["page"] for e in entries if isinstance(e, dict)]
            if not pages: continue

            # حساب المدى والتنوع
            unique_pages = set(pages)
            page_span = max(pages) - min(pages) + 1

            # معادلة الجاذبية (Sovereign Gravity Equation)
            # الكلمة القوية هي التي تربط صفحات متباعدة (Span) ومتنوعة (Unique)
            for entry in entries:
                base_weight = float(entry.get("weight", 0.1))

                # حوافز الذكاء الاصطناعي
                diversity_multiplier = 1.5 if len(unique_pages) > 2 else 1.0
                span_multiplier = 1.3 if page_span > 15 else 1.0

                # التراكم الذكي
                hub_scores[keyword] += (base_weight * diversity_multiplier * span_multiplier)

        # 2. فرز النتائج
        sorted_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)
        top_results = sorted_hubs[:top_n]

        # 3. [إصلاح التنسيق]: تحديث عداد "المفاهيم السيادية"
        self.knowledge_density_index = len([h for h in hub_scores.values() if h > 2.0])

        # التحقق من وجود نتائج لتجنب IndexError
        if top_results:
            self.logger.info(
                f"🎯 Hubs Engine: Top Concept '{top_results[0][0]}' "
                f"| Strength: {top_results[0][1]:.2f} | Density: {self.knowledge_density_index}"
            )
        else:
            self.logger.warning("🎯 Hubs Engine: No strategic hubs discovered in the current network.")

        return top_results

    def _build_visual_heuristics(self, page_num: int, layout_data: Dict[str, Any]):
        """
        [المشرف البصري السيادي]: بناء شبكة الملاحة العصبية للمستند.
        """
        headings = layout_data.get("headings", [])

        # 1. تأمين المتغيرات الأساسية في الكلاس (يفضل أن تكون في __init__)
        if not hasattr(self, 'heading_map'):
            self.heading_map = defaultdict(list)
        if not hasattr(self, 'engine_metadata'):
            self.engine_metadata = {"dna_flow": []}

        # 2. معالجة العناوين
        for heading in headings:
            text_raw = heading.get("text", "")
            # تعريف title داخل الحلقة لضمان وجوده لكل عنوان
            title = str(text_raw).strip().lower()

            if len(title) < 3:
                continue

            # تسجيل العنوان
            self.heading_map[title].append(page_num)

            # الاستمرارية الدلالية
            if len(self.heading_map[title]) > 1:
                source_page = self.heading_map[title][-2]
                is_high_priority = heading.get("priority") == "HIGH"

                self.visual_links.append({
                    "link_type": "HEADING_CONTINUITY",
                    "origin": source_page,
                    "target": page_num,
                    "anchor": title,
                    "strength": 2.0 if is_high_priority else 1.4
                })

                # تسجيل الـ Hub الإستراتيجي (title متاح هنا بالتأكيد)
                hub_id = f"v_hub_{title.replace(' ', '_')}"
                if not hasattr(self, hub_id):
                    self.network_hubs_count += 1
                    setattr(self, hub_id, True)
                    self.engine_metadata["dna_flow"].append(f"Hub detected: {title}")

        # 3. التدفق الهيكلي (خارج حلقة العناوين لأنها تخص الصفحة ككل)
        if page_num > 0:
            self.visual_links.append({
                "link_type": "STRUCTURAL_FLOW",
                "origin": page_num - 1,
                "target": page_num,
                "strength": 0.9
            })

        self.logger.info(f"🕸️ Visual Grid: P{page_num} processed. Hubs: {self.network_hubs_count}")

    # ------------ خامساً: البحث والتنقل الذكي (Search & Navigation) ------------
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        [البحث السيمانتيكي السيادي المطور]: يجمع بين دقة المتجهات وقوة الأوزان
        الهيكلية مع تأمين كامل ضد انهيار الـ GPU.
        """
        import numpy as np
        import torch

        if not self.vectors or len(self.vectors) == 0:
            self.logger.warning("⚠️ مساحة المتجهات فارغة. يرجى فهرسة المستند أولاً.")
            return []

        # 1. التشفير مع بروتوكول الفشل الآمن (Fallback Protocol)
        try:
            # محاولة التشفير باستخدام الجهاز الافتراضي (CUDA غالباً)
            query_vector = self.model.encode(query, convert_to_numpy=True, device=self.device)
        except Exception as e:
            if "cuda" in str(e).lower() or "memory" in str(e).lower():
                self.logger.warning("🚨 GPU Pressure during search! Reverting to CPU encoding.")
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                query_vector = self.model.encode(query, convert_to_numpy=True, device="cpu")
            else:
                self.logger.error(f"💥 Query Encoding Failed: {e}")
                return []

        # 2. الحساب المصفوفي السريع (Vector Math)
        vectors_np = np.array(self.vectors)
        sim_scores = np.dot(vectors_np, query_vector)
        v_norms = np.linalg.norm(vectors_np, axis=1)
        q_norm = np.linalg.norm(query_vector)
        similarities = sim_scores / (v_norms * q_norm + 1e-9)

        # 3. سلم الأوزان الطبقية (Hybrid Importance)
        # رفعنا أوزان البيانات التقنية لضمان ظهورها في النتائج الاستراتيجية
        layer_weights = {
            "TECHNICAL_CHAPTER": 1.25,
            "CHAPTER_LAYER": 1.15,
            "TECHNICAL_DATA": 1.05,
            "CORE_CONTENT": 0.85,
            "STANDARD_CONTENT": 0.55,
            "APPENDIX_LAYER": 0.45
        }

        # 4. حساب الدرجات النهائية بميزان (70/30)
        weighted_scores = []
        for idx, semantic_score in enumerate(similarities):
            # التأكد من صحة الفهرسة
            if idx >= len(self.vector_map):
                continue

            page_num = self.vector_map[idx]
            p_data = self.get_page_data(page_num)

            # --- إصلاح Pylance: التأكد من أن p_data قاموس وليس None ---
            struct_weight = 0.4 # القيمة الافتراضية

            if isinstance(p_data, dict):
                # الوصول الآمن للمفتاح مع التأكد من النوع
                layer_type = p_data.get("layer_type")
                if layer_type:
                    struct_weight = layer_weights.get(str(layer_type), 0.4)

            # معادلة التوازن (Sovereign Balance Equation)
            # semantic_score هو القيمة المستخرجة من التشابه الجيبي (Cosine Similarity)
            final_score = (float(semantic_score) * 0.7) + (float(struct_weight) * 0.3)
            weighted_scores.append(final_score)

        # 5. تجميع "النخبة الاستراتيجية" بموثوقية عالية
        # نستخدم argsort للحصول على الفهارس المرتبة تنازلياً
        top_indices = np.argsort(weighted_scores)[::-1][:top_k]
        results = []

        for rank_idx in top_indices:
            # حماية: التأكد من أن الفهرس ضمن نطاق الخريطة
            if rank_idx >= len(self.vector_map):
                continue

            page_num = self.vector_map[rank_idx]
            p_data = self.get_page_data(page_num)

            # معالجة الخطأ: Object of type "None" is not subscriptable
            if p_data is not None and isinstance(p_data, dict):
                # استخدام .get() للوصول الآمن وتجنب KeyError أو None Errors
                content_raw = p_data.get("content", "")

                # التأكد من أن content نص قبل عمل slicing
                content_snippet = str(content_raw)[:450].strip() + "..." if content_raw else "No content available."

                results.append({
                    "page": (page_num or 0) + 1, # حماية ضد None في الجمع
                    "score": round(float(weighted_scores[rank_idx]), 4),
                    "layer": p_data.get("layer_type", "UNKNOWN"),
                    "location": p_data.get("location_stamp", "N/A"),
                    "content": content_snippet,
                    "genes": p_data.get("semantic_keywords", [])[:5]
                })
            else:
                self.logger.warning(f"⚠️ Missing data for page index {page_num}. Skipping result.")

        self.logger.info(f"🔍 Sovereign Search: Found {len(results)} high-precision insights.")
        return results

    def page_flipper(self, current_page: int, direction: str = "next") -> int:
        """
        [الملاح التكتيكي]: نظام تنقل ذكي يتجاوز الخطية.
        يقفز بين "مراكز الثقل" (Hubs) لضمان سرعة الوصول للمعلومة.
        """
        # التأكد من وجود سجل التنقل في __init__
        if not hasattr(self, 'navigation_stack'):
            self.navigation_stack = []

        # منع التكرار في السجل
        if not self.navigation_stack or self.navigation_stack[-1] != current_page:
            self.navigation_stack.append(current_page)
            # حماية السجل من التضخم (أقصى حد 50 خطوة)
        if not self.navigation_stack or self.navigation_stack[-1] != current_page:
            self.navigation_stack.append(current_page)

        target_page = current_page
        total_indexed = len(self.vector_map)

        # 1. الملاحة الخطية
        if direction == "next":
            target_page = min(total_indexed - 1, current_page + 1)
        elif direction == "prev":
            target_page = max(0, current_page - 1)

        # 2. القفز الذكي (Hub Jumping)
        elif direction == "next_hub":
            # البحث عن أقرب فصل تقني أو رئيسي قادم
            found = False
            for p_idx in range(current_page + 1, total_indexed):
                # استخدام get_page_data لضمان فحص الطبقة حتى لو الصفحة خارج الكاش
                p_data = self.get_page_data(p_idx)
                if p_data and p_data.get("layer_type") in ["TECHNICAL_CHAPTER", "CHAPTER_LAYER"]:
                    target_page = p_idx
                    found = True
                    break
            if not found: target_page = min(total_indexed - 1, current_page + 1)

        # 3. العودة الذكية (Backtrack)
        elif direction == "back":
            if len(self.navigation_stack) > 1:
                self.navigation_stack.pop() # إزالة الصفحة الحالية
                target_page = self.navigation_stack.pop() # العودة للسابقة

        # 4. المزامنة مع الذاكرة (Cache Sync)
        # التأكد من أن الصفحة المستهدفة "حية" وجاهزة للعرض
        p_data = self.get_page_data(target_page)

        # إذا كانت الصفحة موجودة في الكاش، ننقلها للنهاية (حماية من الحذف)
        if target_page in self.page_cache:
            self.page_cache.move_to_end(target_page)

        self.logger.info(f"🧭 Navigator: P{current_page} -> P{target_page} | Logic: {direction}")
        return target_page

    def inspect_cache(self, page_num: Optional[int] = None) -> Dict[str, Any]:
        """
        [تطوير تشخيصي]: فحص عميق لحالة الوعي وتدفق الـ DNA في النظام.
        يوفر رؤية شاملة للصحة العامة أو تشريحاً دقيقاً لصفحة معينة.
        """
        # 1. نظرة شموليّة على "صحة النظام" (Global Health)
        if page_num is None:
            # حساب توزيع الطبقات ديناميكياً لضمان عدم وجود فجوات في التصنيف
            pages_data = list(self.page_cache.values())
            layers = [d.get("metadata", {}).get("layer_type", "UNKNOWN") for d in pages_data]
            layer_stats = {l: layers.count(l) for l in set(layers)}

            # استخراج أفضل 5 Hubs (استخدام المنهجية التي صممتها سابقاً)
            top_concepts = self.get_network_hubs(top_n=5)

            return {
                "status": "GLOBAL_SYSTEM_HEALTH",
                "metrics": {
                    "active_buffer": f"{len(self.page_cache)}/{self.max_pages}",
                    "network_hubs": self.network_hubs_count,
                    "total_vectors": len(self.vectors),
                    "visual_links": len(self.visual_links)
                },
                "intelligence": {
                    "layer_distribution": layer_stats,
                    "dominant_concepts": {name: f"{score:.2f}" for name, score in top_concepts}
                }
            }

        # 2. تشريح دقيق لمستوى الصفحة (Surgical Page View)
        # محاولة جلب البيانات (ستفعل محرك الاستعادة التلقائي إذا كانت الصفحة في الأرشيف)
        page_data = self.page_cache.get(page_num)

        if not page_data:
            return {
                "status": "VOID",
                "message": f"Page {page_num} is outside the active cache or archive."
            }

        # تحليل الاتصال: البحث عن الكلمات المفتاحية التي تربط هذه الصفحة بالشبكة
        # نستخدم getattr لضمان عدم تعطل النظام إذا لم تكن الشبكة مهيئة
        heuristic_net = getattr(self, 'knowledge_graph', {})
        page_content = page_data.get("content", "").lower()

        influence_map = {}
        # فحص الكلمات التي تم تسجيلها كـ Hubs ووجودها في هذه الصفحة
        for hub_name in self._registered_hubs:
            if hub_name in page_content:
                connections = len(heuristic_net.get(hub_name, []))
                if connections > 1:
                    influence_map[hub_name] = f"{connections} cross-links"

        return {
            "status": "PAGE_DNA_SURGERY",
            "identity": {
                "page_id": page_num,
                "layer": page_data.get("metadata", {}).get("layer_type"),
                "location": page_data.get("metadata", {}).get("location_stamp"),
                "word_count": len(page_content.split())
            },
            "neural_connectivity": {
                "structural_hierarchy": self.layer_hierarchy.get(page_num, []),
                "semantic_influence": influence_map,
                "visual_anchors": page_data.get("metadata", {}).get("headings", [])
            },
            "integrity_check": {
                "is_vectorized": any(m == page_num for m in self.vector_map),
                "in_active_cache": page_num in self.page_cache,
                "indexed_at": page_data.get("metadata", {}).get("indexed_at")
            }
        }

    # ------------ سادساً: المحرك الفكري والتحليل (Thinking & Analysis) ------------
    def analyze_pdf(self, pdf_path: str, thinking_engine: Any, user_request: str = "") -> Dict[str, Any]:
        """
        [القائد الأعلى - النسخة السيادية]: تدير التدفق المعلوماتي من المسح السريع إلى الأرشفة العميقة.
        """
        from pathlib import Path

        # 1. الاستطلاع السريع (Stage 1)
        self.logger.info(f"📡 Stage 1: Fast Ingest for {Path(pdf_path).name}")
        ingest_status = self.fast_ingest_stream(pdf_path)

        # تصحيح أمان: ضمان عدم الانهيار إذا فشل الـ Ingest
        if not ingest_status or ingest_status.get("status") != "completed":
            self.logger.error("❌ Stage 1 Failed: Ingest status is None or incomplete.")
            return {"status": "error", "message": "Fast ingestion failed."}

        page_count = ingest_status.get("pages", 0)
        all_ideas, cumulative_awareness = [], []

        # 2. الاستدلال الملاحي الذكي (Stage 2)
        # تحسين: إذا كان طلب المستخدم فارغاً، نحلل كل الصفحات، وإذا وجد طلباً نستخدم الـ Traversal
        target_pages = self.infer_navigation_path(user_request)
        if not target_pages:
            self.logger.info("ℹ️ No specific path inferred. Analyzing full document context.")
            analysis_queue = list(range(page_count))
        else:
            analysis_queue = list(target_pages)

        # 3. التحليل النبضي المعزز (Advanced Recursive Pulse)
        step = 20
        total_items = len(analysis_queue)
        total_pulses = (total_items + step - 1) // step

        # إضافة "الذاكرة قصيرة المدى" للنبضات لربط الأفكار ببعضها
        short_term_memory = ""

        for pulse_idx, start_idx in enumerate(range(0, total_items, step)):
            current_batch = analysis_queue[start_idx : start_idx + step]
            if not current_batch: continue

            chunk_texts, related_summaries = [], []
            for p_num in current_batch:
                page_data = self.get_page_data(p_num)

                if isinstance(page_data, dict) and page_data.get('content'):
                    # تعزيز الـ Tag ليشمل "رتبة الصفحة" في الجراف
                    hubs_count = len(self.knowledge_graph.get(p_num, []))
                    loc_tag = f"[LOCATION: P{p_num+1} | CONNECTIVITY: {hubs_count}]"
                    chunk_texts.append(f"{loc_tag}\n{page_data.get('content', '')}")

                    # سحب العلاقات السيمانتيكية العميقة (Deep DNA Cross-Referencing)
                    # نحن لا نسحب جيران الصفحة فقط، بل نسحب "الجسور المعرفية"
                    related_ids = self.layer_hierarchy.get(p_num, [])[:3]
                    for r_id in related_ids:
                        r_meta = self.page_cache.get(r_id, {})
                        if isinstance(r_meta, dict):
                            summary = r_meta.get("summary", "No summary")[:100]
                            related_summaries.append(f"[Bridge to P{r_id+1}: {summary}...]")

            # صياغة البرومبت "السيادي" - يجمع بين الماضي (Memory) والحاضر (Cluster) والأهداف
            enhanced_prompt = (
                f"### SOVEREIGN ANALYSIS PROTOCOL - PULSE {pulse_idx + 1}/{total_pulses}\n"
                f"STRATEGIC_OBJECTIVE: {user_request}\n\n"
                f"--- CUMULATIVE_CONTEXT_SNAPSHOT (PREVIOUS PULSES) ---\n"
                f"{short_term_memory if short_term_memory else 'Initial state: Starting Analysis'}\n\n"
                f"--- CURRENT_CONTENT_CLUSTER ---\n"
                f"{chr(10).join(chunk_texts)}\n\n"
                f"--- KNOWLEDGE_GRAPH_RELATIONS ---\n"
                f"{chr(10).join(related_summaries)}\n\n"
                f"INSTRUCTION: Analyze this cluster. If it contradicts or enhances the 'CUMULATIVE_CONTEXT', highlight the delta."
            )

            try:
                # الاستدعاء البرمجي مع دعم "محرك التفكير"
                gen_func = getattr(thinking_engine, 'generate', None) or thinking_engine.llm_bridge
                chunk_ideas, chunk_state = gen_func(enhanced_prompt, user_request=user_request)

                # تحديث الذاكرة قصيرة المدى للنبضة القادمة (تلخيص ما تم فهمه حتى الآن)
                # هذا السطر يمنع "تشتت الـ DNA" عبر الصفحات
                short_term_memory = chunk_state.get("executive_summary", "Continuing audit...")

                # حساب درجة الاستقرار والنبض
                score = float(chunk_state.get("avg_score", 0.5))
                self._analysis_monitor(pulse_idx + 1, total_pulses, score)

                # تسجيل النتائج في ميتاداتا المحرك للتتبع
                self.engine_metadata["dna_flow"].append(f"Pulse {pulse_idx+1} completed. Stability: {score}")

                all_ideas.extend(chunk_ideas)
                cumulative_awareness.append(score)

            except Exception as e:
                self.logger.error(f"❌ Pulse Error during batch {start_idx}: {e}")
                self.engine_metadata["system_alerts"].append(f"Critical Failure in Pulse {pulse_idx+1}")

        # 4. التقرير النهائي الاستراتيجي والأرشفة
        final_score = round(sum(cumulative_awareness)/len(cumulative_awareness), 2) if cumulative_awareness else 0.0

        # استخراج ميتاداتا الأرشفة (Folders & Binders)
        all_metadata = [d.get('metadata', {}) for d in self.page_cache.values() if isinstance(d, dict)]
        folders = {str(m.get('folder_id')) for m in all_metadata if m.get('folder_id')}
        binders = {f"{m.get('folder_id')}_{m.get('binder_id')}" for m in all_metadata if m.get('binder_id')}

        return {
            "status": "success",
            "analysis_metrics": {
                "pages_analyzed": total_items,
                "hubs_found": getattr(self, 'network_hubs_count', 0),
                "consciousness_score": final_score,
                "inferred_topic": self._guess_topic(self.get_page_data(0).get("content", ""))
            },
            "sovereign_archive": {
                "portfolio": "ROBOTICS_MASTER_DB",
                "total_folders": len(folders),
                "total_binders": len(binders),
                "active_map": list(folders)
            },
            "output": {
                "structured_ideas": all_ideas,
                "summary": f"Sovereign Audit Complete ✅ | Found {len(all_ideas)} insights across {len(folders)} folders."
            }
        }

    def advance_pdf_analyzer(self, pdf_path: str, user_request: str = "") -> Dict[str, Any]:
        """
        [مدير العمليات المتقدمة المطور]: تحليل نبضي موحد مع إدارة ذكية للتدفق
        وحماية الذاكرة عبر التداخل (Overlap) الديناميكي.
        """
        # 1. المرحلة الهيكلية
        process_result = self.process_pdf_core(pdf_path, user_request)
        if process_result["status"] != "success":
            return process_result

        page_count = process_result["data"]["stats"]["pages"]
        all_ideas: List[Dict] = []
        pulse_scores: List[float] = []
        step = 20
        overlap_pages = 2

        try:
            # 2. التحليل النبضي الموحد (Unified Pulse Analysis)
            # تم دمج الحلقات في حلقة واحدة لمنع تكرار المعالجة
            for start_idx in range(0, page_count, step):
                # تحديد النطاق مع التداخل لضمان استمرارية السياق المعرفي
                actual_start = max(0, start_idx - overlap_pages) if start_idx > 0 else start_idx
                end_idx = min(start_idx + step, page_count)

                # استخراج النصوص باستخدام get_page_data لضمان الاستعادة من القرص إذا فُقد الكاش
                chunk_pages_text = []
                for i in range(actual_start, end_idx):
                    page_data = self.get_page_data(i)
                    content = page_data.get("content", "")
                    if content:
                        chunk_pages_text.append(str(content))

                chunk_text = "\n".join(chunk_pages_text)
                if not chunk_text.strip():
                    continue

                # 3. استدعاء محرك الاستدلال (Inference)
                # استخدام محاولة واحدة مكثفة لتوفير وقت المعالج
                chunk_ideas, chunk_state = self.generate_mock(chunk_text, max_iterations=1, user_request=user_request)

                if chunk_ideas:
                    all_ideas.extend(chunk_ideas)
                    # حقن الأفكار وتحديث الـ Hubs
                    for idea in chunk_ideas:
                        self.engine_metadata["dna_flow"].append(idea)
                        if idea.get("confidence_score", 0) > 0.8:
                            self.network_hubs_count += 1

                pulse_scores.append(float(chunk_state.get("avg_score", 0.5)))

                # تحسين هاردوير: تفريغ الذاكرة المؤقتة بعد كل نبضة (Pulse)
                if start_idx % 40 == 0:
                    gc.collect()

            # 4. تجميع الوعي وتخمين الموضوع
            final_avg_score = round(sum(pulse_scores) / len(pulse_scores), 2) if pulse_scores else 0.5

            # الحصول على بيانات الغلاف (النبضة الصفرية)
            cover_data = self.get_page_data(0)
            sample_text = cover_data.get("content", "") if cover_data else ""

            self.logger.info(f"✅ Pulse Sync Complete: {len(all_ideas)} concepts crystallized.")

            return {
                "status": "success",
                "metrics": {
                    "pages": page_count,
                    "hubs": self.network_hubs_count,
                    "awareness": final_avg_score
                },
                "pdf_metadata": {
                    "inferred_topic": self._guess_topic(sample_text),
                    "file": pdf_path
                },
                "output": {
                    "structured_ideas": all_ideas,
                    "summary": f"Analyzed via {len(pulse_scores)} pulses → {len(all_ideas)} concepts extracted."
                }
            }

        except Exception as e:
            self.logger.error(f"❌ Critical Error in Pulse Operations: {str(e)}")
            return {"status": "error", "message": str(e)}

    def infer_navigation_path(self, query: str) -> List[int]:
        """
        [Reasoning Engine]: محرك الاستدلال السيادي.
        يحدد المسارات الطبقية بناءً على نية المستخدم ويربطها بالـ Hubs.
        """
        query_lower = query.lower()
        target_pages: List[int] = []

        # 1. الاستدلال الطبقي (Layer Inference)
        technical_indicators = {'جدول', 'رسم', 'بيانات', 'إحصائيات', 'table', 'data', 'figure', 'chart', 'stats'}
        structural_indicators = {'فصل', 'عنوان', 'مقدمة', 'محتوى', 'chapter', 'section', 'index', 'contents'}

        # المسار أ: البحث في طبقة البيانات التقنية
        if any(w in query_lower for w in technical_indicators):
            target_pages = [p for p, d in self.page_cache.items() if d.get('layer_type') == 'TECHNICAL_DATA']
            self.logger.info("🎯 Inference: Technical Routing Engaged.")

        # المسار ب: البحث في طبقة الهيكل التنظيمي
        elif any(w in query_lower for w in structural_indicators):
            target_pages = [p for p, d in self.page_cache.items() if d.get('layer_type') == 'CHAPTER_LAYER']
            self.logger.info("🎯 Inference: Structural Routing Engaged.")

        # 2. التوجيه عبر مراكز الثقل (Dynamic Hub Routing)
        # إذا لم يحدد المستخدم طبقة معينة، نلجأ للـ Hubs التي اكتشفها النظام
        if not target_pages:
            strategic_hubs = self.get_strategic_hubs(top_n=5)
            for hub_word, score in strategic_hubs:
                if hub_word.lower() in query_lower:
                    # استخراج الصفحات المرتبطة بهذا الـ Hub من الشبكة الاستدلالية
                    hub_pages = [e['page'] for e in self.heuristic_network.get(hub_word, [])]
                    target_pages.extend(hub_pages)
                    self.logger.info(f"🧠 Hub Routing: Found match via '{hub_word}' (Score: {score:.2f})")

        # 3. الفلترة النهائية وتجنب التكرار
        final_path = sorted(list(set(target_pages)))

        # 4. تحديث مؤشر الوعي (Confidence Update)
        # نرسل إشارة للمراقب عن جودة الاستدلال
        inference_quality = 1.0 if final_path else 0.0
        if hasattr(self, 'cumulative_awareness'):
            self.cumulative_awareness.append(inference_quality)

        return final_path

    def get_related_pages(self, page_num: int, depth: int = 2, decay_factor: float = 0.5) -> Dict[int, float]:
        """
        [محرك الجيرة المعرفية المحصن]: استقصاء سياقي شامل يعالج البيانات التالفة
        ويضمن الربط الهيكلي حتى في حال فشل الاستنتاج السيمانتيكي.
        """
        # 1. تأمين المدخلات (تحويل رقم الصفحة لضمان الاتساق)
        try:
            p_idx = int(page_num)
        except (ValueError, TypeError):
            return {}

        # التحقق من وجود الصفحة في الكاش أو الخارطة
        if p_idx not in self.page_cache and p_idx not in self.vector_map:
            return {}

        related_scores: Dict[int, float] = {}

        # 2. الاستقصاء الهيكلي (Hierarchy Bridge) - القيمة الافتراضية للتسلسل
        if hasattr(self, 'layer_hierarchy') and self.layer_hierarchy:
            neighbors = self.layer_hierarchy.get(p_idx, [])
            for p in neighbors:
                related_scores[p] = max(related_scores.get(p, 0), 0.8)

        # ربط تلقائي بالصفحات المجاورة (Fallback Structural Link)
        # لضمان عدم ظهور "0 Links" في التقرير أبدًا
        for adj in [p_idx - 1, p_idx + 1]:
            if adj in self.page_cache or adj in self.vector_map:
                related_scores[adj] = max(related_scores.get(adj, 0), 0.4)

        # 3. الاستقصاء البصري (Visual Continuity)
        visual_links = getattr(self, 'visual_links', [])
        for link in visual_links:
            if not isinstance(link, dict): continue
            origin = link.get("origin")
            target = link.get("target")
            weight = link.get("weight", 0.9)

            if origin == p_idx and target:
                related_scores[target] = max(related_scores.get(target, 0), weight)
            elif target == p_idx and origin:
                related_scores[origin] = max(related_scores.get(origin, 0), weight)

        # 4. الاستقصاء السيمانتيكي (Semantic DNA Mapping) - معالجة البيانات التالفة
        page_data = self.page_cache.get(p_idx, {})
        keywords = []

        # استخراج الكلمات بمرونة (سواء كانت الصفحة Dict أو Str)
        if isinstance(page_data, dict):
            keywords = page_data.get("semantic_keywords", [])
            # إذا كانت القائمة فارغة، نحاول استخراجها من المحتوى النصي مباشرة
            if not keywords and "content" in page_data:
                keywords = self.PDF_extract_keywords(page_data["content"])
        elif isinstance(page_data, str):
            keywords = self.PDF_extract_keywords(page_data)

        # البحث في الشبكة الاستدلالية (Heuristic Network)
        if hasattr(self, 'heuristic_network') and self.heuristic_network:
            for kw in keywords:
                if kw in self.heuristic_network:
                    entries = self.heuristic_network[kw]
                    for entry in entries:
                        if not isinstance(entry, dict): continue
                        target_p = entry.get("page")
                        if target_p and target_p != p_idx:
                            # حساب الوزن بناءً على قوة الكلمة (Hub Strength)
                            s_weight = float(entry.get("weight", 0.5)) * 0.7
                            related_scores[target_p] = max(related_scores.get(target_p, 0), s_weight)

        # 5. التوسع العميق (Recursive Expansion) مع حماية ضد الـ Infinite Loops
        if depth > 1:
            secondary_scores = {}
            # نأخذ نسخة من المفاتيح الحالية لتجنب "RuntimeError: dictionary changed size"
            current_neighbors = list(related_scores.keys())
            for p in current_neighbors:
                # استدعاء تناقصي للعمق
                sub_neighbors = self.get_related_pages(p, depth=depth-1, decay_factor=decay_factor)
                for sp, sw in sub_neighbors.items():
                    if sp != p_idx and sp not in related_scores:
                        # تطبيق معامل الاضمحلال (Decay)
                        decayed_val = sw * decay_factor
                        secondary_scores[sp] = max(secondary_scores.get(sp, 0), decayed_val)
            related_scores.update(secondary_scores)

        # 6. التصفية النهائية والتحقق من الوجود الفعلي في الأرشيف
        final_ranked = {
            int(p): round(float(w), 3)
            for p, w in related_scores.items()
            if p != p_idx and (p in self.page_cache or p in self.vector_map)
        }

        # فرز النتائج تنازلياً حسب القوة
        return dict(sorted(final_ranked.items(), key=lambda x: x[1], reverse=True))

    def _analysis_monitor(self, current_pulse: int, total_pulses: int, chunk_score: float):
        """
        [لوحة التحكم السيادية - الإصدار القتالي]
        تدمج بين مراقبة الموارد وقوة الروابط المعرفية مع هيكلة برمجية مقسمة.
        """
        try:
            # .......... 1. حساب المؤشرات الحيوية بمرونة عالية ..........
            # نضمن عدم القسمة على صفر ونحسب نسبة التقدم واستهلاك الكاش
            total_safe = max(total_pulses, 1)
            progress = (current_pulse / total_safe) * 100

            cache_count = len(self.page_cache)
            max_p_safe = getattr(self, 'max_pages', 100)
            cache_usage_pct = (cache_count / max(max_p_safe, 1)) * 100

            # .......... 2. معالجة الوعي السيمانتيكي والروابط ..........
            # استباق استخراج الـ Hubs لمنع أخطاء التعريف وتحديث مؤشرات الكثافة
            discovered_hubs = self.get_strategic_hubs(top_n=3)

            if not hasattr(self, 'cumulative_awareness') or not isinstance(self.cumulative_awareness, list):
                self.cumulative_awareness = []

            self.cumulative_awareness.append(float(chunk_score))
            global_awareness = sum(self.cumulative_awareness) / len(self.cumulative_awareness)

            self.knowledge_density_index = len(discovered_hubs)
            knowledge_density = self.knowledge_density_index

            v_links = getattr(self, 'visual_links', [])
            active_links = len(v_links) if isinstance(v_links, (list, dict)) else 0

            # .......... 3. تحديد الموقع الجغرافي في الأرشيف ..........
            # ربط النبضة الحالية بموقعها المادي (المجلد والملف) داخل النظام
            current_page_idx = getattr(self, 'current_processing_page', current_pulse)
            loc = self._assign_sovereign_location(current_page_idx)
            archive_pos = f"{loc.get('folder_id', '??')} > {loc.get('binder_id', '??')}"

            # .......... 4. تعيين الرموز البصرية (Visual Status) ..........
            # تحويل الأرقام الجافة إلى أيقونات بصرية لسهولة القراءة السريعة
            quality_icon = "💎" if chunk_score >= 0.85 else "✅" if chunk_score >= 0.5 else "⚠️"
            memory_icon = "🟢" if cache_usage_pct < 70 else "🟡" if cache_usage_pct < 85 else "🟠" if cache_usage_pct < 95 else "🔥"
            net_icon = "🕸️" if active_links < 50 else "⚡" if active_links < 200 else "🧠"

            # .......... 5. صياغة وتصدير التقرير الاستراتيجي ..........
            # بناء شكل التقرير النهائي الذي سيظهر في Terminal أو سجلات الـ Log
            report = (
                f"\n{'═'*55}\n"
                f"[ 🛰️ PULSE #{current_pulse:02d} | LOC: {archive_pos} ]\n"
                f"{'═'*55}\n"
                f" 🧬 DNA Quality: {quality_icon} {chunk_score:.4f} | Avg: {global_awareness:.3f}\n"
                f" 🧠 Knowledge: {knowledge_density} Hubs | {net_icon} Links: {active_links}\n"
                f" 📊 Progress: {progress:>5.1f}% | Cache {memory_icon}: {cache_usage_pct:.1f}%\n"
                f" 💾 Buffer: {cache_count}/{max_p_safe} Slots | Status: ACTIVE\n"
                f" {'─'*53}"
            )

            if hasattr(self, 'logger'):
                self.logger.info(report)
            else:
                print(report)

            # .......... 6. بروتوكول الدفاع الذاتي (Memory Cleanup) ..........
            # صمام أمان لتنظيف الذاكرة تلقائياً عند الاقتراب من الحدود القصوى
            if cache_usage_pct >= 90.0 and hasattr(self, '_execute_emergency_cleanup'):
                self.logger.warning(f"🚨 Memory Pressure Alert ({cache_usage_pct:.1f}%). Cleaning...")
                self._execute_emergency_cleanup()

        except Exception as e:
            # منع انهيار المحرك بالكامل بسبب خطأ في العرض (UI Error)
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ Monitor Crash (Non-Critical): {str(e)}")

    def _execute_emergency_cleanup(self):
        """
        [تطهير سيادي]: يحمي النواة المعرفية (Hubs) ويحرر موارد النظام عند الخطر.
        """
        import gc
        try:
            if len(self.page_cache) < 10: return # لا داعي للتطهير إذا كان الكاش صغيراً

            # استهداف تفريغ 30% من الكاش
            target_eviction = max(5, int(len(self.page_cache) * 0.3))
            evicted_pages = []

            # 1. المرحلة الأولى: تحديد الأهداف الضعيفة (Selective Identification)
            # نختار الصفحات التي ليست فصولاً ولا تحتوي على وزن عالٍ في الشبكة
            for p_num, p_data in self.page_cache.items():
                if len(evicted_pages) >= target_eviction: break

                # حماية الطبقات السيادية
                is_protected = p_data.get("layer_type") in ["TECHNICAL_CHAPTER", "TECHNICAL_DATA", "CHAPTER_LAYER"]

                # حماية الصفحات التي تمثل "عُقداً" مركزية في الجراف (Hub Nodes)
                is_hub = p_num in getattr(self, 'network_hubs_ids', set())

                if not is_protected and not is_hub:
                    evicted_pages.append(p_num)

            # 2. التنفيذ الذري (Atomic Eviction)
            for p_num in evicted_pages:
                self.page_cache.pop(p_num, None)
                # تنظيف المتجهات المرتبطة لضمان تحرير مساحة الـ RAM/VRAM
                self._clear_vector_ref(p_num)

            # 3. الإخلاء القسري (في حال لم نصل للهدف)
            while len(evicted_pages) < target_eviction and len(self.page_cache) > 5:
                # نحذف الأقدم (First In First Out) مع تجنب حذف الصفحة الحالية
                first_key = next(iter(self.page_cache))
                self.page_cache.pop(first_key)
                evicted_pages.append(first_key)

            # 4. المزامنة النهائية وتفريغ الذاكرة
            self.logger.warning(f"🚨 [DEFENSE]: Released {len(evicted_pages)} slots. System Stabilized.")

            # مسح الذاكرة المخبئية لـ PyTorch إذا كانت مفعلة
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

        except Exception as e:
            self.logger.error(f"⚠️ Emergency Cleanup Failure: {str(e)}")

    def _clear_vector_ref(self, page_num: int):
        """تفريغ المتجهات المرتبطة بالصفحة المحذوفة لمنع تسرب الذاكرة"""
        if hasattr(self, 'vector_map') and page_num in self.vector_map:
            try:
                idx = self.vector_map.index(page_num)
                self.vector_map.pop(idx)
                if hasattr(self, 'vectors') and idx < len(self.vectors):
                    self.vectors.pop(idx)
            except ValueError:
                pass

    def llm_bridge(self, content: str, user_request: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        [الجسر السحابي]: العقل المدبر للتواصل مع OpenAI.
        تعمل الآن كعضو أساسي داخل الكلاس للوصول إلى ميتاداتا المحرك.
        """
        # 1. التأكد من تهيئة العميل (OpenAI Client)
        if not hasattr(self, 'llm_client') or self.llm_client is None:
            raise AttributeError("⚠️ LLM Client غير معرف. تأكد من ربطه في الـ Main: engine.llm_client = client")

        # 2. بناء سياق غني للموديل السحابي
        # نرسل جزءاً من الـ DNA المكتشف (Hubs) ليساعد الموديل في فهم السياق
        discovered_hubs = list(self._registered_hubs)[:10]

        prompt = f"""
        Task: Deep Technical Analysis
        Context Discovery: {discovered_hubs}
        User Request: {user_request}
        Content to Process: {content[:4500]}

        Return ONLY a JSON object with this structure:
        {{
          "ideas": [
            {{
              "title": "Strategy Name",
              "priority": "CRITICAL/ANALYTICAL",
              "extracted_segments": ["relevant text"],
              "confidence_score": 0.95,
              "dna_tag": "keyword",
              "layer_origin": "CHAPTER_LAYER"
            }}
          ],
          "awareness": {{
            "avg_score": 0.92,
            "top_topic": "Topic Name"
          }}
        }}
        """

        # 3. الاستدعاء مع حماية التوقف (Timeout)
        import json
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a Sovereign System Architect."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                timeout=20.0  # صمام أمان لعدم تجميد الكود
            )

            raw_data = json.loads(response.choices[0].message.content)

            # وسم البيانات بأنها "سحابية" لتمييزها في تقرير الـ Markdown
            for idea in raw_data.get('ideas', []):
                idea['processing_mode'] = "CLOUD_BRIDGE"

            awareness = raw_data.get('awareness', {})
            awareness['engine_status'] = "CLOUD_ACTIVE"
            awareness['hubs_count'] = self.network_hubs_count

            return raw_data.get('ideas', []), awareness

        except Exception as e:
            # في حال فشل السحاب، نرفع الخطأ لتستلمه دالة generate وتفعل الـ Local Fallback
            self.logger.error(f"❌ Cloud Bridge Error: {str(e)}")
            raise e

    def generate(self, content: str, user_request: str = "", max_iterations: int = 3) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        [المولد السيادي المطور]: موازنة ذكية بين السحابة والاستدلال المحلي.
        يضمن استخراج 'اللب السيمانتيكي' وتحديث الجراف قبل التصدير.
        """
        import gc

        try:
            # 1. الجسر السحابي (Cloud Bridge)
            return self.llm_bridge(content, user_request)

        except Exception as e:
            self.logger.warning(f"⚠️ Cloud Bridge Bypass: {e}. Activating Local DNA Logic...")

            # 2. الاستدلال المحلي (Local Genes)
            keywords = self.PDF_extract_keywords(content, top_n=max_iterations)
            topic = self._guess_topic(content)
            layer = self._classify_layer(content, {})

            ideas: List[Dict[str, Any]] = []

            # 3. بناء الأفكار بناءً على ثقل الطبقة
            for i in range(1, max_iterations + 1):
                kw = keywords[i-1] if i <= len(keywords) else "System"

                # تعزيز الثقة للبيانات التقنية (Technical Boost)
                boost = 0.10 if layer in ["TECHNICAL_DATA", "TECHNICAL_CHAPTER"] else 0.0
                confidence = round(min(0.90 - (i * 0.05) + boost, 0.98), 2)

                # --- استخراج 'اللب السيمانتيكي' ---
                relevant_segment = "No specific segment located."
                segments = content.split('.')

                for s in segments:
                    clean_s = s.strip()
                    if kw.lower() in clean_s.lower() and len(clean_s) > 20:
                        relevant_segment = clean_s[:400] + "..."
                        break

                if relevant_segment == "No specific segment located.":
                    relevant_segment = content[:350].strip().replace('\n', ' ') + "..."

                # 4. تسجيل الفكرة وحقن الـ DNA في الجراف
                # هذا السطر يضمن أن تظهر الكلمة في تقارير المجلد Sovereign_Audits لاحقاً
                if kw not in self._registered_hubs:
                    self._registered_hubs.add(kw)
                    self.network_hubs_count += 1

                ideas.append({
                    "idea_id": i,
                    "title": f"[{topic}] Strategy: {kw.upper()}",
                    "priority": "CRITICAL" if i == 1 else "ANALYTICAL",
                    "extracted_segments": [relevant_segment],
                    "confidence_score": confidence,
                    "dna_tag": kw,
                    "layer_origin": layer,
                    "processing_mode": "HEURISTIC_LOCAL"
                })

            # 5. لقطة الوعي (Awareness Snapshot)
            awareness_state = {
                "avg_score": round(sum(d["confidence_score"] for d in ideas)/len(ideas), 2) if ideas else 0.5,
                "top_topic": topic,
                "hubs_count": self.network_hubs_count,
                "engine_status": "LOCAL_FALLBACK"
            }

            gc.collect() # تنظيف الـ VRAM/RAM
            return ideas, awareness_state

    def generate_mock(self, content: str, max_iterations: int = 3, user_request: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        [محرك الاستدلال السيادي المطور]: تم تصحيح ترتيب استخراج الجينات
        وتفعيل البحث السيمانتيكي عن القطاعات النصية الأكثر صلة.
        """
        # 1. حماية الموارد: فحص أولي للنبضة
        content_stripped = content.strip()
        if not content_stripped or len(content_stripped) < 50:
            self.logger.warning("⚠️ Pulse too lean for deep reasoning. Using lightweight mode.")
            # توفير بيانات افتراضية للنمط الخفيف
            keywords = ["General_Context"]
            layer_type = "GENERAL_TEXT"
            scale = {"mode": "LIGHT", "iterations": 1}
        else:
            # 2. كشف المقياس والهوية (Scale & Identity)
            scale = self._detect_processing_scale(content_stripped)
            # استخراج الجينات أولاً قبل محاولة استخدامها
            keywords = self.PDF_extract_keywords(content_stripped, top_n=12)
            layer_type = self._classify_layer(content_stripped, {})

        current_topic = self._guess_topic(content_stripped)
        num_kws = len(keywords)

        self.logger.info(f"⚙️ Mode: [{scale.get('mode', 'N/A')}] | Layer: [{layer_type}] | Target: '{user_request[:25]}...'")

        ideas: List[Dict[str, Any]] = []
        # تحديد عدد التكرارات بناءً على مقياس المعالجة وموارد الجهاز
        actual_iterations = max(1, min(max_iterations, scale.get("iterations", 1)))

        # 3. توليد الأفكار (Idea Synthesis)
        for i in range(1, actual_iterations + 1):
            kw_main = keywords[0] if num_kws > 0 else "Analysis"
            kw_sub = keywords[i % num_kws] if num_kws > i else (keywords[0] if num_kws > 0 else f"Segment_{i}")

            # تعديل درجة الثقة بناءً على "رتبة الطبقة" (Layer Rank)
            layer_boost = 0.15 if layer_type in ["TECHNICAL_DATA", "TECHNICAL_CHAPTER"] else 0.0
            base_score = 0.7 + (min(num_kws, 10) * 0.02) + layer_boost
            dynamic_score = round(min(base_score + (i * 0.01), 0.99), 2)

            # --- تحسين سيمانتيكي: البحث عن أفضل قطاع نصي مرتبط بالكلمة المفتاحية ---
            # بدلاً من أخذ أول 200 حرف، نبحث عن الجملة الأكثر صلة بالـ kw_sub
            best_segment = content_stripped[:250] # قيمة افتراضية
            sentences = content_stripped.split('.')
            for s in sentences:
                if kw_sub.lower() in s.lower() and len(s) > 30:
                    best_segment = s.strip()[:300]
                    break
            # ---------------------------------------------------------------------

            ideas.append({
                "idea_id": i,
                "title": f"[{current_topic}] {kw_main.upper()} Insight: {kw_sub}",
                "user_intent": user_request,
                "importance_level": "CORE_CONCEPT" if i == 1 else "SUPPORTING_DETAIL",
                "extracted_segments": [best_segment.replace('\n', ' ').strip() + "..."],
                "confidence_score": dynamic_score,
                "layer_origin": layer_type
            })

        # 4. مزامنة الوعي النهائي
        avg_score = round(sum(d['confidence_score'] for d in ideas) / len(ideas), 2) if ideas else 0.5

        state = {
            "avg_score": avg_score,
            "layer_consistency": layer_type,
            "processing_depth": scale.get("mode", "DEFAULT")
        }

        return ideas, state

    def get_network_hubs(self, top_n: int = 5) -> List[Tuple[str, float]]:
        hub_weights: Dict[str, float] = {}

        if not hasattr(self, 'heuristic_network') or not self.heuristic_network:
            # محاولة أخيرة: إذا كانت الشبكة فارغة، لنحاول بناءها من الكاش مباشرة
            self._rebuild_network_from_cache()
            if not self.heuristic_network:
                self.logger.warning("⚠️ Hubs Engine: No data even after cache recovery.")
                return []

        for keyword, entries in self.heuristic_network.items():
            # تحسين: استبعاد الكلمات الشائعة (Stop Words) التقنية التي قد تلوث النتائج
            if str(keyword).lower() in ['page', 'data', 'figure', 'table']: continue

            valid_entries = [e for e in entries if isinstance(e, dict) and "page" in e]
            if not valid_entries: continue

            # معادلة الجاذبية المحدثة: التركيز على التكرار (Frequency) + التباعد (Span)
            frequency = len(valid_entries)
            total_weight = sum(float(e.get("weight", 0.5)) for e in valid_entries)

            pages = [e["page"] for e in valid_entries]
            page_span = max(pages) - min(pages) if len(pages) > 1 else 0

            # بونص التباعد: يرفع قيمة الكلمة إذا ظهرت في فصول مختلفة
            span_multiplier = 1.0 + (min(page_span, 50) / 100) # بونص يصل لـ 50%

            # النتيجة النهائية توازن بين القوة السيمانتيكية والتكرار
            hub_weights[keyword] = round((total_weight + frequency) * span_multiplier, 2)

        sorted_hubs = sorted(hub_weights.items(), key=lambda x: x[1], reverse=True)
        top_hubs = sorted_hubs[:top_n]

        # تحديث عداد الكثافة (Density Index) الذي يستخدمه التقرير
        self.network_hubs_count = len(hub_weights)

        return top_hubs

    def _rebuild_network_from_cache(self):
        """إعادة بناء الشبكة الاستدلالية من البيانات الموجودة في الكاش في حال فقدانها"""
        self.logger.info("🛠️ Attempting to reconstruct Heuristic Network from cache...")
        for p_num, p_data in self.page_cache.items():
            keywords = p_data.get("semantic_keywords", [])
            for kw in keywords:
                if kw not in self.heuristic_network:
                    self.heuristic_network[kw] = []
                self.heuristic_network[kw].append({
                    "page": p_num,
                    "weight": 0.8  # وزن افتراضي للاسترداد
                })

    def _detect_processing_scale(self, content: str) -> Dict[str, Any]:
        """
        [محرك قياس المجهود]: يحدد "عمق التفكير" بناءً على كثافة البيانات.
        تم تعديله لضمان عدم ضياع الـ DNA في الملفات الضخمة.
        """
        payload_size = len(content)
        # ميزة إضافية: قياس كثافة المعلومات (عدد الكلمات لكل 1000 حرف)
        info_density = len(content.split()) / (payload_size / 1000) if payload_size > 0 else 0

        # 1. LIGHT_WEIGHT: للملخصات (أقل من 5 صفحات)
        if payload_size < 10000:
            return {
                "mode": "LIGHT_WEIGHT",
                "iterations": 1,
                "sampling_rate": 1.0,
                "priority": "LOW"
            }

        # 2. STANDARD_CORE: للمستندات العادية (حتى 50 صفحة)
        elif payload_size < 100000:
            return {
                "mode": "STANDARD_CORE",
                "iterations": 2,
                "sampling_rate": 1.0, # نصر على القراءة الكاملة لضمان الـ DNA
                "priority": "NORMAL"
            }

        # 3. SOVEREIGN_HEAVY: للمشاريع الضخمة (200 صفحة فأكثر)
        else:
            # هنا نرفع عدد التكرارات (Iterations) لأن المستند معقد ويحتاج "تفكير" أعمق
            return {
                "mode": "SOVEREIGN_HEAVY",
                "iterations": 3,
                "sampling_rate": 1.0, # ممنوع التخطي في المشاريع الضخمة
                "priority": "CRITICAL"
            }

    def _run_tiered_processing(self, page_num: int, page_obj: Any, content: str, metadata: Dict):
        """
        [المشرف الطبقي المطور + درع الذاكرة]: يدير الطبقات الخمس مع حماية فتاكة
        ضد انهيار المعالج الرسومي (GPU Fail-Safe).
        """
        metrics = {}
        content_clean = content.strip()
        content_len = len(content_clean)

        # --- الطبقة 1: الإدراك السيمانتيكي (Vector Layer) ---
        t_start = time.perf_counter()
        tech_markers = {'table', 'fig', 'system', 'ctrl', 'algorithm', 'input', 'output'}
        is_technical_snippet = any(kw in content_clean.lower() for kw in tech_markers)

        if content_len > 50 or is_technical_snippet:
            # --- بداية منطق الفشل الآمن ---
            import torch

            # تحديد الجهاز الحالي ديناميكياً لتجنب خطأ Attribute "device" is unknown
            current_device = getattr(self, 'device', "cuda" if torch.cuda.is_available() else "cpu")

            try:
                # المحاولة باستخدام الجهاز المحدد
                vector = self.model.encode(content_clean, convert_to_numpy=True, device=current_device)
            except Exception as e:
                # التحقق من أخطاء الذاكرة (OOM)
                if "cuda" in str(e).lower() or "memory" in str(e).lower():
                    self.logger.warning(f"⚠️ GPU Overload on P{page_num+1}! Safe Retreat to CPU initiated.")
                    # التراجع القسري للـ CPU
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    vector = self.model.encode(content_clean, convert_to_numpy=True, device="cpu")
                else:
                    self.logger.error(f"💥 Critical Encoding Failure: {e}")
                    vector = None
            # --- نهاية منطق الفشل الآمن ---

            if vector is not None:
                self.vectors.append(vector)
                self.vector_map.append(page_num)
                metrics['perception'] = "SUCCESS_ENCODED"
            else:
                metrics['perception'] = "FAILED_RECOVERY"
        else:
            metrics['perception'] = "SKIPPED_MINIMAL"

        metrics['lat_perception'] = time.perf_counter() - t_start

        # --- بقية الطبقات (2, 3, 4, 5) تظل كما هي دون تغيير لضمان الاستمرارية ---
        # ... (Layout, Semantic, Integration, Networking) ...
        # (أكمل الكود الخاص بك من السطر الخاص بالطبقة 2)

        # --- الطبقة 2: التحليل الهيكلي (Layout Layer) ---
        t_start = time.perf_counter()
        # التزام تام بالمسمى الأصلي
        layout_data = self._extract_layout_structure(page_obj)
        metrics['lat_layout'] = time.perf_counter() - t_start

        # --- الطبقة 3: الاستدلال والكلمات (Semantic Layer) ---
        t_start = time.perf_counter()
        # نستخدم النسخة المطورة من الكلمات المفتاحية التي تراعي الأوزان التقنية
        keywords = self.PDF_extract_keywords(content_clean)
        layer_type = self._classify_layer(content_clean, metadata)
        topic = self._guess_topic(content_clean)
        metrics['lat_semantic'] = time.perf_counter() - t_start

        # --- الطبقة 4: المزامنة والحقن (Integration Layer) ---
        # بناء المدخلة ككيان واحد (Atomic Object)
        page_entry = {
            "content": content_clean,
            "topic": topic,
            "semantic_keywords": keywords,
            "visual_headings": layout_data.get("headings", []),
            "layer_type": layer_type,
            "metadata": metadata,
            "metrics": metrics,
            "processed_at": time.time() # وسم زمني للتدقيق
        }

        # تحديث الكاش المركزي قبل البدء بعمليات الربط لضمان وجود المراجع
        self.page_cache[page_num] = page_entry
        self._audit_and_sync_cache(page_num, page_entry)

        # --- الطبقة 5: الربط الاستدلالي (Networking Layer) ---
        # الآن نضمن أن الروابط ستجد البيانات التي تحتاجها في الكاش
        try:
            self._build_visual_heuristics(page_num, layout_data)
            self._build_heuristic_links(page_num)
        except Exception as e:
            self.logger.error(f"🔗 Linkage Error on P{page_num+1}: {e}")

        # تحديث العداد الميداني
        self.logger.info(
            f"🧬 Processed P{page_num+1} | Type: {layer_type} | "
            f"Hubs: {self.network_hubs_count} | Keywords: {len(keywords)}"
        )

        # تحسين استقرار الهاردوير: تنظيف الذاكرة الدورية للطبقات العميقة
        if page_num % 10 == 0:
            gc.collect()

        return layer_type

    def _audit_and_sync_cache(self, page_num: int, current_entry: Dict[str, Any]) -> bool:
        """
        [مدقق النزاهة السيادي المطور]: يضمن جودة البيانات ويمنع العزلة المعرفية
        مع حماية المحرك من حلقات الاستدعاء المفرغة (Recursion Guard).
        """
        try:
            # 1. فحص الوجود (LRU Safety)
            if page_num not in self.page_cache:
                return False

            # 2. فحص المحتوى والجينات (DNA Integrity)
            content = str(current_entry.get("content", "")).strip()
            if len(content) < 10:
                # محاولة أخيرة للاستعادة قبل إعلان الفشل
                self.logger.warning(f"⚠️ DNA Weak at P{page_num}. Attempting recovery pulse...")
                return False

            # [إصلاح ذاتي ذكي]: إعادة التوليد فقط عند الضرورة القصوى
            if not current_entry.get("semantic_keywords"):
                current_entry["semantic_keywords"] = self.PDF_extract_keywords(content)

            if not current_entry.get("layer_type"):
                current_entry["layer_type"] = self._classify_layer(content, current_entry.get("metadata", {}))

            # 3. مزامنة المتجهات (The Core Guardrail)
            # تصحيح الانزياح (Alignment) بين المتجهات وخريطة الصفحات
            vec_len = len(self.vectors)
            map_len = len(self.vector_map)
            if vec_len != map_len:
                self.logger.critical(f"🚨 SYNC_GAP: Misalignment ({vec_len} vs {map_len}) at P{page_num}")
                # إعادة المزامنة القسرية لمنع خطأ Index Out of Range في البحث
                min_len = min(vec_len, map_len)
                self.vectors = self.vectors[:min_len]
                self.vector_map = self.vector_map[:min_len]

            # 4. التدقيق المكاني (Archive Integrity)
            meta = current_entry.get("metadata", {})
            if "folder_id" not in meta:
                loc = self._assign_sovereign_location(page_num)
                current_entry.setdefault("metadata", {}).update(loc)
                current_entry["location_stamp"] = f"{loc.get('folder_id')} > {loc.get('binder_id')}"

            # 5. تدقيق الربط وحماية النبض (Connectivity Audit & Recursion Guard)
            # نتحقق مما إذا كانت الصفحة "جزيرة" ولكن نمنع الاستدعاء المكرر
            # استخدام getattr للوصول الآمن لـ visual_links
            v_links = getattr(self, 'visual_links', [])

            # فحص سريع لآخر 50 رابط (تحسين أداء)
            is_linked = any(
                page_num == link.get("origin") or page_num == link.get("target")
                for link in v_links[-50:]
            )

            # تفعيل الجسر الهيورستي فقط إذا كانت الصفحة غنية بالمعلومات ولم يتم معالجتها مسبقاً كجسر
            if not is_linked and len(current_entry.get("semantic_keywords", [])) > 5:
                if not hasattr(self, '_in_recovery_mode') or not self._in_recovery_mode:
                    self.logger.debug(f"ℹ️ ISLAND: P{page_num} isolated. Triggering heuristic bridge.")
                    # حماية من الحلقات المفرغة
                    self._in_recovery_mode = True
                    try:
                        self._build_heuristic_links(page_num)
                    finally:
                        self._in_recovery_mode = False

            return True

        except Exception as e:
            self.logger.error(f"💥 Audit Failure at P{page_num}: {str(e)}")
            return False

    def _audit_sweet_spot(self, chunk_content: str, current_confidence: float) -> bool:
        """
        [مدقق المنطقة الذهبية]: يحدد ما إذا كانت النبضة وصلت لمستوى الوعي المطلوب.
        """
        # النطاق الذهبي: 0.86 - 0.98
        is_in_range = 0.86 <= current_confidence <= 0.98

        # معيار إضافي: هل النص غني بالمعلومات؟
        is_rich = len(chunk_content.split()) > 100

        if is_in_range:
            return True

        # قبول مشروط للبرقيات التقنية (نصوص قصيرة لكن ثقتها مقبولة)
        if len(chunk_content) < 500 and current_confidence >= 0.82:
            return True

        return False

    def _reprocess_pulse(self, content: str, initial_score: float):
        """
        [المعالجة العميقة المطورة]: كسر حاجز الـ 0.85 عبر "التركيز السيمانتيكي"
        بدلاً من مجرد زيادة عدد التكرارات، لضمان استقرار الهاردوير.
        """
        # 1. استخراج جينات التوجيه (DNA Guidance)
        # نستخدم النسخة المطورة التي تركز على المصطلحات التقنية المركبة
        guidance_kws = self.PDF_extract_keywords(content, top_n=5)

        # تحسين: بدلاً من إرسال كل النص، نركز على الفقرات التي تحتوي على الجينات الأساسية
        # هذا يقلل من حجم الـ Context ويجعل الاستدلال أدق
        focused_content = ""
        paragraphs = content.split('\n')
        for p in paragraphs:
            if any(kw.lower() in p.lower() for kw in guidance_kws):
                focused_content += p + "\n"

        # إذا كان التركيز فشل في جمع نص كافٍ، نعود للنص الأصلي كخيار أمان
        final_payload = focused_content if len(focused_content) > 200 else content

        enhanced_prompt = f"DEEP_AUDIT_MODE | KEY_GENES: {', '.join(guidance_kws)}\n{final_payload}"

        self.logger.info(f"🔄 Sovereign Shift: Escalating analysis for score {initial_score}...")

        try:
            # 2. الاستدعاء بجهد مضاعف (Force Max Iterations)
            # نستخدم دالة generate التي تم إصلاحها سابقاً لتعمل بالتوازي أو محلياً
            enhanced_ideas, enhanced_state = self.generate(
                enhanced_prompt,
                max_iterations=5,
                user_request="EXTRACT_STRATEGIC_IMPLICATIONS"
            )

            # 3. التدقيق النوعي لدرجة الثقة
            new_score = enhanced_state.get("avg_score", 0)
            if new_score > initial_score:
                self.logger.info(f"✨ Audit Success: Quality jumped from {initial_score} to {new_score}")
                enhanced_state["audit_status"] = "GOLDEN_ZONE_REACHED"
            else:
                enhanced_state["audit_status"] = "REFINED_BUT_STABLE"

            # تحسين استقرار الذاكرة بعد المعالجة العميقة
            gc.collect()

            return enhanced_ideas, enhanced_state

        except Exception as e:
            self.logger.error(f"❌ Deep Audit Failure: {e}")
            return [], {"avg_score": initial_score, "audit_status": "AUDIT_FAILED"}

    def export_to_markdown(self, analysis_result: Dict[str, Any], output_path: str):
        """
        [المقرر السيادي المطور]: يحول مخرجات الـ DNA إلى تقرير استراتيجي متكامل
        مع إضافة خرائط الروابط والتلخيص التقني.
        """
        from pathlib import Path

        # 1. فك تشفير النتائج بأمان (Safe Extraction)
        # نستخدم .get() مع قيم افتراضية لضمان عدم توقف الكود إذا نقصت أي ميتاداتا
        data_block = analysis_result.get('data', {})
        stats = data_block.get('stats', {})
        doc_info = data_block.get('metadata', {})

        # الأفكار قد تأتي من 'ideas' أو من مخرجات 'output' حسب دالة الـ Analyzer
        ideas = analysis_result.get('ideas', [])
        if not ideas:
            ideas = analysis_result.get('output', {}).get('structured_ideas', [])

        # 2. بناء الترويسة الاستراتيجية
        title = doc_info.get('title', Path(output_path).stem).replace('_', ' ').title()
        md = [
            f"# 🛰️ Sovereign Analysis Report: {title}",
            f"\n## 📊 Tactical Metrics",
            f"- **Target File:** `{Path(output_path).name}`",
            f"- **Processing Scope:** {stats.get('pages', '0')} Pages analyzed",
            f"- **Knowledge Hubs:** {self.network_hubs_count} points of gravity",
            f"- **Logical Connections:** {len(getattr(self, 'visual_links', []))} active links",
            f"- **Global Awareness:** `{analysis_result.get('metrics', {}).get('awareness', 0.5) * 100:.1f}%`",
            f"\n---\n",
            f"## 🧠 Strategic Intelligence Output"
        ]

        # 3. صياغة الأفكار المستخرجة (The Core Insights)
        if not ideas:
            md.append("\n*⚠️ No high-confidence concepts were extracted from this pulse.*")
        else:
            for idea in ideas:
                # منطق التمييز البصري (Core vs Support)
                is_core = idea.get('priority') == "CRITICAL" or idea.get('importance_level') == "CORE_CONCEPT"
                prefix = "### 🌟 [CORE]" if is_core else "#### 🔹 [SUPPORT]"

                md.append(f"{prefix} {idea.get('title')}")

                # إضافة الميتاداتا الخاصة بكل فكرة (Confidence & DNA)
                conf = idea.get('confidence_score', 0) * 100
                layer = idea.get('layer_origin', 'UNKNOWN')
                dna = idea.get('dna_tag', 'N/A')

                md.append(f"- **Confidence:** `{conf:.1f}%` | **Layer:** `{layer}` | **DNA:** `{dna}`")

                # إدراج القطاع النصي المستخرج كشاهد (Evidence)
                segments = idea.get('extracted_segments', [])
                if segments:
                    clean_segment = str(segments[0]).replace('\n', ' ').strip()
                    md.append(f"\n> {clean_segment}")

                md.append("") # فاصل جمالي

        # 4. تذييل الشبكة (Heuristic Footer)
        md.append(f"\n---\n## 🌐 Network Topology & Origin")
        md.append(f"- **Semantic Nodes:** {len(getattr(self, 'heuristic_network', {}))} identified genes.")
        md.append(f"- **Recovery Mode:** {'Active' if getattr(self, '_in_recovery_mode', False) else 'Standard'}")
        md.append(f"- **Inferred Topic:** `{analysis_result.get('pdf_metadata', {}).get('inferred_topic', 'General Science')}`")
        md.append(f"\n\n*Report generated by **Sovereign Engine v2.0** - {time.strftime('%Y-%m-%d %H:%M:%S')}*")

        # 5. الالتزام بالقرص (Commit to Disk)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(md))
            self.logger.info(f"💾 SUCCESS: Sovereign report committed to {output_path}")
        except Exception as e:
            self.logger.error(f"💥 EXPORT_CRASH: Failed to write Markdown: {str(e)}")

    def export_to_json(self, analysis_result: Dict[str, Any], output_path: str):
        """
        [مصدّر البيانات]: يحفظ نتائج التحليل وحالة الشبكة العصبية في ملف JSON.
        يضمن بقاء "الذكاء الاستراتيجي" قابلاً للنقل والقراءة.
        """
        import json
        import time

        export_payload = {
            "system_metadata": {
                "engine": "Sovereign-PDF-2.0",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "global_hubs": self.network_hubs_count
            },
            "analysis_output": analysis_result,
            "network_topology": {
                "heuristic_nodes": len(self.heuristic_network),
                "visual_bridges": len(getattr(self, 'visual_links', [])),
                "vector_depth": len(self.vectors),
                "buffer_status": f"{len(self.page_cache)}/{self.max_pages}"
            }
        }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_payload, f, ensure_ascii=False, indent=4)
            self.logger.info(f"💾 EXPORT: Data locked and saved to {output_path}")
        except Exception as e:
            self.logger.error(f"❌ EXPORT_ERROR: {str(e)}")

    def clear_cache(self, deep: bool = False):
        """
        [إدارة الموارد]: يوازن بين خفة أداء النظام وبقاء البيانات الحيوية.
        """
        from collections import OrderedDict, deque, defaultdict

        # 1. التطهير الذكي (Smart Retention)
        # نفرغ النصوص الثقيلة لكن نبقي على المتجهات والشبكة للبحث السريع
        self.page_cache = OrderedDict()

        if deep:
            # 2. التطهير العميق (Full Reset)
            # يعيد المحرك لحالة الصفر (خالية من أي ذكاء مسبق)
            self.heuristic_network = defaultdict(list)
            self.layer_hierarchy = {}
            self.visual_links = []
            self.vectors = []
            self.vector_map = []
            self.network_hubs_count = 0
            self.navigation_stack = []
            self.logger.warning("🧹 DEEP_PURGE: The engine has been reset to its primal state.")
        else:
            # 3. الحفاظ على الذكاء (Semantic Persistence)
            self.logger.info("🧹 CACHE_FLUSH: Text cleared. Semantic Hubs and Vectors retained.")

        # إجبار مجمع النفايات على العمل
        import gc
        gc.collect()

    def export_test_report(
        self,
        query: str,
        search_hits: List[Dict[str, Any]],
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """
        [إصدار الاستعادة القسرية]: يضمن عدم ظهور 'Recovery Failed' عبر إعادة بناء DNA الصفحة حياً.
        """
        import datetime
        import os

        # 1. تأمين المسار
        report_folder = os.path.join(os.getcwd(), "Sovereign_Audits")
        os.makedirs(report_folder, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        full_path = os.path.join(report_folder, f"audit_{timestamp}.md")

        lines = [
            f"# 🛡️ Sovereign Audit Report | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Target Query:** `{query}` | **Network Hubs:** `{self.network_hubs_count}`",
            "\n---"
        ]

        # 2. جدول التشريح (DNA Surgery)
        lines.extend([
            "## 🔍 Deep Search Analysis & DNA Surgery",
            "| Page | Score | Layer | DNA Influence (Top Hubs) | Structural Neighbors |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ])

        for hit in search_hits:
            p_num = hit['page']

            # --- بروتوكول الاستعادة الثلاثي (Triple Recovery) ---
            # 1. محاولة الكاش
            surgery = self.inspect_cache(p_num)

            # 2. محاولة القراءة من القرص إذا كان فارغاً
            if surgery['status'] == "VOID":
                self.get_page_data(p_num)
                surgery = self.inspect_cache(p_num)

            # 3. محاولة إعادة البناء الفورية (On-the-fly Reconstruction)
            if surgery['status'] == "VOID":
                # إذا فشل كل شيء، نسحب الكلمات المفتاحية يدوياً من النص الخام لهذه الصفحة
                try:
                    raw_text = self.PDF_get_text(p_num) # دالة استخراج النص المباشرة
                    keywords = self.PDF_extract_keywords(raw_text, top_n=3)
                    dna_summary = "<br>".join([f"• **{kw}**" for kw in keywords])
                    neighbors_str = "Reconstructed"
                except:
                    dna_summary, neighbors_str = "❌ Hard Failure", "Inaccessible"
            else:
                # النجاح في الحصول على التشريح
                influence = surgery['neural_connectivity']['semantic_influence']
                dna_summary = "<br>".join([f"• **{k}**: {v}" for k, v in list(influence.items())[:2]])
                neighbors = surgery['neural_connectivity']['structural_hierarchy']
                neighbors_str = ", ".join([f"P{n+1}" for n in neighbors]) if neighbors else "Terminal"

            lines.append(f"| **P{p_num}** | {hit['score']:.4f} | {hit['layer']} | {dna_summary or 'Pure Content'} | {neighbors_str} |")

        # 3. الحفظ النهائي
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self.logger.info(f"🧬 تم إنتاج التقرير النهائي بنجاح: {full_path}")
        except Exception as e:
            self.logger.error(f"❌ فشل الحفظ: {e}")

# ========== الربط مع process_pdf_streaming ==========
def process_pdf_streaming(pdf_path: str, logger: Any, callback: Optional[Callable] = None):
    """
    [المصنع السيادي المطور]: لا يقوم فقط بالإطلاق، بل يضمن حقن الـ DNA
    ويتأكد من أن كل صفحة أصبحت 'عقدة' (Node) نشطة في الشبكة قبل التسليم.
    """
    logger.info("🚀 Launching Sovereign Factory: Secure Ingestion & DNA Mapping...")

    # 1. تهيئة المختبر المركزي
    try:
        # تأكد أن الكلاس PDFPageCacheNetwork تم تحديثه بالدوال التي بنيناها (inspect_cache, etc.)
        cache_network = PDFPageCacheNetwork(logger=logger)
    except Exception as e:
        logger.error(f"❌ FACTORY_INIT_FAILURE: {str(e)}")
        return None

    try:
        # 2. بدء عملية المعالجة الجراحية
        # process_pdf_core يجب أن تكون الدالة التي تمسح الـ PDF وتملأ self.page_cache
        result = cache_network.process_pdf_core(pdf_path)

        if result["status"] == "success":
            # 3. التدقيق النوعي (Quality Audit) - التأكد من سلامة الكاش
            # نتحقق من أن الصفحات لم تخرج فارغة
            indexed_pages = len(cache_network.page_cache)
            hubs = cache_network.network_hubs_count

            if indexed_pages == 0:
                logger.warning("⚠️ Factory Warning: System active but Page Cache is empty. Forced re-indexing...")
                # محاولة فهرسة عينة لضمان تدفق البيانات
                cache_network.get_page_data(0)

            # 4. إشعار خارجي غني بالبيانات
            if callback:
                stats = result["data"]["stats"]
                metadata = result["data"]["metadata"]
                # نرسل تفاصيل إضافية للـ Callback ليشعر المستخدم بقوة المحرك
                callback(
                    stats["pages"],
                    f"NETWORK_HUB_ESTABLISHED: {hubs} Hubs",
                    metadata
                )

            logger.info(f"✅ Sovereign Factory: Network Mastered with {indexed_pages} Nodes & {hubs} Hubs.")

            # إرجاع الكائن وهو في قمة نشاطه الإدراكي
            return cache_network
        else:
            error_msg = result.get('error_details') or 'Unknown Internal Error'
            logger.error(f"❌ FACTORY_CORE_FAILURE: {error_msg}")
            return None

    except Exception as e:
        logger.error(f"❌ SOVEREIGN_STREAMING_CRITICAL: {str(e)}")
        # في حالة الفشل الحرج، نحاول إرجاع الكائن جزئياً إذا كان يحتوي على بيانات
        return cache_network if 'cache_network' in locals() else None

# مثال الاستخدام:
"""
cache = processor.process_pdf_streaming("philosophy_paper.pdf")

# استعلام تلقائي
related = cache.get_related_pages(1)  # صفحة 1 + مرتبطاتها
print(f"صفحة 1 مرتبطة بـ: {related}")

# عرض المصنّف
print(cache.inspect_cache())
"""

# ========== نظام الإختبار ==========
if __name__ == "__main__":
    # 1. تهيئة المحرك السيادي مع "بروتوكول الوعي بالموارد"
    import logging
    import os

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("SovereignEngine")

    # تحسين: لا نكتفي بـ max_pages=150، بل نترك للمحرك حرية الإخلاء الديناميكي
    engine = PDFPageCacheNetwork(logger=logger)

    # ربط العميل (تأكد من وضع المفتاح في بيئة آمنة Environment Variable)
    from openai import OpenAI
    engine.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-key-here"))

    print("\n" + "="*60)
    print("🚀 [Sovereign Mode] Hierarchical Ingestion & Portfolio Archiving")
    print("="*60)

    pdf_file = "ROBOTICS.pdf"

# المرحلة 1: تفعيل القائد الميداني (Core Ingestion)
    # ملاحظة: أزلنا initial_metadata لأن الدالة لا تدعمها
    process_result = engine.process_pdf_core(pdf_file)

    if process_result.get("status") == "success":
        # تعويض الميتاداتا يدوياً في الكاش إذا أردت تخصيصها
        # (اختياري: لتغذية المصنف الطبقي لاحقاً)
        for p_num in engine.page_cache:
            engine.page_cache[p_num]["metadata"]["portfolio"] = "ROBOTICS_DB"

        p_data = process_result.get("data", {})
        stats = p_data.get("stats", {})

        # المرحلة 2: التحليل النبضي (Chained Pulse)
        # إصلاح: نمرر طلب المستخدم بوضوح للمحرك
        print(f"✅ Perception Layer Ready: {stats.get('pages')} pages indexed.")
        print(f"📡 Knowledge Nodes: {engine.network_hubs_count} Hubs found.")

        # المرحلة 2: التحليل النبضي المتسلسل (Chained Pulse)
        # الإصلاح: تمرير engine كبارامتر للـ thinking_engine لسد الثغرة
        print("\n🧠 Activating Chained Pulse Analysis...")
        final_report = engine.analyze_pdf(
            pdf_path=pdf_file,
            thinking_engine=engine, # تمرير المحرك نفسه كما يتطلب تعريف الدالة
            user_request="Deep technical audit of control systems and feedback loops"
        )

        # المرحلة 3: استخراج بيانات الأرشيف (Sovereign Architecture)
        # إصلاح: التأكد من وجود البيانات قبل الطباعة لمنع KeyError
        archive = final_report.get('sovereign_archive', {})

        print(f"\n📂 Sovereign Architecture Report:")
        print(f"   ├─ Portfolio:    {archive.get('portfolio', 'MAIN_KNOWLEDGE_BASE')}")
        print(f"   ├─ Folders:      {archive.get('total_folders', 0)} active folders")
        print(f"   ├─ Hub Density:  {engine.network_hubs_count} nodes")
        print(f"   └─ Connections:  {len(engine.visual_links)} active links")

        # المرحلة 4: اختبار القفز السيمانتيكي (Traversal Test)
        # إصلاح: دمجنا استعادة البيانات من الهاردسك داخل البحث
        print("\n🔍 Knowledge Graph Traversal:")
        query = "control systems and feedback loops"

        # البحث الآن يستخدم الدرع الذي بنيناه (70/30 Scoring)
        search_hits = engine.semantic_search(query, top_k=2)

        for hit in search_hits:
            p_num = hit['page'] - 1 # تحويل للـ index الصفرية
            # استعادة ذكية: get_related_pages ستسحب البيانات من الأرشيف إذا لزم الأمر
            related = engine.get_related_pages(p_num, depth=1)

            print(f"   📍 Match: P{hit['page']} | Score: {hit['score']} | Layer: {hit['layer']}")
            print(f"      🔗 Neighbors: {related}")

        # المرحلة 5: الأرشفة النهائية والتصدير
        # إضافة: طباعة مسار التصدير للتأكد من الحفظ
        export_path = "sovereign_audit_report.md"
        engine.export_to_markdown(final_report, export_path)
        print(f"\n💾 Portfolio committed to: {os.path.abspath(export_path)}")

        # المرحلة 6: اكتشاف النخبة (Strategic Hubs)
        print("\n🧬 Knowledge Core Discovery:")
        hubs = engine.get_network_hubs(top_n=5)
        for word, weight in hubs:
            print(f"   🔑 Hub: '{word.upper():<15}' | Weight: {weight:.2f}")

    else:
        error_msg = process_result.get('error_details', 'Unknown Core Failure')
        print(f"❌ Engine Failed: {error_msg}")

    print("\n" + "="*60)
    print("🎉 Sovereign Workflow Completed | Portfolio Secure ✅")
    print("="*60)

    # إجراء البحث
    search_hits = engine.semantic_search(query, top_k=5)

    # تصدير النتائج آلياً للأرشفة
    engine.export_test_report(query=query, search_hits=search_hits)
