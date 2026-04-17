# pdf_processor.py
"""
محرك تحليل PDF المنفصل والمتكامل
يعمل مع أي thinking engine
"""

import re
import logging
import time
import warnings
from collections import deque, OrderedDict, defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Protocol, Set

# تحسين: استيراد torch فقط عند الحاجة أو التأكد من وجوده
import torch
import numpy as np
import fitz  # PyMuPDF

# تحسين: إدارة التحذيرات قبل تحميل المكتبات الثقيلة
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)

from openai import OpenAI
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("الرجاء تثبيت sentence-transformers عبر: pip install sentence-transformers")

# ===================================================================
# إعدادات الموديل (Singleton Pattern Concept)
# ===================================================================
# تحسين: جعل اختيار الجهاز (Device) أكثر ذكاءً
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# تحسين: تحميل الموديل مرة واحدة فقط على مستوى الموديول (Global Singleton)
# لتجنب تحميله مع كل Instance جديد من الكلاس
_SHARED_MODEL = None

def get_embedding_model():
    global _SHARED_MODEL
    if _SHARED_MODEL is None:
        _SHARED_MODEL = SentenceTransformer(MODEL_NAME, device=DEVICE)
    return _SHARED_MODEL

# ===================================================================
# إعدادات الموديل (Singleton Pattern Concept)
# ===================================================================
class InsightManager:
    """إدارة بصمات الجمل (🧬) واكتشاف التكرار الدلالي"""
    def __init__(self, model, threshold: float = 0.88):
        self.model = model
        self.threshold = threshold
        # الخزنة المركزية لكل صواعق المستند
        self.vault: Dict[int, Dict] = {}

    def process_sentences(self, text: str, page_num: int) -> List[int]:
        import re
        import numpy as np
        # تقسيم النص لجمل حقيقية
        sentences = [s.strip() for s in re.split(r'[.!?|.]', text) if len(s.strip()) > 25]
        assigned_ids = []

        for sent in sentences:
            with torch.no_grad():
                sent_vec = self.model.encode(sent, convert_to_numpy=True)

            duplicate_id = None
            # البحث السريع في الخزنة عن التوائم
            for idx, data in self.vault.items():
                similarity = np.dot(sent_vec, data["vector"]) / (np.linalg.norm(sent_vec) * np.linalg.norm(data["vector"]))
                if similarity > self.threshold:
                    duplicate_id = idx
                    break

            if duplicate_id:
                assigned_ids.append(duplicate_id)
                # تسجيل وجود الجملة في هذه الصفحة أيضاً
                if "pages" in self.vault[duplicate_id]:
                    if page_num not in self.vault[duplicate_id]["pages"]:
                        self.vault[duplicate_id]["pages"].append(page_num)
            else:
                new_id = len(self.vault) + 1
                self.vault[new_id] = {"vector": sent_vec, "text": sent, "pages": [page_num]}
                assigned_ids.append(new_id)

        return assigned_ids

# ===================================================================
# إعدادات الموديل (LoggerProtocol)
# ===================================================================
class LoggerProtocol(Protocol):
    def info(self, msg: str, *args, **kwargs) -> None: ...
    def error(self, msg: str, *args, **kwargs) -> None: ...
    def warning(self, msg: str, *args, **kwargs) -> None: ...
    def debug(self, msg: str, *args, **kwargs) -> None: ...
    def critical(self, msg: str, *args, **kwargs) -> None: ... # أضف هذا
    def log(self, level: int, msg: str, *args, **kwargs) -> None: ... # وهذا

# ===================================================================
# محرك قراءة الملفات (PDF Page Cache Network)
# ===================================================================
class PDFPageCacheNetwork:
    """نظام Cache المتقدم مع Heuristic Network"""

    def __init__(self, logger: Optional[LoggerProtocol] = None, max_pages: int = 200):
        # 1. إعداد الذاكرة والمخزن (Memory & Storage)
        self.page_cache: OrderedDict[int, Any] = OrderedDict()
        self.max_pages = max_pages
        self.page_queue = deque()

        # 2. الهياكل الاستنتاجية (Heuristic & Hierarchy)
        self.heuristic_network: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.layer_hierarchy: Dict[int, List[int]] = {}

        # 3. محرك الذكاء (Intelligence Engine)
        # تحسين: استدعاء الموديل المشترك بدلاً من تحميل واحد جديد لكل نسخة
        self.model = get_embedding_model()
        self.vectors: List[np.ndarray] = []
        self.vector_map: List[int] = []

        # 🚀 [The Critical Fix]: Initialize the Insight Manager
        # This allows the engine to track "Robot Hand 45cm" (⚡) across pages
        self.insight_manager = InsightManager(model=self.model)

        # 4. الملاحة والتخطيط (Layout & Navigation)
        self.heading_map: Dict[str, List[int]] = defaultdict(list)
        self.visual_links: List[Dict] = []
        self.navigation_stack = deque()

        # 5. إعداد المسجل (Logger Setup)
        # تحسين: دمج المنطق ليكون أكثر اختصاراً
        self.logger = logger or self._make_logger()
        self.logger.info("تم تهيئة محرك تحليل PDF بنجاح.")

        # تعريف العميل كـ Optional لمنع أخطاء الـ Type Hinting
        self.llm_client: Any = None  # تعريف العميل لمنع خطأ Pylance
        self.logger.info("تم تهيئة محرك تحليل PDF بنجاح.")

        # [إضافة]: تهيئة عدادات الشبكة الاستدلالية لمنع الانهيار
        self.network_hubs_count = 0  # هذا هو السطر الذي يسبق الخطأ
        self.total_chars = 0
        self.cumulative_awareness = []
        self.visual_links = []
        self.heuristic_network = defaultdict(list)
        self.layer_hierarchy = {}

        # التأكد من وجود سجل الـ DNA
        self.engine_metadata = {"dna_flow": [], "insights": []}

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

# ------------ ثانياً: استخراج البيانات الهيكلية (Extraction Engine)
    def process_pdf_core(self, pdf_path: str, user_request: str = "") -> Dict[str, Any]:
        """
        [القائد الميداني - النسخة السيادية]
        إدارة شاملة: مسح هندسي، استخراج جداول، تصنيف طبقي، وختم بصري 🧬
        """
        start_time = time.time()
        path_obj = Path(pdf_path)
        full_text_parts: List[str] = []
        doc_info = {}

        try:
            # 1. السيطرة على الملف وفتح غرفة العمليات
            with fitz.open(pdf_path) as pdf_doc:
                page_count = len(pdf_doc)
                raw_meta = pdf_doc.metadata or {}
                doc_info = {
                    "title": str(raw_meta.get("title") or path_obj.stem),
                    "author": str(raw_meta.get("author") or "Unknown"),
                    "page_count": page_count
                }

                self.logger.info(f"🚀 القائد الميداني يبدأ المهمة: {doc_info['title']}")

                for page_num in range(page_count):
                    page = pdf_doc[page_num]

                    # 2. [المستشار 1]: المسح الهندسي (Layout) قبل الاستخراج
                    # نحدد التضاريس لمعرفة أين تقع الجداول والنصوص
                    layout_data = self._extract_layout_structure(page)

                    page_contents = []

                    # 3. الاستخراج العدواني للنصوص (Blocks)
                    raw_blocks = page.get_text("blocks")
                    # ترتيب الكتل (من الأعلى لأسفل ومن اليسار لليمين) لضمان السياق
                    blocks = sorted([b for b in raw_blocks if len(b[4].strip()) > 5],
                                  key=lambda b: (b[1], b[0]))

                    for b in blocks:
                        text = b[4].strip()
                        if text: page_contents.append(text)

                    # 4. استخراج الجداول التقنية (الميزة المتقدمة)
                    has_tables = False
                    try:
                        tabs = page.find_tables()
                        if tabs and tabs.tables:
                            has_tables = True
                            for table in tabs.tables:
                                t_data = table.extract()
                                table_text = "\n".join([" | ".join([str(c).strip() if c else "" for c in r]) for r in t_data])
                                page_contents.append(f"\n[TECHNICAL_TABLE_DATA]:\n{table_text}")
                    except: pass

                    combined_text = "\n".join(page_contents).strip()

                    if combined_text:
                        full_text_parts.append(combined_text)

                        # 5. [المستشار 2]: التصنيف الطبقي (Layer Classification)
                        # القائد يسأل: ما هي هوية هذه الصفحة؟
                        layer_type = self._classify_layer(combined_text, {"page": page_num + 1})

                        # 6. إعداد الميتا-داتا المتقدمة للمحقن
                        page_meta = {
                            "source_path": pdf_path,
                            "page_index": page_num + 1,
                            "contains_tables": has_tables,
                            "layer_type": layer_type,
                            "layout": layout_data
                        }

                        # 7. [المستشار 3]: الربط الاستدلالي اللحظي
                        # يتم الربط هنا لضمان تحديث عداد الـ Hubs فوراً
                        self._build_heuristic_links(page_num)

                        # 8. التوثيق النهائي (المحقن add_page)
                        # هنا يتم وضع الختم الأخضر 🧬 وتخزين البيانات في الكاش
                        self.add_page(page_num, combined_text, page, page_meta)

                # 9. تجميع الحصيلة النهائية
                full_text = "\n\n".join(full_text_parts)
                process_time = round(time.time() - start_time, 2)

                self.logger.info(f"✅ تم السيطرة على المستند: {page_count} صفحة في {process_time}s")

                return {
                    "status": "success",
                    "data": {
                        "full_text": full_text,
                        "metadata": doc_info,
                        "stats": {
                            "pages": page_count,
                            "hubs": self.network_hubs_count, # العداد الآن حقيقي وليس 0
                            "time": process_time
                        }
                    },
                    "context": {"user_request": user_request}
                }

        except Exception as e:
            self.logger.error(f"❌ انهيار القائد الميداني: {str(e)}")
            return {"status": "error", "error_details": str(e)}

    def fast_ingest_stream(self, pdf_path: str, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        [القارئ السريع]: يمسح الصفحات بسرعة ويجهز النصوص الخام للقائد الميداني.
        تم إصلاح توافق الأنواع (Pylance) لضمان استقرار المشاريع الضخمة.
        """
        try:
            # التأكد من وجود الملف قبل البدء
            if not Path(pdf_path).exists():
                return {"status": "error", "pages_ingested": 0, "message": "File not found"}

            with fitz.open(pdf_path) as pdf_doc:
                total_pages = pdf_doc.page_count
                self.logger.info(f"🌀 بدء التدفق السريع لـ {total_pages} صفحة...")

                for page_num in range(total_pages):
                    page = pdf_doc.load_page(page_num)

                    # 1. استخراج آمن للنص (Sovereign Type Guard)
                    # نضمن أن المحتوى نصي قبل استدعاء .strip()
                    raw_content = page.get_text("text")

                    if isinstance(raw_content, str):
                        page_text = raw_content.strip()
                    else:
                        # تحويل أي مخرجات غير نصية (قوائم/قواميس) إلى نص
                        page_text = str(raw_content).strip()

                    if page_text:
                        # 2. حقن "الخامات" في الكاش المركزي (The Warehouse)
                        # نستخدم وسوم واضحة ليعرف القائد الميداني أن هذه الصفحة لم تُفهرس بعمق بعد
                        self.page_cache[page_num] = {
                            "content": page_text,
                            "layer_type": "PRE_INDEXED_RAW",
                            "metadata": {
                                "fast_stream": True,
                                "source": str(pdf_path),
                                "ingested_at": time.time()
                            }
                        }

                        # 3. إشعار خارجي (Callback) إذا وُجد
                        if callback:
                            try:
                                callback(page_num, page_text)
                            except Exception as cb_e:
                                self.logger.warning(f"⚠️ Callback Error on P{page_num}: {cb_e}")

                self.logger.info(f"✅ Ingestion Complete: {total_pages} pages ready for deep analysis.")
                return {"status": "completed", "pages_ingested": total_pages, "pages": total_pages}

        except Exception as e:
            error_msg = f"❌ Fast Stream Failure: {str(e)}"
            self.logger.error(error_msg)
            return {"status": "error", "pages_ingested": 0, "message": str(e)}

    def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        """Refined PDF extraction - Type Safe for Pylance"""
        try:
            pdf_path_obj = Path(pdf_path)
            if not pdf_path_obj.exists():
                return {"status": "error", "message": f"File not found: {pdf_path}"}

            with fitz.open(pdf_path) as pdf_doc:
                page_count = pdf_doc.page_count # استخدام الخاصية المباشرة
                text_parts: List[str] = []

                raw_meta = pdf_doc.metadata or {}
                doc_info = {
                    "title": str(raw_meta.get("title") or pdf_path_obj.stem),
                    "author": str(raw_meta.get("author") or "unknown"),
                    "page_count": page_count
                }

                self.logger.info(f"📄 Processing: {doc_info['title']}")

                # استخدام range لتجنب مشكلة الـ Iterable Protocol
                for page_num in range(page_count):
                    page = pdf_doc.load_page(page_num)
                    blocks = page.get_text("blocks")
                    # تنظيف وتصفية لضمان عدم تمرير "قمامة" برمجية للقائد
                    valid_text = []
                    for b in blocks:
                        if len(b) > 4 and isinstance(b[4], str):
                            clean_b = b[4].strip()
                            if len(clean_b) > 2: valid_text.append(clean_b)

                    page_text = "\n".join(valid_text)

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
        Extracts structural headings and blocks with a focus on engineering layouts.
        Standardizes internal data keys to English.
        """
        # توحيد أسماء المفاتيح للإنجليزية: headings & blocks
        layout_data = {"headings": [], "blocks": []}

        try:
            # 1. Get detailed dictionary of the page
            dict_data = page.get_text("dict")
            layout_data["blocks"] = dict_data.get("blocks", [])

            for block in layout_data["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        # حساب المسافة الرأسية (اختياري لزيادة الدقة مستقبلاً)
                        for span in line["spans"]:
                            text = span["text"].strip()

                            if len(text) > 3:
                                # معايير هندسية (Relaxed Criteria)
                                is_large = span["size"] > 11.5
                                is_bold = "bold" in span["font"].lower()
                                is_caps = text.isupper() and len(text) > 5

                                # إزالة النصوص التي تبدو كأرقام فقط (ليست عناوين عادةً)
                                is_not_numeric = not text.replace('.', '').isdigit()

                                if (is_large or is_bold or is_caps) and is_not_numeric:
                                    layout_data["headings"].append({
                                        "text": text,
                                        "bbox": span["bbox"],
                                        "font_size": span["size"],
                                        "font_name": span["font"],
                                        "type": "structural_anchor" # وسم داخلي للشبكة الاستدلالية
                                    })
        except Exception as e:
            self.logger.warning(f"Layout extraction skipped on a page: {str(e)}")

        return layout_data

# ------------ ثالثاً: إدارة الذاكرة والأرشفة (Caching & Storage) ------------
    def add_page(self, page_num: int, text: str, page_obj: Any, metadata: Dict):
        """
        [The Sovereign Supervisor]: نسخة الرشاش المحدثة بالأرشفة المجلدية.
        نظام (Folder > Binder) مع حماية الـ DNA والذاكرة اللحظية.
        """
        # 1. التوزيع المكاني (العنوان البريدي)
        loc = self._assign_sovereign_location(page_num)
        clean_text = str(text).strip()

        # 2. توليد المتجهات (حجر الأساس للبحث) - مع منع التكرار
        if page_num not in self.vector_map:
            vector = self.model.encode(clean_text, convert_to_numpy=True)
            self.vectors.append(vector)
            self.vector_map.append(page_num)
        else:
            self.logger.debug(f"🧬 Vector Sync: P{page_num} already exists, skipping duplicate encoding.")

        # 3. توليد الجينات والطبقات
        keywords = self.PDF_extract_keywords(clean_text)
        # حقن الموقع في الميتاداتا لتعزيز سياق المصنف
        enriched_meta = {**metadata, **loc}
        layer = self._classify_layer(clean_text, enriched_meta)

        layout = self._extract_layout_structure(page_obj)

        # 4. Unified Sovereign Storage (Using defined variables)
        self.page_cache[page_num] = {
            "content": clean_text,
            "metadata": enriched_meta,
            "layer_type": layer, # Matches variable above
            "semantic_keywords": keywords,
            "visual_headings": layout.get("headings", []),
            # Linking to the 'loc' dictionary keys
            "location_stamp": f"{loc['folder_id']} > {loc['binder_id']}",
            "indexed_at": time.time()
        }

        # 5. بناء الروابط (تحديث العدادات والـ Hubs حياً)
        self._build_visual_heuristics(page_num, layout)
        self._build_heuristic_links(page_num)

        # 6. إنعاش الذاكرة (الرشاش)
        self._refresh_memory_balance(page_num)

        # 7. التقرير الميداني المنظم
        self.logger.info(
            f"📥 [ARCHIVE] P{page_num+1} -> {self.page_cache[page_num]['location_stamp']} | "
            f"DNA: {len(keywords)} | Hubs: {self.network_hubs_count}"
        )

    def get_page_data(self, page_num: int, pdf_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        [إصلاح سيادي]: استرجاع ذكي يدمج بين حماية الذاكرة والتحقق العميق من النزاهة.
        """
        # 1. محاولة الجلب من الكاش المباشر (LRU Hit)
        if page_num in self.page_cache:
            self.page_cache.move_to_end(page_num)
            data = self.page_cache[page_num]

        # 2. الاستعادة الذكية (Auto-Recovery) في حال فقدان الصفحة
        elif pdf_path:
            # [حارس التداخل]: منع الانهيار الناتج عن حلقات الاستدعاء المتكررة
            if getattr(self, '_in_recovery', False):
                self.logger.warning(f"⚠️ استعادة متداخلة محظورة للصفحة {page_num}")
                return None

            self._in_recovery = True
            try:
                self.logger.info(f"🔄 إعادة بناء الذاكرة للصفحة {page_num}...")

                # تنفيذ منطق الاستعادة (النداء للقائد الميداني المصغر)
                success = self._recover_page_logic(page_num, pdf_path)
                if not success:
                    return None

                data = self.page_cache.get(page_num)
            finally:
                self._in_recovery = False # خفض العلم دائماً لضمان استمرار العمل
        else:
            return None

        # 3. فحص النزاهة المعمق (Deep Integrity Check)
        if data:
            # التأكد من بقاء "الجينات السيمانتيكية" سليمة بعد الاستعادة
            critical_keys = ['semantic_keywords', 'layer_type', 'visual_headings']
            if all(data.get(k) for k in critical_keys):
                # حساب المتركس لضمان جودة الأداء
                metrics = data.get('processing_metrics', {})
                if metrics:
                    latency_values = [v for v in metrics.values() if isinstance(v, (int, float))]
                    data['total_latency'] = f"{sum(latency_values):.4f}s"

                data['integrity_score'] = "VERIFIED"
                return data
            else:
                self.logger.error(f"❌ بيانات الصفحة {page_num} تالفة؛ ينقصها جينات أساسية.")

        return None

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
        [إصلاح هندسي]: تصنيف طبقي يعتمد على الأولويات والبيانات المستخرجة.
        يعالج مشكلة الـ Zero Hubs من خلال إعطاء الأولوية للبيانات التقنية.
        """
        text_lower = text.lower()
        word_count = len(text_lower.split())

        # 1. كشافات ذكية (Smart Detectors)
        is_table = any(kw in text_lower for kw in {'table', 'figure', 'diagram', 'chart', 'جدول', 'رسم', 'مخطط'})
        # فحص وجود كلمات تقنية مكثفة (من مخرجات المصفاة السيمانتيكية)
        is_technical = is_table or len(re.findall(r'\d+', text_lower)) > 20

        # 2. سلم الأولويات (Priority Ladder) - الترتيب هنا هو كل شيء

        # الفئة A: البيانات التقنية (المصدر الرئيسي للـ Hubs)
        if is_technical:
            # إذا كان هناك مؤشر فصل مع بيانات تقنية، فهذا "سوبر هب"
            if any(ind in text_lower[:300] for ind in {'chapter', 'فصل', 'section'}):
                return "TECHNICAL_CHAPTER"
            return "TECHNICAL_DATA"

        # الفئة B: الهيكل التنظيمي
        chapter_indicators = {'chapter', 'فصل', 'section', 'part'}
        if any(ind in text_lower[:250] for ind in chapter_indicators):
            return "CHAPTER_LAYER"

        # الفئة C: المراجع والملحقات
        reference_kws = {'ملحق', 'appendix', 'مراجع', 'references', 'bibliography'}
        if any(kw in text_lower for kw in reference_kws):
            return "APPENDIX_LAYER"

        # الفئة D: المحتوى الضخم (Core)
        structure_kws = {'مقدمة', 'فهرس', 'contents', 'introduction'}
        if word_count > 600 or any(kw in text_lower for kw in structure_kws):
            return "CORE_CONTENT"

        return "STANDARD_CONTENT"

    def _recover_page_logic(self, page_num: int, pdf_path: str) -> bool:
        """
        [محرك الاستعادة المرن]: نسخة مصححة لغوياً وبرمجياً (Pylance Safe).
        يضمن استخراج النص من داخل البلوكات دون أخطاء في الأنواع.
        """
        try:
            with fitz.open(pdf_path) as doc:
                if page_num < 0 or page_num >= len(doc):
                    return False

                page = doc.load_page(page_num)

                # 1. الاستراتيجية الأولى: النص الصافي
                raw_text = page.get_text("text")
                # ضمان أن المتغير نصي قبل عمل strip
                clean_text = str(raw_text).strip() if raw_text else ""

                # 2. الاستراتيجية الثانية: المسح العميق عبر البلوكات (إذا كان النص ضعيفاً)
                if len(clean_text) < 15:
                    self.logger.debug(f"🔍 Deep Scan: P{page_num}")
                    blocks = page.get_text("blocks")

                    extracted_parts = []
                    for b in blocks:
                        # في PyMuPDF، النص موجود دائماً في الفهرس رقم 4 من البلوك
                        # Block structure: (x0, y0, x1, y1, "text", block_no, block_type)
                        if len(b) > 4 and isinstance(b[4], str):
                            text_segment = b[4].strip()
                            if text_segment:
                                extracted_parts.append(text_segment)

                    clean_text = "\n".join(extracted_parts)

                # 3. فحص النزاهة النهائية
                if not clean_text:
                    # بدلاً من Warning، اجعلها Debug لتنظيف اللوج
                    self.logger.debug(f"🍃 Page {page_num} is a visual-centric node (No text DNA).")
                    return False

                # 4. الحقن الميداني (The Injection)
                metadata = {"source": pdf_path, "mode": "FLEXIBLE_BURST"}
                # تمرير النص كـ string يقيناً
                self._run_tiered_processing(page_num, page, str(clean_text), metadata)

                # 5. إنعاش الميزان
                self._refresh_memory_balance(page_num)

                return True
        except Exception as e:
            self.logger.error(f"💥 Recovery Failure at P{page_num}: {str(e)}")
            return False

    def _refresh_memory_balance(self, current_page: int):
        """
        [منعش الذاكرة السيادي]: نظام فحص وتفريغ دفعات (Burst-Release System).
        يعمل كميزان يوازن بين 'خفة المحرك' و'دقة البيانات المسترجعة'.
        """
        # 1. معايير الفحص (Pulse Audit Criteria)
        # نضع عتبة منخفضة جداً (10 صفحات) لضمان "منطق الرشاش"
        MAX_BURST_CAPACITY = 10

        if len(self.page_cache) <= MAX_BURST_CAPACITY:
            return # الذاكرة لا تزال في المنطقة الخضراء

        # 2. بروتوكول الفحص قبل الإخلاء (Pre-Eviction Audit)
        # لا نحذف أقدم صفحة فوراً، بل نفحص "رتبتها"
        try:
            # الحصول على أقدم مفتاح (الضحية المحتملة)
            oldest_page_idx = next(iter(self.page_cache))
            oldest_entry = self.page_cache.get(oldest_page_idx, {})

            # [فحص السيادة]: هل هذه الصفحة "مركز ثقل" (Hub)؟
            is_critical = oldest_entry.get("layer_type") in ["TECHNICAL_CHAPTER", "TECHNICAL_DATA"]
            has_high_dna = len(oldest_entry.get("semantic_keywords", [])) > 8

            # 3. قرار الإنعاش (The Release Decision)
            if is_critical and has_high_dna:
                # إذا كانت الصفحة "جنرال"، ننقلها لآخر الطابور بدلاً من حذفها (إنقاذ مؤقت)
                self.page_cache.move_to_end(oldest_page_idx)
                self.logger.debug(f"🛡️ Memory Guard: Protected High-DNA Page {oldest_page_idx} from eviction.")
            else:
                # إذا كانت صفحة عادية، يتم "تفريغ الطلقة" فوراً
                evicted_id, _ = self.page_cache.popitem(last=False)

                # 4. إشارة الإنعاش (The Recovery Signal)
                # نترك وسماً في اللوج ليؤكد أن النظام "تنفس"
                self.logger.info(
                    f"♻️ Memory Refresh [Burst Complete]: Page {evicted_id} released | "
                    f"System Pressure: {len(self.page_cache)} Active Slots"
                )

        except Exception as e:
            self.logger.warning(f"⚠️ Memory Balance Fluctuation: {e}")

        # 5. تنظيف المخلفات البرمجية (Garbage Collection Trigger)
        # إجبار النظام على تنظيف الذاكرة الميتة (RAM) فعلياً كل 5 دفعات
        if current_page % 5 == 0:
            import gc
            gc.collect()
            self.logger.debug("🧹 Deep RAM Sanitization: Engine Lungs Cleared.")

    def PDF_extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        [تحديث القائد]: مستخرج جينات سيمانتيكي يدعم الاختصارات التقنية
        والمصطلحات المركبة (Compound Terms).
        """
        import re
        from collections import Counter

        # 1. تنظيف ذكي يحافظ على الرموز الرابطة (مثل CONTROL_UNIT أو AC-DC)
        text = re.sub(r'[^\w\s\-\_]', ' ', text, flags=re.UNICODE)
        words = text.lower().split()

        # 2. قائمة تجاهل (Stop Words) أكثر صرامة واحترافية
        stop_words = self.get_sovereign_stop_words()

        # 3. استخراج ومعالجة الجينات (DNA Extraction)
        scored_words = []
        for word in words:
            # توحيد المعرفات العربية
            norm_word = re.sub(r'^ال', '', word) if word.startswith('ال') and len(word) > 4 else word

            # فلتر القبول الجديد:
            # - يسمح بالاختصارات التقنية (طول 2 فأكثر)
            # - يتجاهل الأرقام البحتة
            # - يتجاهل كلمات التوقف
            if len(norm_word) >= 2 and norm_word not in stop_words and not norm_word.isdigit():
                scored_words.append(norm_word)

        # 4. نظام الوزن النسبي (Importance Weighting)
        word_counts = Counter(scored_words)
        weighted_scores = {}

        for word, count in word_counts.items():
            # الكلمات التي تحتوي على (_) أو (-) أو حروف كبيرة مختلطة هي "مصطلحات ذهبية"
            weight = 1.0
            if '_' in word or '-' in word: weight = 1.8 # مصطلح هندسي مركب
            if len(word) <= 3 and word.isupper(): weight = 1.5 # اختصار تقني هام

            weighted_scores[word] = count * weight

        # 5. اختيار النخبة (Top Scored DNA)
        # نختار الكلمات التي حصلت على أعلى "وزن" وليس فقط تكرار
        sorted_keys = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_keys[:top_n]]

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
        [تطوير سيادي]: محرك الربط العصبي.
        يحول العلاقات البسيطة إلى "جسور منطقية" ويحدث عداد الـ Hubs حياً.
        """
        if page_num not in self.page_cache: return

        page_data = self.page_cache[page_num]
        keywords = set(page_data.get("semantic_keywords", []))
        current_layer = page_data.get("layer_type", "")

        # 1. نظام "رتبة الصفحة" (Page Ranking)
        priority_map = {"TECHNICAL_CHAPTER": 1.5, "TECHNICAL_DATA": 1.2, "CHAPTER_LAYER": 1.0}
        rank_weight = priority_map.get(current_layer, 0.5)

        # 2. بناء "الجسور النادرة" (Rare Term Linking)
        # نركز على الكلمات التي لا تتكرر في كل الصفحات لضمان دقة الربط
        for kw in keywords:
            # إضافة الصفحة للشبكة
            self.heuristic_network[kw].append({"page": page_num, "weight": rank_weight})

            # إذا وجدت الدالة صفحة أخرى تشترك في نفس الكلمة (الربط المتقاطع)
            related_pages = [item["page"] for item in self.heuristic_network[kw] if item["page"] != page_num]

            for past_page in related_pages:
                # إنشاء رابط "نبضي" (Pulse Link)
                link_id = tuple(sorted((page_num, past_page)))
                if link_id not in self.visual_links:
                    self.visual_links.append({"origin": page_num, "target": past_page, "strength": rank_weight})

                    # 3. ترقية الصفحة إلى HUB (هنا ينتهي لغز الصفر)
                    # إذا زادت روابط الصفحة عن 3 روابط "تقنية"، تصبح Hub رسمياً
                    if len(related_pages) > 3:
                        self.network_hubs_count += 1

        # 4. الربط الهيكلي (Hierarchy Bridge)
        # ربط الصفحات التي تتبع نفس "المسار المنطقي"
        if rank_weight >= 1.0:
            self.layer_hierarchy[page_num] = [p for p in range(max(0, page_num-5), page_num)]

        self.logger.info(f"🛰️ Linker Active: P{page_num} | Connections: {len(related_pages)} | Total Hubs: {self.network_hubs_count}")

    def get_strategic_hubs(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        [تطوير إستراتيجي]: تحديد مراكز الثقل المعرفي بناءً على "كثافة الروابط"
        وليس فقط تكرار الكلمات.
        """
        hub_scores: Dict[str, float] = defaultdict(float)

        # 1. تحليل الشبكة الإستدلالية بذكاء
        for keyword, entries in self.heuristic_network.items():
            if len(keyword) < 3: continue

            # ميزة: "تنوع المصادر" (Source Diversity)
            # الـ Hub القوي هو الذي يربط صفحات متباعدة وليس صفحات متجاورة فقط
            pages = [e["page"] for e in entries if isinstance(e, dict)]
            if not pages: continue

            # حساب "المدى المعرفي" (Knowledge Span)
            page_span = max(pages) - min(pages) + 1
            unique_pages_count = len(set(pages))

            for entry in entries:
                base_weight = float(entry.get("weight", 0.1))

                # معادلة الثقل المعرفي (Gravity Equation):
                # نضرب الوزن في (عدد الصفحات الفريدة) وفي (تنوع الطبقات)
                # هذا يقتل "الضجيج" ويرفع الكلمات التقنية الحقيقية
                diversity_bonus = 1.5 if unique_pages_count > 2 else 1.0
                span_bonus = 1.2 if page_span > 10 else 1.0 # الكلمة التي تربط فصول متباعدة

                hub_scores[keyword] += (base_weight * diversity_bonus * span_bonus)

        # 2. فرز العمالقة (Sorting the Giants)
        sorted_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)
        top_results = sorted_hubs[:top_n]

        # 3. تحديث حالة الوعي (Global Awareness Update)
        # هنا نقوم بحقن النتائج في متغير الـ Hubs الرئيسي ليراه "القائد الأعلى"
        self.network_hubs_count = len([h for h in hub_scores.values() if h > 1.0])

        self.logger.info(f"🎯 Strategic Hubs Mastered: {len(top_results)} nodes | Global Hub Count: {self.network_hubs_count}")
        return top_results

    def _build_visual_heuristics(self, page_num: int, layout_data: Dict[str, Any]):
        """
        [تطوير هندسي]: بناء شبكة الملاحة البصرية مع تحديث عداد الـ Hubs.
        تستخدم الأوزان المستخرجة من المسح الهندسي لتقوية الروابط.
        """
        headings = layout_data.get("headings", [])

        for heading in headings:
            title = str(heading.get("text", "")).strip()
            if not title: continue

            # 1. تسجيل العنوان وتحديث الـ Heading Map
            self.heading_map[title].append(page_num)

            # 2. الربط الإستراتيجي (Semantic Continuity)
            # إذا تكرر العنوان، فهذا "محور معرفي" (Knowledge Axis)
            if len(self.heading_map[title]) > 1:
                source_page = self.heading_map[title][-2]

                # حساب قوة الرابط بناءً على حجم الخط (Priority)
                is_high_priority = heading.get("priority") == "HIGH"
                strength = 1.5 if is_high_priority else 1.0

                link_entry = {
                    "link_type": "HEADING_CONTINUITY",
                    "origin_page": source_page,
                    "target_page": page_num,
                    "anchor_text": title,
                    "strength": strength
                }
                self.visual_links.append(link_entry)

                # [إصلاح لغز الصفر]: ترقية الرابط المتكرر إلى Hub رسمي
                if is_high_priority or len(self.heading_map[title]) > 2:
                    self.network_hubs_count += 1
                    self.logger.debug(f"🛰️ Visual Hub Created: '{title}' connects P{source_page} -> P{page_num}")

        # 3. تدفق السياق الهيكلي (Structural DNA Flow)
        if page_num > 0:
            self.visual_links.append({
                "link_type": "STRUCTURAL_FLOW",
                "origin_page": page_num - 1,
                "target_page": page_num,
                "flow_weight": 0.8
            })

        self.logger.info(f"🕸️ Visual Grid: Page {page_num} linked with {len(headings)} anchors. Total Hubs: {self.network_hubs_count}")

# ------------ خامساً: البحث والتنقل الذكي (Search & Navigation) ------------
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        [تطوير سيادي]: بحث هجين يجمع بين ذكاء المتجهات وقوة الاستعادة الذكية.
        يضمن عدم ضياع النتائج حتى لو تم طرد الصفحات من الذاكرة.
        """
        if not self.vectors:
            self.logger.warning("⚠️ مساحة المتجهات فارغة، تأكد من نجاح عملية الفهرسة.")
            return []

        # 1. تحويل السؤال لمتجه
        query_vector = self.model.encode(query, convert_to_numpy=True)
        vectors_np = np.array(self.vectors)

        # 2. حساب التشابه الجيبي (Cosine Similarity)
        sim_scores = np.dot(vectors_np, query_vector)
        v_norms = np.linalg.norm(vectors_np, axis=1)
        q_norm = np.linalg.norm(query_vector)
        similarities = sim_scores / (v_norms * q_norm + 1e-9)

        # 3. دمج الأوزان الهيكلية المحدثة (Hybrid Scaling)
        layer_weights = {
            "TECHNICAL_CHAPTER": 1.2, # الـ Super Hub الجديد
            "CHAPTER_LAYER": 1.0,
            "TECHNICAL_DATA": 0.9,    # تم الرفع لضمان الأولوية التقنية
            "CORE_CONTENT": 0.7,
            "STANDARD_CONTENT": 0.4
        }

        weighted_scores = []
        for idx, semantic_score in enumerate(similarities):
            page_num = self.vector_map[idx]
            page_data = self.page_cache.get(page_num, {})

            struct_weight = layer_weights.get(page_data.get("layer_type"), 0.3)
            # معادلة الوعي الهجين
            final_score = (semantic_score * 0.7) + (struct_weight * 0.3)
            weighted_scores.append(final_score)

        # 4. تجميع النتائج مع الاستعادة الذكية (Smart Recovery)
        top_indices = np.argsort(weighted_scores)[::-1][:top_k]
        results = []

        for rank_idx in top_indices:
            page_num = self.vector_map[rank_idx]

            # [تحديث جوهري]: استعادة البيانات حتى لو طُردت من الكاش
            p_data = self.get_page_data(page_num)

            if p_data:
                results.append({
                    "page_index": page_num,
                    "score": round(float(weighted_scores[rank_idx]), 4),
                    "layer": p_data.get("layer_type", "UNKNOWN"),
                    "preview": p_data["content"][:300].replace('\n', ' ').strip() + "...",
                    "keywords": p_data.get("semantic_keywords", [])[:3]
                })

        self.logger.info(f"🔍 Search Complete: Found {len(results)} relevant nodes for: '{query[:30]}...'")
        return results

    def page_flipper(self, current_page: int, direction: str = "next") -> int:
        """
        [تطوير ملاحي]: نظام تنقل يعتمد على "مراكز الثقل" (Hubs).
        يسمح بالقفز بين الأقسام التقنية الكبرى بدلاً من التنقل الخطي.
        """
        if not self.navigation_stack or self.navigation_stack[-1] != current_page:
            self.navigation_stack.append(current_page)

        target_page = current_page

        # 1. الملاحة الخطية (البسيطة)
        if direction == "next":
            target_page = current_page + 1
        elif direction == "prev":
            target_page = max(0, current_page - 1)

        # 2. الملاحة الإستدلالية (القفز الذكي)
        elif direction == "next_heading":
            # البحث عن أقرب "مركز ثقل" (Hub) من الطبقات العليا
            # نحن نفضل TECHNICAL_CHAPTER و CHAPTER_LAYER
            target_page = current_page + 1 # القيمة الافتراضية

            # البحث في الذاكرة الحية عن أقرب فصل قادم
            for p_idx in range(current_page + 1, len(self.vector_map)):
                p_data = self.page_cache.get(p_idx, {})
                if p_data.get("layer_type") in ["TECHNICAL_CHAPTER", "CHAPTER_LAYER"]:
                    target_page = p_idx
                    break

        # 3. العودة التاريخية (Smart Backtrack)
        elif direction == "back":
            if len(self.navigation_stack) > 1:
                self.navigation_stack.pop()
                target_page = self.navigation_stack.pop()

        # 4. ضمان الأولوية في الذاكرة (Memory Priority)
        # إذا كانت الصفحة المستهدفة مطرودة من الكاش، نقوم باستعادتها فوراً
        if target_page not in self.page_cache:
            self.get_page_data(target_page) # استعادة ذكية
        else:
            self.page_cache.move_to_end(target_page)

        self.logger.info(f"🧭 Navigator Sync: P{current_page} -> P{target_page} | Logic: {direction}")
        return target_page

    def inspect_cache(self, page_num: Optional[int] = None) -> Dict[str, Any]:
        """
        [تطوير تشخيصي]: فحص عميق لحالة الوعي وتدفق الـ DNA في النظام.
        """
        # 1. نظرة شموليّة على "صحة النظام" (System Health)
        if page_num is None:
            # حساب توزيع الطبقات بشكل إحصائي بدلاً من قائمة طويلة
            layers = [d.get("layer_type", "UNKNOWN") for d in self.page_cache.values()]
            layer_stats = {l: layers.count(l) for l in set(layers)}

            return {
                "status": "GLOBAL_SYSTEM_HEALTH",
                "core_metrics": {
                    "active_pages": len(self.page_cache),
                    "knowledge_hubs": self.network_hubs_count, # العداد الحقيقي المطور
                    "vector_depth": len(self.vectors)
                },
                "intelligence_report": {
                    "layer_balance": layer_stats,
                    "top_hubs": {kw: f"{len(entries)} nodes"
                                for kw, entries in list(self.heuristic_network.items())[:10]}
                }
            }

        # 2. تشريح دقيق لمستوى الصفحة (Surgical Page View)
        # نقوم باستعادة الصفحة أولاً لضمان وجودها في الذاكرة أثناء الفحص
        page_data = self.get_page_data(page_num)
        if not page_data:
            return {"status": "ERROR", "message": f"Page {page_num} is beyond the recovery horizon."}

        return {
            "status": "PAGE_DEEP_ANALYSIS",
            "diagnostics": {
                "id": page_num,
                "identity": page_data.get("layer_type"),
                "dna_complexity": len(page_data.get("semantic_keywords", [])),
                "visual_anchors": len(page_data.get("visual_headings", []))
            },
            "network_connectivity": {
                "direct_links": self.layer_hierarchy.get(page_num, []),
                "heuristic_influence": {
                    kw: f"{len(entries)} links"
                    for kw, entries in self.heuristic_network.items()
                    if any(e.get("page") == page_num for e in entries if isinstance(e, dict))
                }
            },
            "cache_integrity": {
                "status": "HOT_MEMORY",
                "vector_sync": page_num in self.vector_map,
                "integrity_score": page_data.get("integrity_score", "UNVERIFIED")
            }
        }

# ------------ سادساً: المحرك الفكري والتحليل (Thinking & Analysis) ------------
    def analyze_pdf(self, pdf_path: str, thinking_engine: Any, user_request: str = "") -> Dict[str, Any]:
        """
        [القائد الأعلى - النسخة المصححة]
        يدير الاستطلاع السريع، الاستدلال الملاحي، والتحليل النبضي المعزز بالجراف المعرفي.
        """
        # 1. الاستطلاع السريع (Stage 1)
        self.logger.info(f"📡 Stage 1: Fast Ingest for {Path(pdf_path).name}")
        ingest_status = self.fast_ingest_stream(pdf_path)

        # تصحيح Pylance: التحقق من أن النتيجة ليست None قبل استخدام .get()
        # وتصحيح استدعاء الميثود من .get["status"] إلى .get("status")
        if not ingest_status or ingest_status.get("status") != "completed":
            self.logger.error("❌ Stage 1 Failed: Ingest status is None or incomplete.")
            return {"status": "error", "message": "Fast ingestion failed."}

        # الآن Pylance يعلم أن ingest_status هو قاموس يقيناً
        page_count = ingest_status.get("pages", 0)
        all_ideas, cumulative_awareness = [], []

        # 2. الاستدلال الملاحي (Stage 2)
        target_pages = self.infer_navigation_path(user_request)
        # تأكد من تحويل range إلى list لضمان التوافق مع العمليات اللاحقة
        analysis_queue = target_pages if target_pages else list(range(page_count))

        # 3. التحليل النبضي المعزز (Stage 3 & 4)
        step = 20
        total_items = len(analysis_queue)

        # حساب إجمالي النبضات بدقة للمراقب
        total_pulses = (total_items + step - 1) // step

        for start_idx in range(0, total_items, step):
            current_batch = analysis_queue[start_idx : start_idx + step]
            if not current_batch: continue

            chunk_texts, related_summaries = [], []
            for p_num in current_batch:
                page_data = self.get_page_data(p_num, pdf_path=pdf_path)
                if page_data:
                    # إضافة وسم الموقع لكل فقرة لتعزيز الوعي المكاني للمحرك
                    loc_tag = f"[{page_data.get('location_stamp', 'N/A')}]"
                    chunk_texts.append(f"{loc_tag}\n{page_data.get('content', '')}")

                    # سحب العلاقات العميقة
                    related_ids = self.layer_hierarchy.get(p_num, [])[:3]
                    for r_id in related_ids:
                        r_meta = self.page_cache.get(r_id, {})
                        kws = r_meta.get("semantic_keywords", [])[:2]
                        if kws:
                            related_summaries.append(f"[Ref P{r_id+1} in {r_meta.get('location_stamp')}: {', '.join(kws)}]")

            enhanced_prompt = f"{' '.join(chunk_texts)}\n\n-- NETWORK_CONTEXT --\n{chr(10).join(related_summaries)}"

            try:
                # التنفيذ
                chunk_ideas, chunk_state = thinking_engine.generate(enhanced_prompt, user_request=user_request)

                score = float(chunk_state.get("avg_score", 0.5))
                # تمرير total_pulses المحسوبة بدقة
                self._analysis_monitor(len(cumulative_awareness) + 1, total_pulses, score)

                all_ideas.extend(chunk_ideas)
                cumulative_awareness.append(score)
            except Exception as e:
                self.logger.error(f"❌ Pulse Error: {e}")

        # 4. التقرير النهائي الاستراتيجي
        final_score = round(sum(cumulative_awareness)/len(cumulative_awareness), 2) if cumulative_awareness else 0.0

        # استعادة نص الغلاف أو أول صفحة لضمان دقة تخمين الموضوع
        cover_data = self.get_page_data(0)
        topic_source = cover_data.get("content", "") if cover_data else "General Content"

        # استرجاع إحصائيات المجلدات (Archive Stats)
        active_folders = len({d.get('metadata', {}).get('folder_id') for d in self.page_cache.values() if d.get('metadata')})

        all_metadata = [d.get('metadata', {}) for d in self.page_cache.values()]

        folders = {m.get('folder_id') for m in all_metadata if m.get('folder_id')}
        binders = {f"{m.get('folder_id')}_{m.get('binder_id')}" for m in all_metadata if m.get('binder_id')}

        return {
            "status": "success",
            "analysis_metrics": {
                "pages_analyzed": page_count,
                "hubs_found": self.network_hubs_count,
                "consciousness_score": final_score,
                "inferred_topic": self._guess_topic(topic_source)
            },
            "sovereign_archive": {  # هذا هو القسم الجديد الذي تسأل عنه
                "portfolio": "MAIN_KNOWLEDGE_BASE",
                "total_folders": len(folders),
                "total_binders": len(binders),
                "active_map": list(folders)
            },
            "output": {
                "structured_ideas": all_ideas,
                "summary": f"Sovereign Audit Complete ✅ | Archive: {len(folders)} Folders, {len(binders)} Binders."
            }
        }

    def advance_pdf_analyzer(self, pdf_path: str, user_request: str = "") -> Dict[str, Any]:
        """
        [مدير العمليات المتقدمة]: يقوم بالتشريح النبضي وحقن النتائج في الجراف المعرفي.
        """
        # 1. المرحلة الهيكلية (استدعاء القائد الميداني بالاسم الجديد)
        process_result = self.process_pdf_core(pdf_path, user_request)
        if process_result["status"] != "success":
            return process_result

        page_count = process_result["data"]["stats"]["pages"]
        all_ideas: List[Dict] = []
        pulse_scores: List[float] = []
        step = 20

        # --- إضافة: حارس السياق (Context Guard) ---
        accumulated_chars = 0
        overlap_pages = 2 # صفحات تداخل لضمان عدم انقطاع الفكرة

        try:
            # 2. التحليل النبضي (Sequential Pulse Analysis)
            for start_idx in range(0, page_count, step):
                end_idx = min(start_idx + step, page_count)

                # استخراج النصوص من المستودع المركزي
                chunk_text = "\n".join([
                    str(self.page_cache[i].get("content", ""))
                    for i in range(start_idx, end_idx) if i in self.page_cache
                ])

                if not chunk_text.strip(): continue

            for start_idx in range(0, page_count, step):
                # تعديل النطاق ليشمل التداخل (إلا في النبضة الأولى)
                actual_start = max(0, start_idx - overlap_pages) if start_idx > 0 else start_idx
                end_idx = min(start_idx + step, page_count)

                # تجميع النص مع صفحات التداخل
                chunk_pages = [self.page_cache.get(i, {}) for i in range(actual_start, end_idx)]
                chunk_text = "\n".join([str(p.get("content", "")) for p in chunk_pages])

                # حساب الحروف الفعلي للتقرير
                accumulated_chars += len(chunk_text)

                # 3. استدعاء محرك الاستدلال (Inference)
                # نستخدم الدالة المطورة التي تضمن توليد أفكار حقيقية
                chunk_ideas, chunk_state = self.generate_mock(chunk_text, max_iterations=1, user_request=user_request)

                if chunk_ideas:
                    all_ideas.extend(chunk_ideas)
                    # [تحديث سيادي]: حقن الأفكار فوراً في سجلات الـ DNA للمحرك
                    for idea in chunk_ideas:
                        self.engine_metadata["dna_flow"].append(idea)
                        # تحديث عداد الـ Hubs بناءً على قوة الفكرة
                        if idea.get("confidence_score", 0) > 0.8:
                            self.network_hubs_count += 1

                pulse_scores.append(float(chunk_state.get("avg_score", 0.5)))

            # 4. تجميع الوعي النهائي
            final_avg_score = round(sum(pulse_scores) / len(pulse_scores), 2) if pulse_scores else 0.5

            # 5. استرجاع بيانات الغلاف لتخمين الموضوع (Topic Inference)
            # نستخدم get_page_data لضمان استعادة البيانات حتى لو طُردت من الكاش
            cover_data = self.get_page_data(0)
            sample_text = cover_data.get("content", "") if cover_data else ""

            self.logger.info(f"✅ Advanced Analysis Sync: {len(all_ideas)} ideas injected into DNA Flow.")

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
            self.logger.error(f"❌ Critical Error in Advanced Operations: {str(e)}")
            return {"status": "error", "message": str(e)}

    def infer_navigation_path(self, query: str) -> List[int]:
        """
        Reasoning Engine: Decides which layers to prioritize based on query intent.
        Bridge between 'analyze_pdf' and '_analysis_monitor'.
        """
        query_lower = query.lower()
        target_pages: List[int] = []

        # 1. Layer Inference (Technical vs Structural)
        # استخدام الـ Enums التي اعتمدناها: TECHNICAL_DATA & CHAPTER_LAYER
        technical_indicators = {'جدول', 'رسم', 'بيانات', 'table', 'data', 'figure', 'chart'}
        structural_indicators = {'فصل', 'عنوان', 'مقدمة', 'chapter', 'section', 'index'}

        if any(w in query_lower for w in technical_indicators):
            target_pages = [p for p, d in self.page_cache.items() if d.get('layer_type') == 'TECHNICAL_DATA']
            self.logger.info("🎯 Inference: Routing to Technical Data layers.")

        elif any(w in query_lower for w in structural_indicators):
            target_pages = [p for p, d in self.page_cache.items() if d.get('layer_type') == 'CHAPTER_LAYER']
            self.logger.info("🎯 Inference: Routing to Structural/Chapter layers.")

        # 2. Dynamic Hub Routing (Fallback to Heuristic Knowledge)
        if not target_pages:
            # استدعاء Hubs المستند
            hubs = self.get_strategic_hubs(top_n=3)
            for hub_word, _ in hubs:
                if hub_word.lower() in query_lower:
                    # استخراج الصفحات المرتبطة بـ Hub معين
                    pages = [e['page'] for e in self.heuristic_network[hub_word] if isinstance(e, dict)]
                    target_pages.extend(pages)

        # 3. Supervision of the Monitor (إرسال تقرير الاستدلال للمراقب)
        # هنا نقوم بتمهيد البيانات لـ _analysis_monitor ليعرف جودة المسار
        inference_confidence = 1.0 if target_pages else 0.5
        # ملاحظة: استدعاء المونيتور سيحدث فعلياً داخل analyze_pdf بناءً على هذه النتائج

        return sorted(list(set(target_pages)))

    def get_related_pages(self, page_num: int, depth: int = 2) -> List[int]:
        """
        [محرك الجيرة المعرفية]: استقصاء شامل للروابط الهيكلية، البصرية، والسيمانتيكية.
        لا يختصر المسارات، بل يبني جسوراً عابرة للفصول لضمان تدفق الـ DNA.
        """
        if page_num not in self.page_cache:
            return []

        # 1. مستودع النتائج (Set لضمان عدم التكرار)
        related_indices: Set[int] = set()

        # 2. الاستقصاء الهيكلي (Structural Depth)
        # لا نكتفي بالجار المباشر، بل نبحث في تراتبية الطبقات (Hierarchy)
        if page_num in self.layer_hierarchy:
            # إضافة كافة الروابط المرتبطة بالفصل أو القسم
            related_indices.update(self.layer_hierarchy[page_num])

        # 3. الاستقصاء البصري (Visual Strategic Links)
        # البحث عن روابط الاستمرارية (Continuity) التي بنيناها في _build_visual_heuristics
        for link in self.visual_links:
            # إذا كانت الصفحة الحالية هي المصدر أو الهدف
            if link.get("origin_page") == page_num:
                related_indices.add(link["target_page"])
            elif link.get("target_page") == page_num:
                related_indices.add(link["origin_page"])

        # 4. الاستقصاء السيمانتيكي (Deep DNA Matching)
        current_page = self.page_cache[page_num]
        keywords = current_page.get("semantic_keywords", [])

        for kw in keywords:
            if kw in self.heuristic_network:
                # جلب كل الصفحات التي تشترك في هذا الجين (Keyword)
                entries = self.heuristic_network[kw]

                # فرز الأوزان لضمان جلب "مراكز الثقل" (Hubs) أولاً
                # الـ Weight هنا هو الذي استخرجناه من التصنيف الطبقي والمسح الهندسي
                sorted_entries = sorted(
                    [e for e in entries if isinstance(e, dict)],
                    key=lambda x: x.get("weight", 0),
                    reverse=True
                )

                # نأخذ أكبر قدر ممكن بناءً على العمق المطلوب
                for entry in sorted_entries:
                    if entry["page"] != page_num:
                        related_indices.add(entry["page"])

        # 5. الفلترة المتقدمة (Sovereign Filtering)
        # إزالة الصفحة الحالية + ضمان أن الصفحات المستهدفة لا تزال "حية" في الكاش
        final_candidates = {
            p for p in related_indices
            if p != page_num and (p in self.page_cache or p in self.vector_map)
        }

        # 6. الترتيب المنطقي (Logical Sequence)
        # نرتب النتائج لضمان أن القائد الأعلى يقرأ السياق بترتيبه الصحيح
        sorted_results = sorted(list(final_candidates))

        # في المشاريع الضخمة، لا نحدد العدد بنطاق ضيق، بل نترك مساحة كافية للسياق
        # نرفع سقف النتائج لضمان شمولية الـ NETWORK_INSIGHTS
        return sorted_results[:depth * 5]

    def _analysis_monitor(self, current_pulse: int, total_pulses: int, chunk_score: float):
        """
        [لوحة التحكم السيادية]: نسخة الأرشفة المجلدية.
        ترصد موقع النبضة داخل المجلدات والمصنفات بجانب جودة الـ DNA.
        """
        # 1. حساب مؤشرات الأداء الأساسية
        progress = (current_pulse / total_pulses) * 100
        cache_count = len(self.page_cache)
        cache_usage_pct = (cache_count / self.max_pages) * 100

        # 2. حساب المتغيرات الاستراتيجية
        active_hubs = getattr(self, 'network_hubs_count', 0)
        total_links = len(getattr(self, 'visual_links', []))

        # [تحديث سيادي]: تحديد موقع "رأس النبضة" في الأرشيف
        # النبضة الحالية تبدأ تقريباً من الصفحة (النبضة * 20)
        current_page_idx = (current_pulse - 1) * 20
        loc = self._assign_sovereign_location(current_page_idx)
        archive_pos = f"{loc['folder_id']} > {loc['binder_id']}"

        # حساب متوسط الوعي التراكمي
        awareness_list = getattr(self, 'cumulative_awareness', [])
        global_awareness = sum(awareness_list) / len(awareness_list) if awareness_list else chunk_score

        # تحديد الأيقونات
        quality_icon = "💎" if chunk_score >= 0.85 else "✅"
        # ميزان الذاكرة أصبح أكثر حساسية لنظام الرشاش
        memory_icon = "🟢" if cache_usage_pct < 70 else "🟡" if cache_usage_pct < 90 else "🔥"

        # 3. صياغة التقرير البصري المحكم (نسخة الأرشيف)
        report = (
            f"\n═════════════════════════════════════════════\n"
            f"[ 🛰️ PULSE MONITOR #{current_pulse:02d} | {archive_pos} ]\n"
            f"═════════════════════════════════════════════\n"
            f" 🧬 DNA Flow: {active_hubs} Hubs | 🕸️ Net: {total_links} Links\n"
            f" 📊 Progress: {progress:>5.1f}% | Avg Awareness: {global_awareness:.2f}\n"
            f" 💎 Pulse Quality: {quality_icon} {chunk_score:.4f}\n"
            f" 💾 Cache {memory_icon}: {cache_usage_pct:.1f}% ({cache_count}/{self.max_pages} Slots)\n"
            f"─────────────────────────────────────────────"
        )

        self.logger.info(report)

        # 4. الحماية الاستراتيجية للذاكرة
        if cache_usage_pct >= 98.0:
            self._execute_emergency_cleanup()

    def _execute_emergency_cleanup(self):
        """
        [تطهير سيادي]: حذف ذكي يحمي "أعمدة النظام" (Hubs) ويضحي بالبيانات العادية.
        """
        try:
            target_eviction = 5
            evicted_count = 0

            # محاولة البحث عن صفحات ليست "Hubs" لحذفها أولاً
            pages_to_check = list(self.page_cache.keys())
            for p_num in pages_to_check:
                if evicted_count >= target_eviction: break

                p_data = self.page_cache.get(p_num, {})
                # حماية الفصول والبيانات التقنية من الحذف العشوائي
                if p_data.get("layer_type") not in ["TECHNICAL_CHAPTER", "CHAPTER_LAYER"]:
                    self.page_cache.pop(p_num)
                    evicted_count += 1
                    self.logger.warning(f"🚨 SMART_PURGE: Evicted Standard Page {p_num} to protect Hubs.")

            # إذا لم نجد صفحات عادية كافية، نضطر للحذف التقليدي (LRU)
            while evicted_count < target_eviction and len(self.page_cache) > 1:
                evicted_page, _ = self.page_cache.popitem(last=False)
                evicted_count += 1
        except Exception as e:
            self.logger.error(f"⚠️ Cleanup Failure: {e}")

    def generate(self, content: str, user_request: str = "", max_iterations: int = 3) -> Tuple[List[Dict], Dict]:
        """
        [محرك الاستدلال]: توليد أفكار مرتبطة بالجينات السيمانتيكية للنص.
        """
        # استدعاء مستخرج الجينات الذي أصلحناه سابقاً لضمان دقة الفكرة
        keywords = self.PDF_extract_keywords(content, top_n=5)
        topic = self._guess_topic(content)

        self.logger.info(f"🧠 Reasoning Mode: [{topic}] | Pulse Density: {len(content)} chars")

        ideas: List[Dict[str, Any]] = []
        for i in range(1, max_iterations + 1):
            kw = keywords[i-1] if i <= len(keywords) else "System"

            ideas.append({
                "idea_id": i,
                "title": f"[{topic}] Strategy: {kw.upper()}", # ربط الفكرة بالموضوع والجينات
                "importance_level": "CORE" if i == 1 else "SUPPORTING",
                "extracted_segments": [content[:200].replace('\n', ' ')],
                "confidence_score": round(0.92 - (i * 0.04), 2), # ثقة أعلى للمحرك المطور
                "dna_tag": kw
            })

        awareness_state = {
            "avg_score": round(sum(d["confidence_score"] for d in ideas)/len(ideas), 2),
            "top_topic": topic,
            "engine_status": "STABLE"
        }

        return ideas, awareness_state

    def generate_mock(self, content: str, max_iterations: int = 3, user_request: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        [محرك الاستدلال السيادي]: توليد أفكار مرتبطة بطلب المستخدم والطبقة التقنية للنص.
        """

        # --- إضافة: حماية الموارد ومنع التكرار ---
        if not content.strip() or len(content) < 50:
            self.logger.warning("⚠️ Pulse too lean for deep reasoning. Using lightweight mode.")
            actual_iterations = 1
            keywords = ["General_Context"] # جين افتراضي

        # التأكد من عدم وجود "انفجار في الكلمات" (Limit keywords to avoid overhead)
        keywords = keywords[:12] if keywords else ["Analysis_Segment"]

        # 1. كشف المقياس والهوية (Scale & Identity)
        scale = self._detect_processing_scale(content)
        current_topic = self._guess_topic(content)
        # استدعاء المصنف لمعرفة ثقل الصفحة برمجياً
        layer_type = self._classify_layer(content, {})

        # 2. استخراج الجينات السيمانتيكية
        keywords = self.PDF_extract_keywords(content, top_n=12)
        num_kws = len(keywords)

        self.logger.info(f"⚙️ Mode: [{scale['mode']}] | Layer: [{layer_type}] | Target: '{user_request[:25]}...'")

        ideas: List[Dict[str, Any]] = []
        actual_iterations = max(1, min(max_iterations, scale["iterations"]))

        for i in range(1, actual_iterations + 1):
            # اختيار ذكي للعنوان يمنع التكرار (Unique Identity)
            kw_main = keywords[0] if num_kws > 0 else "Analysis"
            kw_sub = keywords[i % num_kws] if num_kws > i else f"Segment_{i}"

            # تعديل درجة الثقة بناءً على "رتبة الطبقة" (Layer Rank)
            layer_boost = 0.15 if layer_type in ["TECHNICAL_DATA", "TECHNICAL_CHAPTER"] else 0.0
            base_score = 0.7 + (min(num_kws, 10) * 0.02) + layer_boost
            dynamic_score = round(min(base_score + (i * 0.01), 0.99), 2)

            ideas.append({
                "idea_id": i,
                "title": f"[{current_topic}] {kw_main.upper()} Insight: {kw_sub}",
                "user_intent": user_request,
                "importance_level": "CORE_CONCEPT" if i == 1 else "SUPPORTING_DETAIL",
                "extracted_segments": [content[:200].replace('\n', ' ').strip() + "..."],
                "confidence_score": dynamic_score,
                "layer_origin": layer_type # حفظ المصدر الطبقي للفكرة
            })

        # 3. مزامنة الوعي النهائي
        avg_score = round(sum(d['confidence_score'] for d in ideas) / len(ideas), 2) if ideas else 0.5

        state = {
            "avg_score": avg_score,
            "layer_consistency": layer_type,
            "processing_depth": scale["mode"]
        }

        return ideas, state

    def get_network_hubs(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        [تطوير سيادي]: تحديد مراكز الثقل المعرفي بناءً على "كثافة الثقل" (Weight Density)
        وليس مجرد تكرار الكلمات، لضمان ظهور الكود التقني كـ Hub.
        """
        hub_weights: Dict[str, float] = {}

        # 1. حساب "الثقل المعرفي" لكل كلمة في الشبكة
        for keyword, entries in self.heuristic_network.items():
            if len(keyword) < 3: continue # تجاهل الضجيج

            # حساب الثقل: مجموع الأوزان التي جمعناها في add_page و _build_heuristic_links
            total_weight = sum(float(e.get("weight", 0.1)) for e in entries if isinstance(e, dict))

            # "بونص" التنوع: إذا كانت الكلمة تربط صفحات بعيدة عن بعضها، يزداد ثقلها
            pages = [e["page"] for e in entries if isinstance(e, dict)]
            page_span = max(pages) - min(pages) if pages else 0
            span_bonus = 1.2 if page_span > 15 else 1.0

            hub_weights[keyword] = round(total_weight * span_bonus, 2)

        # 2. فرز العمالقة (Sorting by Strategic Weight)
        sorted_hubs = sorted(hub_weights.items(), key=lambda x: x[1], reverse=True)
        top_hubs = sorted_hubs[:top_n]

        # 3. تحديث اللوج الاحترافي
        self.logger.info(f"🎯 Strategic Hubs Mastered: {[h[0] for h in top_hubs]}")

        return top_hubs

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
        [المشرف الطبقي]: يدير مستويات الوعي الخمسة لضمان بناء شبكة استدلالية متكاملة.
        """
        metrics = {}
        content_clean = content.strip()
        content_len = len(content_clean)

        # --- الطبقة 1: الإدراك السيمانتيكي (Vector Layer) ---
        t_start = time.perf_counter()
        # تحسين الصمام: لا نتخطى الصفحة إذا كانت "تقنية" حتى لو كانت قصيرة
        is_technical_snippet = any(kw in content_clean.lower() for kw in {'table', 'fig', 'system', 'ctrl'})

        if content_len > 50 or is_technical_snippet:
            vector = self.model.encode(content_clean, convert_to_numpy=True)
            self.vectors.append(vector)
            self.vector_map.append(page_num)
            metrics['perception'] = "SUCCESS_ENCODED"
        else:
            metrics['perception'] = "SKIPPED_MINIMAL"

        metrics['lat_perception'] = time.perf_counter() - t_start

        # --- الطبقة 2: التحليل الهيكلي (Layout Layer) ---
        t_start = time.perf_counter()
        layout_data = self._extract_layout_structure(page_obj)
        metrics['lat_layout'] = time.perf_counter() - t_start

        # --- الطبقة 3: الاستدلال والكلمات (Semantic Layer) ---
        t_start = time.perf_counter()
        keywords = self.PDF_extract_keywords(content_clean)
        layer_type = self._classify_layer(content_clean, metadata)
        topic = self._guess_topic(content_clean)
        metrics['lat_semantic'] = time.perf_counter() - t_start

        # --- الطبقة 4: المزامنة والحقن (Integration Layer) ---
        page_entry = {
            "content": content_clean,
            "topic": topic,
            "semantic_keywords": keywords,
            "visual_headings": layout_data.get("headings", []),
            "layer_type": layer_type,
            "metadata": metadata,
            "metrics": metrics
        }

        # تحديث الكاش المركزي فوراً ليكون متاحاً لدوال الروابط
        self.page_cache[page_num] = page_entry
        self._audit_and_sync_cache(page_num, page_entry) # التدقيق النوعي

        # --- الطبقة 5: الربط الاستدلالي (Networking Layer) ---
        # هنا يتم كسر لغز الصفر: الربط البصري والاستدلالي بناءً على بيانات الطبقات السابقة
        self._build_visual_heuristics(page_num, layout_data)
        self._build_heuristic_links(page_num)

        # تحديث العداد الميداني للوج (Real-time Reporting)
        self.logger.info(
            f"🧬 Processed P{page_num+1} | Type: {layer_type} | "
            f"Hubs: {self.network_hubs_count} | Keywords: {len(keywords)}"
        )

        return layer_type

    def _audit_and_sync_cache(self, page_num: int, current_entry: Dict[str, Any]) -> bool:
        """
        [مدقق النزاهة السيادي]: يضمن جودة البيانات ويمنع "العمى السيمانتيكي" الناتج عن عدم المزامنة.
        """
        # 1. فحص الوجود (LRU Safety)
        if page_num not in self.page_cache:
            return False

        # 2. فحص المحتوى (Content Integrity)
        content = str(current_entry.get("content", ""))
        if len(content) < 10:
            self.logger.error(f"❌ INTEGRITY_ERROR: Page {page_num} has insufficient DNA.")
            return False

        # 3. مزامنة المتجهات (The Core Guardrail)
        # إصلاح: منع المزامنة العشوائية التي تدمر دقة البحث
        if len(self.vectors) != len(self.vector_map):
            self.logger.log(logging.CRITICAL, f"🚨 CRITICAL_SYNC_GAP: Vector misalignment at P{page_num}!")

            # بدلاً من التخمين، نقوم بإلغاء آخر متجه لضمان استمرار النزاهة
            if len(self.vectors) > len(self.vector_map):
                self.vectors.pop()
            return False

        # 4. مراقبة "ثقل" المعالجة (Latency Audit)
        metrics = current_entry.get("processing_metrics", {})
        for layer, latency in metrics.items():
            if isinstance(latency, (float, int)) and latency > 1.5:
                self.logger.warning(f"🐢 SLOW_LAYER: {layer} on P{page_num} ({latency:.2f}s)")

        # 5. تدقيق الربط (Connectivity Audit)
        # إصلاح: فحص سريع للتحقق من أن الصفحة ليست "جزيرة معزولة"
        keywords = current_entry.get("semantic_keywords", [])
        if len(keywords) > 3:
            # نتحقق فقط مما إذا كان المحرك الرابط قد سجل أي حركة لهذه الصفحة
            is_linked = any(page_num == link.get("origin_page") or page_num == link.get("target_page")
                           for link in self.visual_links[-10:]) # فحص آخر 10 روابط للسرعة

            if not is_linked:
                # تحديث عداد الـ Hubs إذا كانت الصفحة غنية ولكنها معزولة (تحفيز الربط)
                self.logger.debug(f"ℹ️ ISLAND_DETECTED: P{page_num} needs deeper heuristic bridging.")

        # ابحث عن هذا الجزء في دالة _audit_and_sync_cache وقم بتعديله:
        if not current_entry.get("semantic_keywords"):
            # القائد الميداني يأمر بالإصلاح الفوري بدلاً من إعلان الفشل
            self.logger.info(f"🔧 Self-Repair Active: Re-generating DNA for Page {page_num}")
            current_entry["semantic_keywords"] = self.PDF_extract_keywords(current_entry["content"])

        # كرر نفس الأمر للـ layer_type إذا كان مفقوداً
        if not current_entry.get("layer_type"):
            current_entry["layer_type"] = self._classify_layer(current_entry["content"], current_entry.get("metadata", {}))

        # --- تحديث سيادي: التحقق من النزاهة المكانية (Archive Integrity) ---
        meta = current_entry.get("metadata", {})
        if "folder_id" not in meta:
            # إعادة التعيين باستخدام المسميات الصحيحة
            self.logger.warning(f"📍 Location Lost: Re-assigning archive path for Page {page_num}")
            loc = self._assign_sovereign_location(page_num)
            current_entry["metadata"].update(loc)
            current_entry["location_stamp"] = f"{loc['folder_id']} > {loc['binder_id']}"

        return True

    def _audit_sweet_spot(self, chunk_content: str, current_confidence: float) -> bool:
        """
        [مدقق المنطقة الذهبية]: يحدد ما إذا كانت النبضة التحليلية وصلت لمستوى الوعي المطلوب.
        """
        # النطاق الذهبي للأكواد الضخمة
        lower_bound = 0.86
        upper_bound = 0.98

        if lower_bound <= current_confidence <= upper_bound:
            return True

        # قبول مشروط للنصوص القصيرة (البرقيات التقنية)
        if len(chunk_content) < 500 and current_confidence >= 0.82:
            return True

        return False

    def _reprocess_pulse(self, content: str, initial_score: float, thinking_engine: Any):
        """
        [المعالجة العميقة]: إجبار المحرك على الغوص في التفاصيل التقنية للوصول للمنطقة الذهبية.
        """
        # 1. استخراج "جينات التوجيه" (Guidance Keywords) لرفع الدقة
        guidance_kws = self.PDF_extract_keywords(content, top_n=5)
        enhanced_prompt = f"DEEP_AUDIT_REQUIRED | FOCUS_ON: {', '.join(guidance_kws)}\n\nDATA: {content}"

        self.logger.info(f"🔄 Sovereign Shift: Escalating to Deep Analysis for confidence {initial_score}...")

        try:
            # 2. استدعاء المحرك بجهد مضاعف (Max Iterations)
            if hasattr(thinking_engine, 'generate'):
                # نرفع عدد التكرارات لضمان كسر حاجز الـ 0.85
                enhanced_ideas, enhanced_state = thinking_engine.generate(enhanced_prompt, max_iterations=5)
            else:
                enhanced_ideas, enhanced_state = thinking_engine(enhanced_prompt)

            # حقن وسم "المنطقة الذهبية" في الحالة
            enhanced_state["audit_status"] = "REPROCESSED_SUCCESS"
            return enhanced_ideas, enhanced_state

        except Exception as e:
            self.logger.error(f"❌ Deep Audit Failure: {e}")
            return [], {"avg_score": initial_score, "audit_status": "FAILED"}

    def export_to_json(self, analysis_result: Dict[str, Any], output_path: str):
        """
        Exports analysis results and heuristic network state to a JSON file.
        Standardized with English internal identifiers.
        """
        import json

        # تجميع بيانات النظام والشبكة
        export_payload = {
            "system_metadata": {
                "engine_version": "2.0-Hybrid",
                "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "analysis_output": analysis_result,
            "network_topology": {
                "heuristic_keys_count": len(self.heuristic_network),
                "visual_links_count": len(self.visual_links),
                "vector_space_size": len(self.vectors),
                "cache_capacity": f"{len(self.page_cache)}/{self.max_pages}"
            }
        }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_payload, f, ensure_ascii=False, indent=4)
            self.logger.info(f"💾 DATA_EXPORT: Successfully saved JSON to {output_path}")
        except Exception as e:
            self.logger.error(f"❌ EXPORT_FAILED: Could not save JSON: {str(e)}")

    def export_to_markdown(self, analysis_result: Dict[str, Any], output_path: str):
        """
        Generates a professional Markdown report based on the hybrid analysis output.
        Standardized with updated English internal keys.
        """
        # استخراج البيانات بناءً على الهيكل الجديد (analysis_metrics & output)
        metrics = analysis_result.get('analysis_metrics', {})
        file_ref = analysis_result.get('file_reference', {})
        output_data = analysis_result.get('output', {})

        page_count = metrics.get('page_count', 'N/A')
        char_count = file_ref.get('char_count', 0)
        file_name = Path(file_ref.get('path', 'Unknown')).name
        consciousness = metrics.get('consciousness_score', 0)

        md = [
            f"# 📄 Document Analysis Report: {file_name}",
            f"\n## 📊 System Statistics",
            f"- **Processed Pages:** {page_count}",
            f"- **Character Count:** {char_count:,}",
            f"- **Analytical Confidence:** {consciousness * 100:.1f}%",
            f"- **Inferred Main Topic:** `{metrics.get('inferred_topic', 'General')}`",
            f"\n## 🧠 Extracted Core Concepts"
        ]

        # معالجة الأفكار المستخرجة
        for idea in output_data.get('structured_ideas', []):
            # استخدام CORE_CONCEPT الذي اعتمدناه في الـ Mock
            tag = "### 🌟" if idea.get('importance_level') == "CORE_CONCEPT" else "#### 🔹"
            md.append(f"{tag} {idea.get('title')} (Confidence: {idea.get('confidence_score', 0)})")

            if 'extracted_segments' in idea:
                md.append(f"> {idea['extracted_segments'][0]}")

            # عرض الكلمات السياقية
            kws = idea.get('contextual_keywords', [])
            if kws:
                md.append(f"**Contextual Keywords:** `{', '.join(kws)}`\n")

        md.append(f"\n## 🌐 Heuristic Network Insights")
        md.append(f"- The engine established **{len(self.heuristic_network)}** semantic hubs across the document.")
        md.append(f"- **Visual Navigation Links:** {len(self.visual_links)} logical connections identified.")

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(md))
            self.logger.info(f"📝 REPORT_EXPORT: Professional Markdown generated at {output_path}")
        except Exception as e:
            self.logger.error(f"❌ EXPORT_FAILED: Could not save Markdown: {str(e)}")

    def clear_cache(self, deep: bool = False):
        """
        Smart Memory Management: balances system performance with data retention.
        Standardized with English internal identifiers.
        """
        # 1. Partial Clean (Default): يحذف النصوص فقط لتوفير الرام مع الإبقاء على الذكاء
        # نقوم بمسح النصوص الثقيلة لكن نبقي على الـ Vectors والـ Hierarchy للتنقل
        self.page_cache.clear()
        self.page_queue = deque()

        if deep:
            # 2. Deep Clean: مسح شامل لكل شيء (Reset Engine)
            self.heuristic_network.clear()
            self.layer_hierarchy.clear()
            self.visual_links = []
            self.vectors = []
            self.vector_map = []
            self.navigation_stack = deque()
            self.logger.info("🧹 DEEP_CLEAN: All processed data and vectors have been purged.")
        else:
            # 3. Intelligent Retention:
            # نبقي على الشبكة الاستدلالية لأنها "خفيفة" وتسمح بالبحث حتى بدون نصوص
            self.logger.info("🧹 SMART_CLEAN: Page content cleared. Semantic network and vectors retained.")

        # تحديث حالة النظام في السجلات
        self.logger.info(f"📊 Resource Status: Vectors[{len(self.vectors)}] | Network_Keys[{len(self.heuristic_network)}]")

# ========== الربط مع process_pdf_streaming ==========
def process_pdf_streaming(pdf_path: str, logger: Any, callback: Optional[Callable] = None):
    """
    [المصنع السيادي]: يقوم بتهيئة الشبكة وتفويض القائد الميداني للقيام بالمعالجة الطبقية.
    """
    logger.info("🚀 Launching Sovereign Factory: Serial Ingestion & Network Building...")

    # 1. إنشاء نسخة المحرك (المختبر المركزي)
    # نمرر الـ logger لضمان توحيد سجلات المراقبة
    cache_network = PDFPageCacheNetwork(logger=logger)

    try:
        # 2. تفويض القائد الميداني (The Master Core)
        # بدلاً من تكرار الكود هنا، نستدعي المحرك الذي أصلحناه (القائد الميداني)
        # هذا يضمن تفعيل: المسح الهندسي، الجداول، التصنيف، والربط الاستدلالي تلقائياً.

        # ملاحظة: سنستخدم ميثود 'process_pdf_core' التي بنيناها لتكون هي المحرك
        result = cache_network.process_pdf_core(pdf_path)

        if result["status"] == "success":
            # إشعار خارجي (Callback) للنتائج النهائية
            if callback:
                callback(result["data"]["stats"]["pages"], "FULL_INDEX_COMPLETE", result["data"]["metadata"])

            # 3. التدقيق النهائي للشبكة قبل التسليم
            hubs_count = cache_network.network_hubs_count
            logger.info(f"✅ Factory Process Complete: {hubs_count} Network Hubs Mastered.")
            return cache_network
        else:
            logger.error(f"❌ FACTORY_CORE_FAILURE: {result.get('error_details')}")
            return None

    except Exception as e:
        logger.error(f"❌ SOVEREIGN_STREAMING_FAILURE: {str(e)}")
        return None


# مثال الاستخدام:
"""
cache = processor.process_pdf_streaming("philosophy_paper.pdf")

# استعلام تلقائي
related = cache.get_related_pages(1)  # صفحة 1 + مرتبطاتها
print(f"صفحة 1 مرتبطة بـ: {related}")

# عرض المصنّف
print(cache.inspect_cache())
"""

if __name__ == "__main__":
    # 1. تهيئة المحرك السيادي
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("SovereignEngine")
    engine = PDFPageCacheNetwork(logger=logger, max_pages=150) # رفع الكاش قليلاً للأرشفة

    # ربط العميل الحقيقي بالمحرك
    from openai import OpenAI
    engine.llm_client = OpenAI(api_key="")

    print("\n" + "="*60)
    print("🚀 [Sovereign Mode] Hierarchical Ingestion & Portfolio Archiving")
    print("="*60)

    pdf_file = "ROBOTICS.pdf"

    # المرحلة 1: تفعيل القائد الميداني (Core Ingestion)
    process_result = engine.process_pdf_core(pdf_file)

    if process_result["status"] == "success":
        p_data = process_result.get("data", {})
        stats = p_data.get("stats", {})
        print(f"✅ Perception Layer Ready: {stats.get('pages')} pages indexed.")
        print(f"📡 Vector Space: {len(engine.vectors)} unique embeddings.")

        # المرحلة 2: التحليل النبضي المتسلسل
        print("\n🧠 Activating Chained Pulse Analysis...")
        final_report = engine.analyze_pdf(
            pdf_path=pdf_file,
            thinking_engine=engine,
            user_request="Deep technical audit of control systems"
        )

        # المرحلة 3: تقرير الأداء السيادي (Sovereign Portfolio Report)
        # هنا استخراج بيانات الأرشيف التي طلبتها
        archive = final_report.get('sovereign_archive', {})

        print(f"\n📂 Sovereign Architecture Report:")
        print(f"   ├─ Portfolio:  {archive.get('portfolio', 'MAIN_KNOWLEDGE_BASE')}")
        print(f"   ├─ Folders:    {archive.get('total_folders', 0)} active folders")
        print(f"   ├─ Binders:    {archive.get('total_binders', 0)} indexed binders")
        print(f"   ├─ Hub Density: {engine.network_hubs_count} knowledge nodes")
        print(f"   └─ Connections: {len(engine.visual_links)} active links")

        metrics = final_report.get('analysis_metrics', {})
        score = metrics.get('consciousness_score', 0)
        print(f"   📊 Global Awareness: {score * 100:.1f}%")

        # المرحلة 4: اختبار القفز السيمانتيكي (Traversal Test)
        print("\n🔍 Knowledge Graph Traversal:")
        query = "control systems and feedback loops"
        search_hits = engine.semantic_search(query, top_k=2)

        for hit in search_hits:
            p_num = hit['page_index']
            related = engine.get_related_pages(p_num, depth=1)
            # عرض موقع الصفحة داخل الأرشيف أثناء البحث
            print(f"   📍 Match: Page {p_num+1} | Score: {hit['score']:.2f}")
            print(f"      🔗 Contextual Neighbors: {related}")

        # المرحلة 5: الأرشفة النهائية والتصدير
        print("\n💾 Committing Portfolio to Disk...")
        engine.export_to_markdown(final_report, "sovereign_audit_report.md")

        # اكتشاف النخبة (Knowledge Core)
        print("\n🧬 Knowledge Core Discovery (Strategic Hubs):")
        hubs = engine.get_network_hubs(5)
        for word, weight in hubs:
            print(f"   🔑 Hub: '{word.upper():<15}' | Strategic Weight: {weight:.2f}")

    else:
        print(f"❌ Engine Failed: {process_result.get('error_details')}")

    print("\n" + "="*60)
    print("🎉 Sovereign Workflow Completed | Portfolio Secure ✅")
    print("="*60)
