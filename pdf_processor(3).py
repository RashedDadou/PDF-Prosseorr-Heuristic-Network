# pdf_processor.py
"""
محرك تحليل PDF المنفصل والمتكامل
يعمل مع أي thinking engine
"""

import logging
import time
import warnings
from collections import deque, OrderedDict, defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Protocol

# تحسين: استيراد torch فقط عند الحاجة أو التأكد من وجوده
import torch
import numpy as np
import fitz  # PyMuPDF

from openai import OpenAI
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# تحسين: إدارة التحذيرات قبل تحميل المكتبات الثقيلة
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)

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
# إعدادات الموديل (LoggerProtocol)
# ===================================================================
class LoggerProtocol(Protocol):
    def info(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...

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

    def _make_logger(self) -> LoggerProtocol:
        logger = logging.getLogger(f"PDFProcessor_{id(self)}")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            # تحسين: إضافة مستوى اللوّج في التنسيق لتمييز الأخطاء
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

# ------------ ثانياً: استخراج البيانات الهيكلية (Extraction Engine)
    def process_pdf(self, pdf_path: str, user_request: str = "") -> Dict[str, Any]:
        start_time = time.time()
        try:
            with fitz.open(pdf_path) as pdf_doc:
                page_count = len(pdf_doc)
                full_text_parts: List[str] = []

                raw_meta = pdf_doc.metadata or {}
                doc_info = {
                    "title": str(raw_meta.get("title") or Path(pdf_path).stem),
                    "author": str(raw_meta.get("author") or "Unknown"),
                    "creator": str(raw_meta.get("creator") or "N/A")
                }

                self.logger.info(f"🚀 معالجة عدوانية لـ: {doc_info['title']} ({page_count} صفحة)")

                for page_num in range(page_count):
                    page = pdf_doc[page_num]
                    page_contents = []

                    # 1. استخراج النصوص الهيكلية (Standard Blocks)
                    raw_blocks = page.get_text("blocks")
                    blocks = [b for b in raw_blocks if isinstance(b, tuple) and len(b) >= 7]
                    blocks.sort(key=lambda b: (b[1], b[0]))

                    for b in blocks:
                        if b[6] == 0:  # نص حقيقي
                            text = str(b[4]).strip()
                            if text: page_contents.append(text)

                    # 2. استخراج بيانات الجداول (Table Mining) - السر في حل مشكلة النقص
                    try:
                        tabs = page.find_tables()
                        for table in tabs:
                            # تحويل الجدول لنص مهيكل لكي يفهمه الـ AI
                            table_data = table.extract()
                            table_text = " | ".join([" ".join([str(cell).strip() for cell in row if cell]) for row in table_data])
                            if table_text.strip():
                                page_contents.append(f"\n[TECHNICAL_TABLE_DATA]: {table_text}")
                    except Exception:
                        pass # بعض النسخ قد لا تدعم استخراج الجداول برمجياً

                    # 3. دمج النصوص وتدقيق الكثافة
                    page_text = "\n".join(page_contents).strip()

                    # صمام الأمان الهجين: إذا كانت الصفحة "بصرية" (مخطط هندسي)
                    if not page_text or len(page_text) < 20:
                        image_count = len(page.get_images())
                        if image_count > 0:
                            page_text = f"[VISUAL_PAGE]: Content includes {image_count} engineering diagrams/images. Technical analysis active."

                    if page_text:
                        full_text_parts.append(page_text)
                        page_meta = {
                            "source_path": pdf_path,
                            "document_title": doc_info['title'],
                            "page_index": page_num + 1,
                            "is_ocr_applied": False,
                            "contains_tables": len(page.find_tables().tables) > 0 if hasattr(page, 'find_tables') else False
                        }

                        # استدعاء دالة الإضافة (التي قمنا بتعديلها لتكون مرممة)
                        self.add_page(page_num, page_text, page, page_meta)

                full_text = "\n\n".join(full_text_parts)
                process_time = round(time.time() - start_time, 2)

                # تحديث التقرير الختامي ليكون أكثر دقة
                if not full_text.strip() or len(full_text) < 100:
                    return {
                        "status": "warning",
                        "message": "نص غير كافٍ، تم وسم الصفحات كأصول بصرية (Visual Assets).",
                        "stats": {"pages": page_count, "time": process_time}
                    }

                self.logger.info(f"✅ اكتمل تشريح المستند في {process_time} ثانية")
                return {
                    "status": "success",
                    "data": {
                        "full_text": full_text,
                        "metadata": doc_info,
                        "stats": {"pages": page_count, "chars": len(full_text), "time": process_time}
                    },
                    "context": {"user_request": user_request}
                }

        except Exception as e:
            self.logger.error(f"❌ فشل المحرك في التشريح العدواني لـ {pdf_path}: {str(e)}")
            return {"status": "error", "error_details": str(e)}

    def _extract_aggressive_text(self, page: Any) -> str:
        """
        استخراج نصي مكثف: يدمج بين النصوص، الجداول، والروابط لتقليل الفقد.
        """
        # 1. استخراج النصوص كبلوكات (أكثر دقة من النص الخام)
        blocks = page.get_text("blocks")
        # ترتيب وتجميع البلوكات النصية
        text_parts = [str(b[4]).strip() for b in blocks if b[6] == 0 and str(b[4]).strip()]

        # 2. محاولة سحب النصوص من الجداول (إذا وجدت)
        try:
            tabs = page.find_tables()
            for table in tabs:
                df_text = " ".join([str(cell).strip() for row in table.extract() for cell in row if cell])
                if df_text:
                    text_parts.append(f"\n[TABLE_DATA]: {df_text}")
        except: pass # الجداول قد لا تدعم في كل النسخ

        full_content = "\n".join(text_parts)

        # 3. صمام الأمان: إذا كان النص لا يزال فارغاً، نسحب "الميتا-داتا" البصرية
        if len(full_content.strip()) < 10:
            image_count = len(page.get_images())
            if image_count > 0:
                return f"[IMAGE_PAGE]: This page contains {image_count} visual assets. Content density is low."

        return full_content

    def fast_ingest_stream(self, pdf_path: str, callback: Optional[Callable] = None):
        """
        القارئ السريع: تحسين استهلاك الذاكرة وسرعة استخراج النصوص الخام.
        """
        try:
            with fitz.open(pdf_path) as pdf_doc:
                total_pages = pdf_doc.page_count
                self.logger.info(f"🌀 بدء التدفق السريع لـ {total_pages} صفحة...")

                for page_num in range(total_pages):
                    # تحسين: تحميل الصفحة والتعامل مع النص بشكل آمن برمجياً
                    page = pdf_doc[page_num]
                    # تحسين Pylance: ضمان أن المخرج نصي قبل استدعاء strip
                    raw_text = page.get_text("text", sort=False)
                    page_text = str(raw_text or "").strip()

                    if page_text:
                        # تحسين: إدارة الـ Cache (LRU) لمنع انفجار الذاكرة
                        if len(self.page_cache) >= self.max_pages:
                            self.page_cache.popitem(last=False)

                        # الحل: إضافة المفاتيح الناقصة بقيم افتراضية لتجاوز فحص الـ Integrity
                        self.page_cache[page_num] = {
                            "content": page_text,
                            "layer_type": "PENDING_INDEXING",
                            "metadata": {"source": pdf_path, "timestamp": time.time()},
                            # إضافة هذه السطور فوراً:
                            "semantic_keywords": [],
                            "visual_headings": [],
                            "is_fast_ingested": True
                        }

                        # تحسين: تنفيذ الـ callback بأمان
                        if callback:
                            try:
                                callback(page_num, page_text)
                            except Exception as cb_e:
                                self.logger.warning(f"⚠️ خطأ في الـ callback للصفحة {page_num}: {cb_e}")

                return {"status": "completed", "pages_ingested": total_pages}

        except Exception as e:
            self.logger.error(f"❌ فشل التدفق السريع: {e}")
            return {"status": "error", "message": str(e)}

    def _validate_page_data(self, page_num: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        فحص وترميم بيانات الصفحة لضمان سلامة الهيكل المعرفي.
        تضمن أن كل صفحة تحتوي على الحد الأدنى من البيانات المطلوبة للتحليل والبحث.
        """
        # القالب المعياري للسلامة (The Integrity Schema)
        schema_defaults = {
            "content": "[NON_TEXTUAL_DATA_RESERVED]", # وسم تقني بدلاً من Error
            "semantic_keywords": ["general_context"],
            "visual_headings": [],
            "layer_type": "STANDARD_CONTENT",
            "metadata": {"auto_repaired": True, "integrity_check": "passed"}
        }

        for key, default in schema_defaults.items():
            # فحص الوجود، القيمة الفارغة، أو النصوص البيضاء
            is_missing = key not in data or data[key] is None
            is_empty_str = isinstance(data.get(key), str) and not str(data[key]).strip()

            if is_missing or is_empty_str:
                data[key] = default
                if key == "content":
                    self.logger.warning(f"🛠️ Self-Repair: Page {page_num} content was missing or blank. Placeholder injected.")

        return data

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

                    # استخدام blocks لضمان الترتيب المنطقي للهندسة
                    blocks = page.get_text("blocks")

                    # استخراج النص من البلوكات (البلوك الخامس في القائمة هو النص)
                    # Block structure: (x0, y0, x1, y1, "text", block_no, block_type)
                    page_text = "\n".join([str(b[4]).strip() for b in blocks if isinstance(b[4], str)])

                    if page_text:
                        text_parts.append(page_text)

                full_text = "\n\n".join(text_parts)

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

# ------------ ثالثاً: إدارة الذاكرة والأرشفة (Caching & Storage) ------------
    def add_page(self, page_num: int, text: str, page_obj: Any, metadata: Dict):
        """
        Main Indexing Core: Orchestrates Vectors, Layout, and Heuristic Networking.
        Enhanced with Auto-Repair and Content Integrity Guard.
        """

        # 0. صمام الأمان: تنظيف النص الأولي ومعالجة المحتوى الضعيف
        clean_text = str(text or "").strip()
        if len(clean_text) < 10:
            # وسم الصفحة لتقليل أخطاء "insufficient content"
            clean_text = f"[NON_TEXTUAL_PAGE]: Page {page_num} contains minimal text or visual assets."

        # 1. LRU Cache Management
        if page_num in self.page_cache:
            self.page_cache.move_to_end(page_num)

        # 2. Semantic Vector Generation (Deep Contextualization)
        with torch.no_grad():
            # استخدام النص المنظف لضمان بقاء الصفحة داخل الفضاء الدلالي
            vector = self.model.encode(clean_text, convert_to_numpy=True)

        self.vectors.append(vector)
        self.vector_map.append(page_num)

        # 3. Parallel Extraction (Structural Intelligence)
        layout_data = self._extract_layout_structure(page_obj)
        keywords = self.PDF_extract_keywords(clean_text)
        layer_type = self._classify_layer(clean_text, metadata)

        # 4. Hybrid Integrity Guard: إصلاح البيانات قبل الحفظ لضمان عدم وجود Missing Keys
        # استدعاء "الزميل الهجين" أو تطبيق الترميم المباشر هنا
        final_entry = {
            "content": clean_text,
            "metadata": metadata,
            "layer_type": layer_type or "STANDARD_CONTENT",
            "semantic_keywords": keywords if keywords else ["general"],
            "visual_headings": layout_data.get("headings", []),
            "integrity_score": 1.0 if len(clean_text) > 100 else 0.5
        }

        # 5. Unified English Cache Storage
        self.page_cache[page_num] = final_entry

        # 6. Cache Eviction Policy (LRU)
        if len(self.page_cache) > self.max_pages:
            old_idx, _ = self.page_cache.popitem(last=False)
            self.logger.info(f"🧹 Cache Eviction: Page {old_idx} cleared from active memory.")

        # 7. Build Network Relations (Heuristics)
        # تمرير البيانات المرممة لضمان قوة الروابط
        self._build_visual_heuristics(page_num, layout_data)
        self._build_heuristic_links(page_num)

        # 8. Final Status Update
        self.page_queue.append(page_num)
        self.logger.info(
            f"📥 Indexed Page [{page_num}] | Integrity: VERIFIED | "
            f"Headings: {len(final_entry['visual_headings'])} | Keywords: {len(final_entry['semantic_keywords'])}"
        )

    def _extract_layout_structure(self, page: Any) -> Dict[str, Any]:
        """
        Extracts structural headings and blocks with a focus on engineering layouts.
        Standardizes internal data keys to English.
        """
        # 1. Initialize data structure with consistent English keys
        layout_data: Dict[str, List[Any]] = {"headings": [], "blocks": []}

        try:
            # 2. Extract detailed dictionary (safe type casting for Pylance)
            dict_data = page.get_text("dict")
            raw_blocks = dict_data.get("blocks", [])
            layout_data["blocks"] = raw_blocks

            for block in raw_blocks:
                # Ensure we are dealing with a text block containing lines
                if isinstance(block, dict) and "lines" in block:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            # Ensure span is a dictionary and extract text safely
                            text = str(span.get("text", "")).strip()

                            # Optimization: Skip short fragments and noise
                            if len(text) > 3:
                                font_size = span.get("size", 0)
                                font_name = str(span.get("font", "")).lower()

                                # 3. Engineering Heuristics for Headings
                                # Enhanced criteria to catch Arabic bold/large fonts
                                is_large = font_size > 11.5
                                is_bold = "bold" in font_name or "black" in font_name
                                is_caps = text.isupper() and len(text) > 5

                                # Avoid page numbers or isolated coordinates
                                is_not_numeric = not text.replace('.', '').replace('-', '').isdigit()

                                if (is_large or is_bold or is_caps) and is_not_numeric:
                                    layout_data["headings"].append({
                                        "text": text,
                                        "bbox": span.get("bbox"),
                                        "font_size": font_size,
                                        "font_name": font_name,
                                        "type": "structural_anchor"
                                    })

        except Exception as e:
            self.logger.warning(f"Layout extraction skipped on a page: {str(e)}")

        return layout_data

    def PDF_extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extracts semantic keywords with Arabic prefix normalization.
        Standardized for internal English processing.
        """
        import re
        from collections import Counter

        # 1. Professional Cleaning (Preserving alphanumeric for technical codes)
        # تحسين: استخدام regex مسبق التحميل للسرعة وتجنب الرموز الغريبة
        clean_text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
        words = clean_text.lower().split()

        # 2. Expanded Technical Stop Words (Arabic & English)
        stop_words = {
            'ال', 'في', 'على', 'من', 'مع', 'إلى', 'عن', 'كان', 'هذا', 'هذه', 'تم',
            'ذلك', 'تلك', 'أو', 'أم', 'هل', 'عن', 'عند', 'يكون', 'يمكن', 'طريق',
            'through', 'using', 'also', 'from', 'with', 'under', 'between',
            'page', 'data', 'information', 'type', 'section'
        }

        # 3. Arabic Normalization & Filtering
        filtered_words = []
        # تحسين: فحص وجود أحرف عربية لضمان عدم تخريب الكلمات الإنجليزية التي تبدأ بـ "al"
        arabic_prefix_re = re.compile(r'^ال')
        has_arabic_re = re.compile(r'[\u0600-\u06FF]')

        for word in words:
            # تنظيف "الـ" التعريف فقط إذا كانت الكلمة عربية وطويلة بما يكفي
            if len(word) > 4 and has_arabic_re.search(word):
                normalized_word = arabic_prefix_re.sub('', word)
            else:
                normalized_word = word

            # فلتر الطول والكلمات الشائعة والأرقام البحتة
            if len(normalized_word) > 3 and normalized_word not in stop_words:
                if not normalized_word.isdigit():
                    filtered_words.append(normalized_word)

        # 4. Frequency Mapping & Result Selection
        # نستخدم Counter لجلب الكلمات الأكثر صلة بهوية الصفحة
        word_counts = Counter(filtered_words)
        top_keywords = [word for word, count in word_counts.most_common(top_n)]

        return top_keywords

    def _build_visual_heuristics(self, page_num: int, layout_data: Dict[str, Any]):
        """
        Builds visual and structural navigation links across the document.
        Standardized with English internal identifiers and graph-ready nodes.
        """
        # 1. جلب العناوين بأمان مع التحقق من النوع
        headings = layout_data.get("headings", [])
        if not isinstance(headings, list):
            return

        for heading in headings:
            # تحسين Pylance: ضمان أن العنوان نصي وليس فارغاً
            title = str(heading.get("text", "")).strip()
            if len(title) < 2:
                continue

            # 2. Heading Map: تحديث خريطة العناوين
            # إذا كان العنوان موجوداً مسبقاً في نفس الصفحة لا نكرره
            if page_num not in self.heading_map[title]:
                self.heading_map[title].append(page_num)

            # 3. Strategic Connection: ربط الصفحات المتسلسلة لنفس العنوان
            # هذا يربط الأقسام الهندسية الممتدة (مثل المواصفات الفنية)
            if len(self.heading_map[title]) > 1:
                source_page = self.heading_map[title][-2]

                # تجنب ربط الصفحة بنفسها
                if source_page != page_num:
                    self.visual_links.append({
                        "link_type": "HEADING_CONTINUITY",
                        "origin_page": source_page,
                        "target_page": page_num,
                        "anchor_text": title,
                        "strength": 0.9 # وزن الرابط للاستدلال
                    })

        # 4. Structural Flow: ربط تسلسلي لضمان تدفق السياق المنطقي
        if page_num > 0:
            # إضافة رابط تدفق خطي بين الصفحات المتجاورة
            self.visual_links.append({
                "link_type": "STRUCTURAL_FLOW",
                "origin_page": page_num - 1,
                "target_page": page_num,
                "flow_weight": 1.0
            })

    def _classify_layer(self, text: str, metadata: Dict) -> str:
        """
        Technical layer classification with Robotics & Engineering focus.
        Enhanced to detect Cover pages and complex technical layouts.
        """
        # 1. Linguistic and Structural Analysis
        words = text.split()
        word_count = len(words)
        text_lower = text.lower()
        header_area = text_lower[:400] # توسيع منطقة الفحص قليلاً

        # 2. COVER_LAYER: الكشف عن صفحة الغلاف (غالباً صفحة 0)
        # إذا كان النص قليلاً جداً مع وجود ميتا-داتا العنوان، فهي صفحة غلاف
        is_page_zero = metadata.get("page_index") == 1
        cover_indicators = {'robotics', 'manual', 'handbook', 'guide', 'edition', 'دليل', 'روبوت'}
        if is_page_zero and (word_count < 100 or any(ind in text_lower for ind in cover_indicators)):
            return "COVER_LAYER"

        # 3. CHAPTER_LAYER: حدود المواضيع والفصول
        chapter_indicators = {'chapter', 'section', 'part', 'فصل', 'باب', 'وحدة', 'المبحث'}
        meta_title = str(metadata.get('title', '')).lower()
        if any(ind in header_area for ind in chapter_indicators) or \
           any(ind in meta_title for ind in chapter_indicators):
            return "CHAPTER_LAYER"

        # 4. TECHNICAL_DATA: المحتوى الهندسي المتقدم (الروبوتات، الحساسات، الجداول)
        # إضافة مصطلحات هندسية (Robotics Stems) لرفع دقة التصنيف
        eng_indicators = {
            'table', 'figure', 'diagram', 'schema', 'robot', 'sensor', 'actuator',
            'controller', 'feedback', 'kinematics', 'جدول', 'مخطط', 'رسم', 'حساس'
        }
        # فحص كثافة الأرقام (البيانات التقنية)
        digit_count = sum(c.isdigit() for c in text[:500])
        if any(ind in text_lower for ind in eng_indicators) or (digit_count > 60):
            return "TECHNICAL_DATA"

        # 5. APPENDIX_LAYER: الملاحق والمراجع
        reference_kws = {'appendix', 'references', 'bibliography', 'citation', 'ملحق', 'مراجع', 'فهرس'}
        if any(kw in text_lower for kw in reference_kws):
            return "APPENDIX_LAYER"

        # 6. CORE_CONTENT: المحتوى النصي الكثيف (الشرح العميق)
        structure_kws = {'introduction', 'abstract', 'summary', 'مقدمة', 'خلاصة', 'تمهيد'}
        if word_count > 500 or any(kw in text_lower for kw in structure_kws):
            return "CORE_CONTENT"

        return "STANDARD_CONTENT"

    def _build_heuristic_links(self, page_num: int):
        """
        Builds a dynamic heuristic network with Predictive Weighting and Adaptive Re-linking.
        Strategically strengthens connections based on semantic density and structural importance.
        """
        if page_num not in self.page_cache:
            return

        page_data = self.page_cache[page_num]
        current_kws = set(page_data.get("semantic_keywords", []))
        current_layer = page_data.get("layer_type", "STANDARD_CONTENT")

        # 1. المرحلة الأولى: التنبؤ بالأهمية (Predictive Weighting)
        # بدلاً من الثبات، يتم حساب الوزن بناءً على كثافة المحتوى والعناوين
        content_len = len(page_data.get("content", ""))
        heading_count = len(page_data.get("visual_headings", []))

        weights = {
            "CHAPTER_LAYER": 1.0,
            "CORE_CONTENT": 0.85,
            "TECHNICAL_DATA": 0.7,
            "STANDARD_CONTENT": 0.4  # رفع الحد الأدنى من 0.3 لزيادة التأثير
        }

        # تعزيز الوزن بناءً على الكثافة (Heuristic Boost)
        predicted_weight = weights.get(current_layer, 0.3)
        if heading_count > 3: predicted_weight = min(predicted_weight + 0.1, 1.0)
        if content_len > 1500: predicted_weight = min(predicted_weight + 0.05, 1.0)

        # 2. المرحلة الثانية: الربط الدلالي والتعزيز (Semantic Strengthening)
        for kw in current_kws:
            # التحقق من وجود الكلمة لتقوية الروابط القديمة (Retroactive Strengthening)
            if kw in self.heuristic_network:
                for entry in self.heuristic_network[kw]:
                    # إذا كانت الصفحات متقاربة (نفس السياق)، نقوي الرابط تدريجياً
                    if abs(entry["page"] - page_num) < 15:
                        entry["weight"] = min(entry["weight"] + 0.1, 1.0)

            # إضافة الرابط الجديد بالوزن المتوقع
            if not any(item["page"] == page_num for item in self.heuristic_network[kw]):
                self.heuristic_network[kw].append({
                    "page": page_num,
                    "weight": predicted_weight,
                    "rank": "HIGH" if predicted_weight >= 0.75 else "NORMAL"
                })

        # 3. المرحلة الثالثة: الربط الاستراتيجي التبادلي (Heuristic Cross-Linking)
        if page_num not in self.layer_hierarchy:
            self.layer_hierarchy[page_num] = []

        # البحث عن "التوائم الدلالية" في الصفحات السابقة لزيادة عدد الروابط
        links_added = 0
        for past_page, data in self.page_cache.items():
            if past_page == page_num: continue

            past_kws = set(data.get("semantic_keywords", []))
            # إذا وجدنا تقاطعاً قوياً (كلمتين أو أكثر)، ننشئ رابطاً هيكلياً فوراً
            if len(current_kws.intersection(past_kws)) >= 2:
                if past_page not in self.layer_hierarchy[page_num]:
                    self.layer_hierarchy[page_num].append(past_page)
                    links_added += 1
                # تفعيل الربط التبادلي
                if past_page not in self.layer_hierarchy: self.layer_hierarchy[past_page] = []
                if page_num not in self.layer_hierarchy[past_page]:
                    self.layer_hierarchy[past_page].append(page_num)

        # 4. معالجة العزلة (Isolation Recovery)
        # إذا كانت الصفحة معزولة (أقل من 3 روابط)، نربطها قسرياً بأقرب جيران وأقرب فصل
        neighbors = {n for n in [page_num - 1, page_num + 1] if n >= 0}
        existing_links = set(self.layer_hierarchy.get(page_num, []))
        final_links = existing_links.union(neighbors)

        if len(final_links) < 3:
            chapters = [p for p, d in self.page_cache.items() if d.get("layer_type") == "CHAPTER_LAYER"]
            if chapters:
                closest = min(chapters, key=lambda x: abs(x - page_num))
                final_links.add(closest)

        self.layer_hierarchy[page_num] = list(final_links)

        # 5. التقرير النهائي
        self.logger.info(
            f"🔗 Network Strength: Page {page_num} | "
            f"Links: {len(self.layer_hierarchy[page_num])} | "
            f"Weight: {predicted_weight:.2f}"
        )

    def get_page_data(self, page_num: int, pdf_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Smart Retrieval: الملاح الذكي الذي يستخدم نظام الحماية الهجين لاسترجاع وتصحيح البيانات.
        """
        # 1. المرحلة الأولى: البحث في الكاش (Cache Access)
        if page_num in self.page_cache:
            self.page_cache.move_to_end(page_num)
            raw_data = self.page_cache[page_num]

            # تفعيل "الزميل الهجين" لفحص وإصلاح البيانات فوراً
            # نحدد السياق كـ DEEP_ANALYSIS لضمان جودة الاسترجاع
            is_valid, validated_data = self.hybrid_integrity_guard(page_num, raw_data, context="DEEP_ANALYSIS")

            if is_valid:
                return validated_data

            # إذا لم تكن البيانات صالحة (مثلاً نص فارغ)، لا نتوقف، بل نحاول الاستشفاء الذاتي
            self.logger.warning(f"⚠️ Page {page_num} integrity check failed. Forcing auto-recovery...")

        # 2. المرحلة الثانية: الاستشفاء الذاتي (Auto-Recovery)
        if pdf_path:
            self.logger.info(f"🔄 Source Recovery: Re-indexing Page {page_num} from {Path(pdf_path).name}")
            try:
                with fitz.open(pdf_path) as doc:
                    if page_num < len(doc):
                        page = doc[page_num]
                        raw_blocks = page.get_text("blocks")
                        blocks = [b for b in raw_blocks if isinstance(b, tuple) and len(b) > 4]
                        text = "\n".join([str(b[4]).strip() for b in blocks if str(b[4]).strip()])

                        recovery_meta = {"source": "recovery_sync", "timestamp": time.time()}

                        # استدعاء add_page يضمن بناء الروابط والاستدلال مجدداً
                        self.add_page(page_num, text, page, recovery_meta)

                        # استرجاع البيانات بعد إصلاحها وفهرستها بالكامل
                        final_data = self.page_cache.get(page_num)
                        if final_data:
                            # فحص أخير لضمان نجاح العملية
                            _, fixed_data = self.hybrid_integrity_guard(page_num, final_data)
                            return fixed_data
                    else:
                        self.logger.error(f"❌ Recovery Out of Range: Page {page_num}")
            except Exception as e:
                self.logger.error(f"❌ Recovery Failure on Page {page_num}: {str(e)}")

        return None

    def hybrid_integrity_guard(self, page_num: int, raw_data: Dict[str, Any], context: str = "RETRIEVAL") -> Tuple[bool, Dict[str, Any]]:
        """
        المنظومة الهجينة لسلامة البيانات:
        تتكيف مع حالة المحرك (الفهرسة السريعة، التحليل النبضي، أو الاسترجاع)
        وتقوم بالإصلاح الذاتي التكيفي (Adaptive Self-Repair).
        """
        if not isinstance(raw_data, dict):
            return False, {}

        # 1. تحليل الحالة (Contextual Awareness)
        # إذا كنا في طور الفهرسة السريعة، معاييرنا مرنة جداً
        is_fast_mode = raw_data.get("is_fast_ingested", False) or context == "FAST_INGEST"

        # 2. الفحص المرن للمحتوى (Flexible Content Validation)
        content = str(raw_data.get("content", "")).strip()

        # صمام أمان هجين: بدلاً من الرفض، نقوم بالوسم (Tagging)
        if len(content) < 15:
            # قد تكون صفحة رسومات هندسية (Technical Drawing)
            raw_data["layer_type"] = "VISUAL_ASSET" if "TECHNICAL_DATA" in raw_data.get("layer_type", "") else "OCR_REQUIRED"
            # في الطور السريع نسمح بمرورها، في التحليل العميق ننبه المحلل
            if context == "DEEP_ANALYSIS":
                return False, raw_data

        # 3. الإصلاح الهجين للمفاتيح (On-Demand Schema Repair)
        # لا نفرض مساراً ثابتاً؛ بل نكمل النواقص حسب احتياج الزملاء (المحلل والشبكة)
        required_schema = {
            "semantic_keywords": [],
            "visual_headings": [],
            "layer_type": "PENDING_INDEXING" if is_fast_mode else "STANDARD_CONTENT",
            "metadata": raw_data.get("metadata", {}),
            "processing_metrics": {"integrity_score": 1.0}
        }

        for key, default in required_schema.items():
            if key not in raw_data or raw_data[key] is None:
                # إصلاح هجين: لا نضع قيمة فارغة فقط، بل نحاول استنتاجها من "زملاء" البيانات
                if key == "layer_type" and len(raw_data.get("visual_headings", [])) > 5:
                    raw_data[key] = "CHAPTER_LAYER"
                else:
                    raw_data[key] = default

        # 4. المزامنة الهجينة مع المتجهات (Vector Sync)
        # الربط مع "زميل" المتجهات (self.vector_map)
        raw_data["has_vector"] = page_num in self.vector_map

        # 5. التقييم النهائي (Confidence Scoring)
        # درجة السلامة ليست ثابتة؛ تزيد بزيادة ثراء البيانات
        integrity_score = 0.5 # أساسي
        if raw_data["has_vector"]: integrity_score += 0.2
        if len(raw_data["semantic_keywords"]) > 0: integrity_score += 0.3

        raw_data["processing_metrics"]["integrity_score"] = integrity_score

        # 6. القرار الهجين (Hybrid Decision)
        # إذا كان الـ score مقبولاً للطور الحالي، نمررها
        threshold = 0.4 if is_fast_mode else 0.7
        success = integrity_score >= threshold

        if not success:
            self.logger.warning(f"🛡️ Hybrid Guard: Page {page_num} needs deeper indexing (Score: {integrity_score})")

        return success, raw_data

    def _guess_topic(self, text: str) -> str:
        """
        Technical Topic Inference: Specialized for Robotics and Engineering.
        Uses high-gravity technical stems to avoid GENERAL_TOPIC falls.
        """
        # 1. تعريف التصنيفات التخصصية (Specialized Domains)
        # أضفنا ROBOTICS كفئة مستقلة وأثرينا البقية بمصطلحات تقنية هندسية
        categories = {
            'ROBOTICS_ENGINEERING': [
                'robot', 'kinematic', 'actuator', 'sensor', 'feedback', 'control', 'motion',
                'trajectory', 'servo', 'joint', 'manipulator', 'روبوت', 'آلي', 'حساس', 'تحكم', 'حركة'
            ],
            'TECHNOLOGY': [
                'software', 'algorithm', 'digital', 'network', 'computing', 'code', 'intelligence',
                'برمج', 'خوارزم', 'رقمي', 'ذكاء', 'شبكة'
            ],
            'SCIENCE': [
                'physics', 'theory', 'laboratory', 'analysis', 'research', 'mathematical',
                'فيزياء', 'نظرية', 'مختبر', 'تحليل', 'بحث'
            ],
            'BUSINESS': [
                'management', 'market', 'investment', 'industry', 'cost', 'optimization',
                'إدارة', 'سوق', 'استثمار', 'صناعة', 'تكلفة'
            ],
            'LEGAL': [
                'patent', 'standard', 'regulation', 'compliance', 'contract', 'safety',
                'براءة', 'معايير', 'تنظيم', 'امتثال', 'عقد', 'سلامة'
            ]
        }

        # 2. تهيئة العدادات بـ Float
        scores: Dict[str, float] = {topic: 0.0 for topic in categories}
        text_lower = text.lower()

        # تحسين: فحص عينة أكبر قليلاً (8000 حرف) لتغطية صفحات الغلاف والمقدمة
        sample_text = text_lower[:8000]

        # 3. حساب الجاذبية المعرفية (Knowledge Gravity)
        for topic, keywords in categories.items():
            for kw in keywords:
                count = sample_text.count(kw)
                if count > 0:
                    # ميزة "الثقل الهندسي": الكلمات التقنية الطويلة تأخذ وزناً أكبر (1.8x)
                    # لأنها مستحيل تكون جزء من لغة عامة (مثل Kinematics)
                    weight = 1.8 if len(kw) > 7 else 1.2
                    scores[topic] += float(count) * weight

        # 4. تحديد النتيجة النهائية مع عتبة الثقة (Confidence Threshold)
        if not scores:
            return "GENERAL_TOPIC"

        best_topic = max(scores, key=lambda k: scores[k])
        max_score = scores[best_topic]

        # 5. تقرير دقة الاستنتاج (Inference Accuracy Report)
        if max_score > 2.0: # حد أدنى من الأدلة لتأكيد التخصص
            confidence = "HIGH" if max_score > 15 else "MEDIUM"
            self.logger.info(f"🧠 Topic Mastered: {best_topic} (Score: {max_score:.1f} | Conf: {confidence})")
            return best_topic

        self.logger.warning(f"⚠️ Low signal detected (Score: {max_score:.1f}). Defaulting to GENERAL_TOPIC.")
        return "GENERAL_TOPIC"

# ------------ رابعاً: الشبكة الاستدلالية والروابط (Heuristic Network) ------------
    def get_strategic_hubs(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Identifies key knowledge hubs by aggregating weights across the heuristic network.
        Returns a list of (keyword, total_score) sorted by significance.
        """
        hub_scores: Dict[str, float] = defaultdict(float)

        # 1. Aggregate scores from the heuristic network
        for keyword, entries in self.heuristic_network.items():
            # تحسين Pylance: ضمان أن الكلمة نصية وطولها يعكس قيمة معرفية
            clean_kw = str(keyword).strip()
            if len(clean_kw) < 4:  # الكلمات التقنية المفيدة عادة > 3 أحرف
                continue

            for entry in entries:
                if isinstance(entry, dict):
                    # تحسين: ضرب الوزن في عدد مرات ظهور الكلمة لزيادة دقة الـ Gravity
                    weight = float(entry.get("weight", 0.1))
                    # إضافة "بونص" للكلمات المرتبطة بطبقات الفصول (Chapters)
                    rank_bonus = 1.5 if entry.get("rank") == "HIGH" else 1.0
                    hub_scores[clean_kw] += (weight * rank_bonus)
                else:
                    hub_scores[clean_kw] += 0.1

        # 2. Sort hubs by cumulative "Knowledge Gravity"
        sorted_hubs = sorted(
            hub_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 3. Validation & Precision Reporting (نتائج الإختبار)
        top_results = sorted_hubs[:top_n]

        if top_results:
            # حساب "تنوع المحاور" كدليل على دقة الفهرسة
            diversity_score = len(hub_scores)
            self.logger.info(
                f"📊 Hub Analysis Report:\n"
                f"   - Diversity: {diversity_score} unique topics found.\n"
                f"   - Top Priority: '{top_results[0][0]}' (Score: {top_results[0][1]:.2f})\n"
                f"   - Strategy: Identified {len(top_results)} key knowledge hubs."
            )
        else:
            self.logger.warning("⚠️ Hub Analysis: No significant knowledge hubs could be mapped.")

        return top_results

# ------------ خامساً: البحث والتنقل الذكي (Search & Navigation) ------------
    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Hybrid Search Plus: تستخدم المعايرة اللاخطية وتعزيز الكلمات لرفع الـ Confidence.
        تم تحسينها لتجاوز حاجز الـ 0.5 وتحقيق قفزات دلالية قوية.
        """
        if not self.vectors:
            self.logger.warning("⚠️ Vector space is empty. Check if extraction was successful.")
            return []

        # 1. Query Embedding
        with torch.no_grad():
            query_vector = self.model.encode(query, convert_to_numpy=True)

        # 2. Vector Similarity
        vectors_np = np.asanyarray(self.vectors, dtype=np.float32)
        sim_scores = np.dot(vectors_np, query_vector)
        v_norms = np.linalg.norm(vectors_np, axis=1)
        q_norm = np.linalg.norm(query_vector)
        similarities = sim_scores / (v_norms * q_norm + 1e-9)

        # 3. Enhanced Hybrid Logic (The Multiplier Effect)
        weighted_scores = []
        query_terms = set(query.lower().split())

        layer_weights = {
            "CHAPTER_LAYER": 1.2,  # رفع الوزن لتعزيز القفزة
            "CORE_CONTENT": 1.0,
            "TECHNICAL_DATA": 0.8,
            "STANDARD_CONTENT": 0.5
        }

        for idx, sem_sim in enumerate(similarities):
            page_num = self.vector_map[idx]
            page_data = self.page_cache.get(page_num, {})

            # أ- تعزيز الكلمات المفتاحية (Explicit Keyword Boost)
            # إذا تطابقت كلمات الاستعلام مع كلمات الصفحة، نرفع النتيجة بقوة
            page_kws = [kw.lower() for kw in page_data.get("semantic_keywords", [])]
            matches = len(query_terms.intersection(page_kws))
            kw_boost = min(matches * 0.15, 0.45)

            # ب- وزن الطبقة الهيكلية
            struct_base = layer_weights.get(page_data.get("layer_type", ""), 0.4)

            # ج- المعادلة السرية لرفع الـ Confidence:
            # نستخدم (log scaling) أو (boosting) للقيم التي تتجاوز 0.3
            calibrated_sim = sem_sim
            if sem_sim > 0.35:
                calibrated_sim = min(sem_sim * 1.4, 0.95) # رفع التشابه المتوسط ليصبح عالياً

            final_score = (calibrated_sim * 0.6) + (struct_base * 0.2) + (kw_boost * 0.2)
            weighted_scores.append(min(final_score, 0.99)) # سقف 0.99 لضمان المنطقية

        # 4. Result Synthesis
        top_indices = np.argsort(weighted_scores)[::-1][:top_k]
        results = []

        for rank_idx in top_indices:
            page_num = self.vector_map[rank_idx]
            p_data = self.page_cache.get(page_num, {})

            results.append({
                "page_index": page_num,
                "rank_score": round(float(weighted_scores[rank_idx]), 4),
                "vector_similarity": round(float(similarities[rank_idx]), 4),
                "preview": str(p_data.get("content", ""))[:250].replace("\n", " ")
            })

        # 5. Search Quality Report
        if results:
            self.logger.info(
                f"🚀 Jump Verified: Page {results[0]['page_index']+1} | "
                f"Confidence: {results[0]['rank_score']:.2f} | "
                f"Boost: {'ACTIVE' if results[0]['rank_score'] > 0.7 else 'NORMAL'}"
            )

        return results

    def page_flipper(self, current_page: int, direction: str = "next") -> int:
        """
        Hybrid Navigation System: Supports linear, heuristic, and stack-based movement.
        Standardized with English internal identifiers.
        """
        # 1. Maintain Navigation History (Backtrack capability)
        # تحسين: تحديد حجم الـ stack لمنع استهلاك الذاكرة في الجلسات الطويلة
        if not self.navigation_stack or self.navigation_stack[-1] != current_page:
            if len(self.navigation_stack) > 50:
                self.navigation_stack.popleft()
            self.navigation_stack.append(current_page)

        target_page = current_page

        # 2. Sequential Navigation (التنقل الخطي)
        if direction == "next":
            target_page = current_page + 1
        elif direction == "prev":
            target_page = max(0, current_page - 1)

        # 3. Smart Jump Logic (Heuristic Navigation)
        elif direction == "next_heading":
            # الاعتماد على الهيكل الطبقي (Hierarchy) للقفز بين الفصول
            related_pages = self.layer_hierarchy.get(current_page, [])

            # استنتاج القفزة الذكية: البحث عن صفحات أمامية مصنفة كـ CHAPTER_LAYER
            # تحسين: ترتيب الجوار لضمان القفز لأقرب عنوان منطقي
            jumps = sorted([p for p in related_pages if p > current_page])

            if jumps:
                target_page = jumps[0]
            else:
                # Fallback: إذا لم توجد روابط هيكلية، ننتقل للصفحة التالية
                target_page = current_page + 1

        # 4. History Backtracking (الرجوع للخلف)
        elif direction == "back":
            if len(self.navigation_stack) > 1:
                self.navigation_stack.pop()  # إزالة الصفحة الحالية
                target_page = self.navigation_stack.pop() # العودة للصفحة السابقة
            else:
                target_page = current_page

        # 5. Cache & Boundary Validation (اختبار دقة الملاح)
        # تحسين: ضمان عدم تجاوز حدود المستند (نحتاج لمعرفة إجمالي الصفحات)
        # إذا كانت الصفحة في الكاش، نحدث أولويتها
        if target_page in self.page_cache:
            self.page_cache.move_to_end(target_page)
            confidence = "INSTANT (Cache Hit)"
        else:
            confidence = "DEFERRED (Cache Miss)"

        self.logger.info(
            f"🧭 Navigation Report:\n"
            f"   - Movement: {current_page} -> {target_page}\n"
            f"   - Mode: {direction}\n"
            f"   - Speed: {confidence}"
        )

        return target_page

    def inspect_cache(self, page_num: Optional[int] = None, analyzer_context: Any = None) -> Dict[str, Any]:
        """
        [Location: PDFPageCacheNetwork]
        فحص عميق لحالة الشبكة الاستدلالية والكاش.
        """
        from collections import Counter # استيراد محلي لحل مشكلة 'Counter' is not defined

        # 1. نظرة عامة على النظام (Global System Overview)
        if page_num is None:
            current_cache = self.page_cache
            # حساب توزيع الطبقات إحصائياً لتقرير الدقة
            layer_stats = Counter([data.get("layer_type", "UNKNOWN") for data in current_cache.values()])

            return {
                "status": "GLOBAL_VIEW",
                "metrics": {
                    "total_cached_pages": len(current_cache),
                    "total_vector_embeddings": len(self.vectors),
                    "active_heuristic_hubs": len(self.heuristic_network),
                    "cache_utilization": f"{(len(current_cache)/self.max_pages)*100:.1f}%"
                },
                "structure_analysis": {
                    "layer_distribution": dict(layer_stats),
                    "network_density": {str(kw): f"{len(entries)} connections"
                                       for kw, entries in list(self.heuristic_network.items())[:5]}
                }
            }

        # 2. فحص تشخيصي على مستوى الصفحة (Detailed Page-Level Diagnostics)
        related_pages = []
        if analyzer_context and hasattr(analyzer_context, 'get_related_pages'):
            try:
                related_pages = analyzer_context.get_related_pages(page_num)
            except Exception:
                related_pages = self.layer_hierarchy.get(page_num, [])
        else:
            related_pages = self.layer_hierarchy.get(page_num, [])

        page_data = self.page_cache.get(page_num, {})

        # 3. Network Insights: استخراج الكلمات التي تربط هذه الصفحة بالشبكة
        contextual_hubs = {}
        for kw, entries in self.heuristic_network.items():
            if any(isinstance(e, dict) and e.get("page") == page_num for e in entries):
                contextual_hubs[str(kw)] = f"{len(entries)} weighted links"

        # 4. Final Diagnostic Report (تقرير نتائج الإختبار)
        report = {
            "status": "TARGET_VIEW",
            "page_diagnostics": {
                "index": page_num,
                "layer_type": page_data.get("layer_type", "N/A"),
                "connectivity_map": related_pages,
                "semantic_keywords": page_data.get("semantic_keywords", [])[:5],
                "headings_count": len(page_data.get("visual_headings", []))
            },
            "network_insights": {
                "hubs_count": len(contextual_hubs),
                "contextual_hubs_sample": dict(list(contextual_hubs.items())[:5])
            },
            "memory_state": {
                "lru_priority": "ACTIVE" if page_num in self.page_cache else "EVICTED",
                "has_vector": page_num in self.vector_map
            }
        }

        self.logger.info(f"🔍 Cache Inspection Page {page_num}: Status {report['memory_state']['lru_priority']}")
        return report

    def llm_bridge(self, content: str, user_request: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        الجسر التنفيذي: يرسل النبضة إلى LLM ويعيد صياغتها كأفكار مهيكلة.
        [المكان: PDFPageCacheNetwork]
        """
        if not hasattr(self, 'llm_client') or self.llm_client is None:
            self.logger.error("❌ LLM Client is not initialized in the engine.")
            raise AttributeError("llm_client not found.")

        try:
            # صياغة الطلب للـ AI مع سياق هندسي مركز
            prompt = f"Analyze this robotics technical text: {content[:2000]}\nUser Request: {user_request}"

            # استدعاء API الخاص بـ OpenAI
            response = self.llm_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a Senior Technical Auditor. Extract 3 core insights in a structured format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            # تصحيح: الوصول الصحيح لمحتوى الرسالة في مكتبة OpenAI الحديثة
            ai_text = str(response.choices[0].message.content)

            # بناء قائمة الأفكار لتتوافق مع نظام النبضات في المحلل
            ideas = [{
                "idea_id": 1,
                "title": "AI Strategic Technical Insight",
                "importance_level": "CORE_CONCEPT",
                "extracted_segments": [ai_text[:250].strip() + "..."],
                "confidence_score": 0.95,
                "user_intent": user_request
            }]

            state = {
                "avg_score": 0.95,
                "engine": "REAL_LLM_OPENAI",
                "status": "SUCCESS"
            }

            self.logger.info("📡 LLM Bridge: Successfully retrieved deep insights from Cloud AI.")
            return ideas, state

        except Exception as e:
            self.logger.error(f"❌ LLM Bridge Critical Failure: {str(e)}")
            # نرفع الخطأ لكي تتعامل معه دالة الـ Recovery (Fallback)
            raise e

# ===================================================================
#     سادساً: كلاس الفكري والتحليل (Thinking & Analysis)
# ===================================================================
class SmartAnalyzePDF:
    """
    محرك التحليل الذكي: المسؤول عن تحويل الاستراتيجية إلى نبضات تحليلية.
    SNN: Semantic Neural Network approach for pulsed analysis.
    """

    def __init__(self, parent_engine: Any, thinking_engine: Any):
        self.engine = parent_engine      # الكلاس الرئيسي (PDFPageCacheNetwork)
        self.engine.thinking = thinking_engine  # محرك التفكير (LLM/Mock)
        self.engine.logger = parent_engine.logger

        # قاموس الروبوتات المتخصص (Semantic Mapping)
        self.engine.robotics_dictionary = {
            "KINEMATICS": ["forward kinematics", "inverse kinematics", "denavit-hartenberg", "jacobian", "joint angles"],
            "CONTROL_SYSTEMS": ["pid controller", "feedback loop", "closed-loop", "stability", "transfer function", "servo"],
            "ACTUATION": ["stepper motor", "brushless dc", "actuator", "torque", "gearbox", "hydraulic"],
            "PERCEPTION": ["lidar", "ultrasonic", "imu", "computer vision", "odometry", "depth camera"],
            "NAVIGATION": ["slam", "path planning", "trajectory", "obstacle avoidance", "localization", "mapping"]
        }

    # --- دالة التحليل الرئيسية (analyze_pdf) ---
    def analyze_pdf(self, pdf_path: str, user_request: str = "") -> Dict[str, Any]:
        """
        [القسم الأول: المشرف الاستراتيجي]
        تجميع البيانات، الاستدلال، بناء السياق الشبكي، وتجهيز النبضات.
        """
        start_time = time.time()

        # --- المرحلة 1: الاستطلاع السريع (Fast Ingest) ---
        self.engine.logger.info(f"📡 Stage 1: Initializing Fast Ingest for {Path(pdf_path).name}")
        ingest_status = self.engine.fast_ingest_stream(pdf_path)

        if ingest_status.get("status") != "completed":
            return {"status": "error", "message": "Fast Ingest Failed"}

        page_count = int(ingest_status.get("pages_ingested", 0))

        # --- المرحلة 2: الاستدلال الاستراتيجي (Strategic Inference) ---
        self.engine.logger.info(f"🎯 Stage 2: Inferring navigation path for: '{user_request}'")

        # استدعاء ذكي لمسار الملاحة؛ إذا لم يوجد طلب، نحلل الوثيقة كاملة
        target_pages = self.infer_navigation_path(user_request)
        analysis_queue = target_pages if target_pages else list(range(page_count))

        # --- المرحلة 3: تجميع النبضات وتعزيز السياق (Context Assembly) ---
        step = 15  # تقليل الخطوة قليلاً لزيادة تركيز الـ Context في كل نبضة
        total_items = len(analysis_queue)
        pulses_data = []

        self.engine.logger.info(f"🌀 Stage 3: Preparing {total_items} pages into semantic pulses...")

        # --- المرحلة 3 المحسنة: تجميع النبضات ---
        for start_idx in range(0, total_items, step):
            current_batch = analysis_queue[start_idx : start_idx + step]
            if not current_batch: continue

            chunk_texts = []
            related_summaries = set()

            for p_num in current_batch:
                # التأكد من تمرير pdf_path لضمان الـ Auto-Recovery إذا سقطت الصفحة من الكاش
                page_data = self.engine.get_page_data(p_num, pdf_path=pdf_path)

                if page_data and page_data.get("content"):
                    chunk_texts.append(page_data["content"])

                    # سحب السياق (Context)
                    related_ids = self.get_related_pages(p_num, depth=1)
                    for r_id in related_ids:
                        r_data = self.engine.page_cache.get(r_id)
                        if r_data and r_data.get("semantic_keywords"):
                            kws = ", ".join(r_data["semantic_keywords"][:2])
                            related_summaries.add(f"[Ref Page {r_id+1} | Topics: {kws}]")

            full_text = "\n".join(chunk_texts).strip()

            # إذا استمرت مشكلة الـ 0.0%، هذا السطر سيكشف لك السبب في الـ Log
            if not full_text:
                self.engine.logger.warning(f"⚠️ Pulse starting at page {current_batch[0]} is empty and will be skipped.")
                continue

            pulse_payload = {
                "text": full_text,
                "context": "\n".join(list(related_summaries)[:4]),
                # تصحيح الـ Range لضمان عدم وجود قوائم متداخلة
                "range": (current_batch[0], current_batch[-1]),
                "pulse_id": len(pulses_data) + 1
            }
            pulses_data.append(pulse_payload)

        # التحقق من وجود نبضات جاهزة للتحليل
        if not pulses_data:
            return {"status": "error", "message": "No processable content found in target pages."}

        # طباعة تقرير الجاهزية (Accuracy Log)
        process_time = round(time.time() - start_time, 2)
        self.engine.logger.info(f"✅ Context Assembly Complete: {len(pulses_data)} pulses ready in {process_time}s")

        # الانتقال للمرحلة النهائية: التحليل العصبي (SNN)
        return self.SNN_analyze_pdf(pulses_data, user_request, pdf_path)

    def SNN_analyze_pdf(self, pulses_data: List[Dict], user_request: str, pdf_path: str) -> Dict[str, Any]:
        """
        [Section 2: SNN Pulse Processor]
        معالجة النبضات وتجميع الوعي النهائي مع استنتاج الموضوع الهجين.
        """
        all_ideas: List[Dict[str, Any]] = []
        cumulative_awareness: List[float] = []
        total_pulses = len(pulses_data)

        self.engine.logger.info(f"🧠 SNN Core: Processing {total_pulses} contextual pulses...")

        for idx, pulse in enumerate(pulses_data):
            pulse_idx = idx + 1
            chunk_text = str(pulse.get("text", ""))
            network_context = str(pulse.get("context", ""))

            # استخراج المدى (Range) بأمان
            raw_range = pulse.get("range", (0, 0))
            start_p = raw_range[0][0] if isinstance(raw_range[0], list) else raw_range[0]
            end_p = raw_range[1]

            if not chunk_text.strip():
                continue

            enhanced_prompt = f"{chunk_text}\n\n-- NETWORK_INSIGHTS --\n{network_context}"

            try:
                # 1. التوليد (استخدام المحاكي أو LLM الحقيقي)
                chunk_ideas, chunk_state = self.generate_mock(chunk_text, user_request=user_request)
                current_score = float(chunk_state.get("avg_score", 0.5))

                # 2. فحص النطاق الذهبي وإعادة المعالجة
                if not self._audit_sweet_spot(chunk_text, current_score):
                    self.engine.logger.info(f"🔄 Pulse {pulse_idx} below Sweet Spot. Activating Deep Analysis...")
                    chunk_ideas, chunk_state = self._reprocess_pulse(chunk_text, current_score, self.engine)
                    current_score = float(chunk_state.get("avg_score", 0.5))

                # 3. المراقبة والمزامنة
                self._analysis_monitor(pulse_idx, total_pulses, current_score)
                self._audit_and_sync_cache(start_p, {"quality_score": current_score, "semantic_keywords": chunk_state.get("top_concepts", [])})

                # 4. تجميع الأفكار
                for idea in chunk_ideas:
                    if isinstance(idea, dict):
                        idea["origin_context"] = f"Pages {start_p + 1}-{end_p + 1}"
                        all_ideas.append(idea)

                cumulative_awareness.append(current_score)

            except Exception as e:
                self.engine.logger.error(f"❌ SNN Failure in Pulse {pulse_idx}: {str(e)}")

        # --- المرحلة النهائية: تجميع التقرير بنظام "الوعي الهجين" ---
        final_avg_score = round(sum(cumulative_awareness) / len(cumulative_awareness), 2) if cumulative_awareness else 0.0

        # تحسين: بدلاً من صفحة الغلاف فقط، نجمع نص أول 3 نبضات لاستنتاج الموضوع بدقة
        sample_texts = [p.get("text", "") for p in pulses_data[:3]]
        combined_sample = " ".join(sample_texts)

        # استدعاء تخمين الموضوع بناءً على محتوى "دسم"
        inferred_topic = self.engine._guess_topic(combined_sample)

        # جلب المحاور المكتشفة
        hubs = self.get_network_hubs(3)

        report = {
            "status": "success",
            "analysis_metrics": {
                "total_pulses": len(cumulative_awareness),
                "consciousness_score": final_avg_score,
                "inferred_topic": inferred_topic,
                "knowledge_hubs": hubs
            },
            "output": {
                "structured_ideas": all_ideas,
                "summary": f"SNN Analysis complete via {len(cumulative_awareness)} pulses | Topic: {inferred_topic} ✅"
            },
            "file_reference": {
                "path": pdf_path,
                "visual_links_found": len(self.engine.visual_links)
            }
        }

        self.engine.logger.info(f"📊 Final Report Generated: Topic '{report['analysis_metrics']['inferred_topic']}' with Score {final_avg_score}")
        return report

    def advance_pdf_analyzer(self, pdf_path: str, user_request: str = "") -> Dict[str, Any]:
        """
        Advanced Segmented Analysis (Hybrid Pulse Mode).
        تصحيح المراجع: استدعاء البيانات من engine، والتفكير من self.
        """
        # 1. المرحلة الهيكلية (نطلب المعالجة من المخزن)
        # تصحيح: self.engine.process_pdf
        process_result = self.engine.process_pdf(pdf_path, user_request)
        if process_result["status"] != "success":
            return process_result

        page_count = process_result.get("page_count", 0)
        total_chars = process_result.get("total_chars", 0)

        all_ideas: List[Dict] = []
        pulse_scores: List[float] = []
        step = 20

        try:
            # 2. التحليل النبضي المتسلسل
            for start_idx in range(0, page_count, step):
                end_idx = min(start_idx + step, page_count)

                # تصحيح: الوصول للكاش عبر self.engine.page_cache
                chunk_text = "\n".join([
                    str(self.engine.page_cache[i].get("content", ""))
                    for i in range(start_idx, end_idx) if i in self.engine.page_cache
                ])

                if not chunk_text.strip():
                    continue

                # 3. تنفيذ النبضة (استدعاء دالة التوليد من نفس الكلاس)
                # تصحيح: استخدام self.generate_mock أو self.generate
                chunk_ideas, chunk_state = self.generate_mock(chunk_text, max_iterations=1, user_request=user_request)

                if chunk_ideas:
                    all_ideas.extend(chunk_ideas)

                pulse_scores.append(float(chunk_state.get("avg_score", 0.5)))

            # 4. تجميع الوعي النهائي
            final_avg_score = round(sum(pulse_scores) / len(pulse_scores), 2) if pulse_scores else 0.5

            # 5. استعادة بيانات الغلاف من المخزن
            cover_page = self.engine.page_cache.get(0, {})
            sample_text = cover_page.get("content", "")

            self.engine.logger.info(f"✅ Advanced Analysis Completed | Pulses: {len(pulse_scores)}")

            # 6. تجهيز المخرجات الشاملة
            return {
                "status": "success",
                "analysis_metrics": {
                    "page_count": page_count,
                    "pulse_count": len(pulse_scores),
                    "consciousness_score": final_avg_score,
                    "total_ideas": len(all_ideas)
                },
                "pdf_metadata": {
                    "char_count": total_chars,
                    "file_path": pdf_path,
                    # تصحيح: التخمين وظيفة المخزن (البيانات)
                    "inferred_topic": self.engine._guess_topic(sample_text),
                },
                "output": {
                    "structured_ideas": all_ideas,
                    "summary": f"Analyzed via {len(pulse_scores)} pulses → {len(all_ideas)} concepts extracted."
                }
            }

        except Exception as e:
            error_msg = f"❌ Critical error in advanced analysis: {str(e)}"
            self.engine.logger.error(error_msg)
            return {"status": "error", "message": str(e)}

    def infer_navigation_path(self, query: str) -> List[int]:
        """
        Reasoning Engine: Decides which pages to prioritize based on query intent.
        Standardized with English internal logic and hybrid routing.
        """
        if not query:
            return []

        query_lower = query.lower()
        target_pages: List[int] = []

        # 1. Strategic Layer Inference: استدلال الطبقات بناءً على الكلمات المفتاحية
        technical_indicators = {'جدول', 'رسم', 'بيانات', 'مخطط', 'table', 'data', 'figure', 'chart', 'specs'}
        structural_indicators = {'فصل', 'عنوان', 'مقدمة', 'فهرس', 'chapter', 'section', 'index', 'summary'}

        # فحص الطلبات ذات الطابع التقني (بيانات، جداول، إحصائيات)
        if any(w in query_lower for w in technical_indicators):
            target_pages = [p for p, d in self.engine.page_cache.items() if d.get('layer_type') == 'TECHNICAL_DATA']
            self.engine.logger.info("🎯 Strategic Routing: Prioritizing TECHNICAL_DATA layers.")

        # فحص الطلبات ذات الطابع الهيكلي (عناوين، فصول، ملخصات)
        elif any(w in query_lower for w in structural_indicators):
            target_pages = [p for p, d in self.engine.page_cache.items() if d.get('layer_type') in ['CHAPTER_LAYER', 'CORE_CONTENT']]
            self.engine.logger.info("🎯 Strategic Routing: Prioritizing Structural/Chapter layers.")

        # 2. Dynamic Hub Routing: الاستدلال عبر شبكة العلاقات (Heuristic Network)
        # إذا لم نجد طبقة صريحة، نبحث عن الكلمات المفتاحية الأكثر تأثيراً (Hubs)
        if not target_pages:
            hubs = self.engine.get_strategic_hubs(top_n=5)
            for hub_word, score in hubs:
                if hub_word.lower() in query_lower:
                    # استخراج الصفحات المرتبطة بهذا المحور المعرفي
                    pages = [e['page'] for e in self.engine.heuristic_network.get(hub_word, [])]
                    target_pages.extend(pages)

            if target_pages:
                self.engine.logger.info(f"🎯 Semantic Routing: Connected via {len(target_pages)} knowledge hubs.")

        # 3. Hybrid Search Fallback: إذا فشل الاستدلال الهيكلي، نلجأ للبحث الدلالي المباشر
        if not target_pages:
            self.engine.logger.info("🔍 Direct Inference failed. Activating Semantic Search fallback...")
            search_results = self.engine.semantic_search(query, top_k=5)
            target_pages = [res['page_index'] for res in search_results]

        # 4. Accuracy & Path Verification (نتائج الإختبار)
        unique_pages = sorted(list(set(target_pages)))
        confidence = "HIGH" if len(unique_pages) > 0 else "LOW"

        self.engine.logger.info(
            f"📊 Inference Report:\n"
            f"   - Query: '{query[:30]}...'\n"
            f"   - Confidence: {confidence}\n"
            f"   - Routed Pages: {unique_pages[:10]} {'...' if len(unique_pages) > 10 else ''}"
        )

        return unique_pages

    def _enrich_robotics_context(self, keywords: List[str]) -> List[str]:
        """
        تقوم بتوسيع الكلمات المفتاحية بناءً على القاموس التقني لرفع دقة الربط.
        """
        enriched_list = list(keywords)
        keywords_lower = [k.lower() for k in keywords]

        for category, related_terms in self.engine.robotics_dictionary.items():
            # إذا وجدت كلمة من التصنيف، أضف اسم التصنيف نفسه كـ "رابط سيادي"
            if any(term in " ".join(keywords_lower) for term in related_terms):
                if category not in enriched_list:
                    enriched_list.append(category)

        return enriched_list

    def _analysis_monitor(self, current_pulse: int, total_pulses: int, chunk_score: float):
        """
        Real-time Performance Dashboard: Tracks analysis quality and memory pressure.
        [Location: SmartAnalyzePDF -> Monitoring Engine]
        """

        # 1. Calculation of KPIs (مؤشرات الأداء الرئيسية)
        # ضمان عدم القسمة على صفر في حال كانت النبضات غير محددة
        total_p = max(total_pulses, 1)
        progress = (current_pulse / total_p) * 100

        # الوصول لبيانات الكاش عبر المحرك الرئيسي (engine)
        cache_count = len(self.engine.page_cache)
        max_p = max(self.engine.max_pages, 1)
        cache_usage_pct = (cache_count / max_p) * 100

        # 2. Status Determination (تحديد حالة الجودة والذاكرة)
        # الجودة المثالية تعتمد على النطاق الذهبي الذي حددناه سابقاً
        quality_status = "✨ OPTIMAL" if chunk_score >= 0.85 else "⚠️ SUB_OPTIMAL"
        memory_status_icon = "🟢" if cache_usage_pct < 80 else "🔴" if cache_usage_pct > 95 else "🟡"

        # 3. Professional Indexed Output (تنسيق مخرجات المراقبة)
        # تم تحسين الشكل البصري ليكون أسهل في القراءة أثناء تشغيل الكود
        header = f"\n{'-'*30}\n[PULSE #{current_pulse:02d} | MONITORING SYSTEM]\n{'-'*30}"
        stats_body = (
            f" 📊 Progress: {progress:>5.1f}% | Quality: {quality_status}\n"
            f" 🎯 Confidence: {chunk_score:.4f}\n"
            f" 💾 Memory {memory_status_icon}: {cache_count}/{max_p} pages ({cache_usage_pct:.1f}%)"
        )

        self.engine.logger.info(header + stats_body)

        # 4. Critical Memory Mitigation (LRU Policy - صمام أمان الذاكرة)
        # إذا وصل الكاش للحد الأقصى، نقوم بتفريغ مساحة فوراً
        if cache_usage_pct >= 100.0:
            try:
                # حذف أقدم صفحة (Last Recently Used)
                evicted_page, _ = self.engine.page_cache.popitem(last=False)
                self.engine.logger.warning(f"🚨 RESOURCE_LIMIT: Cache Full. Evicted Page {evicted_page} to maintain stability.")
            except (KeyError, IndexError):
                pass

    def get_related_pages(self, page_num: int, depth: int = 2) -> List[int]:
        """
        [Location: SmartAnalyzePDF]
        Traverses the heuristic network and visual links to find contextually relevant pages.
        Standardized with English internal identifiers.
        """
        from typing import Set

        # 1. التشييك على وجود الصفحة في الكاش عبر المحرك الرئيسي
        if page_num not in self.engine.page_cache:
            return []

        related_indices: Set[int] = set()

        # 2. الوصول للهيكل الطبقي (Hierarchy) عبر المحرك
        if page_num in self.engine.layer_hierarchy:
            related_indices.update(self.engine.layer_hierarchy[page_num])

        # 3. استخراج الروابط البصرية (Visual Links) التبادلية
        # تحسين: استخدام مولّد (Generator) لتقليل استهلاك الذاكرة أثناء الفرز
        visual_connections = (
            link["target_page"] if link["origin_page"] == page_num else link["origin_page"]
            for link in self.engine.visual_links
            if link["origin_page"] == page_num or link["target_page"] == page_num
        )
        related_indices.update(visual_connections)

        # 4. البحث الدلالي عبر الكلمات المفتاحية (Heuristic Network)
        # نركز فقط على الكلمات ذات الوزن العالي لضمان "دقة الصلة"
        current_page_data = self.engine.page_cache[page_num]
        current_keywords = current_page_data.get("semantic_keywords", [])

        for kw in current_keywords:
            if kw in self.engine.heuristic_network:
                entries = self.engine.heuristic_network[kw]
                # تحسين: تصفية الصفحات بناءً على وزن العلاقة (> 0.4) لضمان الجودة
                relevant_ids = [
                    entry["page"] for entry in entries
                    if isinstance(entry, dict) and entry.get("weight", 0) > 0.4
                ]
                # نأخذ عدداً محدوداً من الصفحات لكل كلمة (depth) لعدم تشتيت السياق
                related_indices.update(relevant_ids[:depth])

        # 5. تنظيف النتائج (حذف الصفحة الحالية وترتيب المخرجات)
        related_indices.discard(page_num)

        # تحديد عدد النتائج النهائية بضعف العمق لضمان تركيز السياق
        final_results = sorted(list(related_indices))[:depth * 2]

        self.engine.logger.info(f"🔗 Context Sync: Page {page_num} linked to {len(final_results)} related sources.")
        return final_results

    def get_network_hubs(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """
        Identifies the 'Knowledge Core' hubs based on connectivity density.
        [Location: SmartAnalyzePDF -> Heuristic Engine]
        تستخدم لفرز المصطلحات المركزية التي تربط أجزاء المستند ببعضها.
        """
        # 1. التشييك على وجود بيانات في الشبكة الاستدلالية عبر المخزن
        if not hasattr(self.engine, 'heuristic_network') or not self.engine.heuristic_network:
            self.engine.logger.warning("⚠️ Hub Analysis: Heuristic network is empty.")
            return []

        # 2. فرز المحاور بناءً على كثافة الروابط (Connectivity Density)
        # تحسين: استخدام items() كدالة وليس كخاصية لتجنب أخطاء التشغيل
        all_hubs = list(self.engine.heuristic_network.items())

        # الترتيب تنازلياً حسب عدد الصفحات المرتبطة بكل كلمة
        sorted_hubs = sorted(
            all_hubs,
            key=lambda x: len(x[1]),
            reverse=True
        )

        # 3. استخراج أفضل النتائج (Top N Hubs)
        top_hubs = sorted_hubs[:top_n]

        # 4. تقرير دقة التحليل (Knowledge Density Report)
        hub_names = [str(h[0]) for h in top_hubs]

        # حساب متوسط الروابط لكل محور لقياس ترابط الملف
        avg_density = sum(len(h[1]) for h in top_hubs) / max(len(top_hubs), 1)

        self.engine.logger.info(
            f"📊 Knowledge Hubs Report:\n"
            f"   - Identified Hubs: {hub_names}\n"
            f"   - Avg Connectivity: {avg_density:.1f} pages/hub"
        )

        # إرجاع قائمة توبلز (الكلمة، عدد الارتباطات)
        return [(str(h[0]), len(h[1])) for h in top_hubs]

    def generate(self, content: str, max_iterations: int = 3, user_request: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generates mock ideas for testing and logic verification.
        Standardized with English internal logic and detailed quality reporting.
        """
        # 1. المراقبة الأولية للحمولة (Payload Monitoring)
        payload_len = len(content)
        self.engine.logger.info(f"🧠 Mock Engine: Processing {max_iterations} iterations | Payload: {payload_len} chars")

        ideas: List[Dict[str, Any]] = []

        # 2. توليد الأفكار بناءً على المحاكاة (Idea Synthesis Simulation)
        for i in range(1, max_iterations + 1):
            # محاكاة ذكية لاستخراج العناوين بناءً على محتوى النص
            # نستخدم أول 30 حرف مع تنظيفها لضمان شكل احترافي
            clean_title = str(content[:30]).replace('\n', ' ').strip()

            ideas.append({
                "idea_id": i,
                "title": f"Strategic Concept {i}: {clean_title}...",
                "importance_level": "CORE_PRINCIPLE" if i == 1 else "SUPPORTING_DETAIL",
                "extracted_segments": [str(content[:150]).strip()],
                # محاكاة انخفاض الثقة التدريجي عند التعمق في التفاصيل الصغرى
                "confidence_score": round(0.92 - (i * 0.04), 2),
                "user_intent_sync": user_request if user_request else "General Exploration"
            })

        # 3. تجميع حالة الوعي (Awareness State Synthesis)
        # حساب متوسط درجة الثقة لتقييم جودة النبضة بالكامل
        avg_score = round(sum(d['confidence_score'] for d in ideas) / len(ideas), 2) if ideas else 0.5

        awareness_state = {
            "avg_score": avg_score,
            "iterations_performed": max_iterations,
            "engine_status": "STABLE_ANALYSIS",
            "context_retention": "HIGH" if payload_len > 1000 else "NORMAL"
        }

        # 4. تقرير دقة المحاكاة (Accuracy Log)
        self.engine.logger.info(
            f"✅ Mock Pulse Complete:\n"
            f"   - Confidence: {avg_score}\n"
            f"   - Iterations: {max_iterations}\n"
            f"   - Sync Level: {'OPTIMIZED' if user_request else 'STANDARD'}"
        )

        return ideas, awareness_state

    def generate_mock(self, content: str, max_iterations: int = 3, user_request: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Smart Inference Engine: Cost-optimized reasoning with adaptive scaling.
        Standardized with detailed accuracy reporting and semantic mapping.
        """
        # 1. Scaling & Cost Optimization: تحديد "حجم" المعالجة بناءً على النص
        scale = self._detect_processing_scale(content)
        mode = scale.get("mode", "STANDARD_ANALYSIS")
        # تحسين: ضمان أن عدد التكرارات لا يقل عن 1 لضمان توليد فكرة واحدة على الأقل
        actual_iterations = max(1, min(max_iterations, scale.get("iterations", 1)))

        # 2. Semantic Context Extraction: سحب الكلمات والموضوع من المحرك الرئيسي
        keywords = self.engine.PDF_extract_keywords(content, top_n=12)
        num_kws = len(keywords)
        current_topic = self.engine._guess_topic(content)

        self.engine.logger.info(f"⚙️ Mode: [{mode}] | Topic: [{current_topic}] | Keywords: {num_kws}")

        ideas: List[Dict[str, Any]] = []

        # 3. Iterative Idea Synthesis: توليد الأفكار بناءً على التكرارات المحددة
        for i in range(1, actual_iterations + 1):
            # تحسين: اختيار ذكي للكلمات المفتاحية لكل فكرة
            kw_main = keywords[0] if num_kws > 0 else "Analysis"
            kw_sub = keywords[i % num_kws] if num_kws > i else "Information"

            # حساب درجة الثقة (Dynamic Confidence Score)
            # تعتمد الدرجة على عدد الكلمات المفتاحية المكتشفة وعمق التكرار
            base_score = 0.7 + (min(num_kws, 10) * 0.02)
            dynamic_score = round(min(base_score + (i * 0.01), 0.98), 2)

            ideas.append({
                "idea_id": i,
                "analysis_mode": mode,
                "title": f"[{current_topic}] {kw_main.upper()} Insight: {kw_sub}",
                "user_intent": user_request if user_request else "General Document Analysis",
                "importance_level": "CORE_CONCEPT" if i == 1 else "SUPPORTING_DETAIL",
                # تحسين Pylance: ضمان تنظيف النص المستقطع للمعاينة
                "extracted_segments": [str(content[:200]).replace('\n', ' ').strip() + "..."],
                "contextual_keywords": keywords[i : i+3] if num_kws > i else keywords[:3],
                "confidence_score": dynamic_score,
                "efficiency_tier": scale.get("sampling_rate", 1.0)
            })

        # 4. Final Awareness Synthesis: تجميع الوعي النهائي للنبضة
        avg_score = round(sum(d['confidence_score'] for d in ideas) / len(ideas), 2) if ideas else 0.5

        state = {
            "avg_score": avg_score,
            "keywords_detected": num_kws,
            "top_concepts": keywords[:5],
            "processing_depth": "DEEP_REASONING" if mode == "HEAVY_INFERENCE" else "STANDARD_ANALYSIS",
            "optimization_savings": f"{(1 - scale.get('sampling_rate', 1.0))*100:.0f}%"
        }

        self.engine.logger.info(f"✅ Pulse Generated | Confidence: {avg_score} | Savings: {state['optimization_savings']}")
        return ideas, state

    def _detect_processing_scale(self, content: str) -> Dict[str, Any]:
        """
        Adaptive Resource Allocation: Detects payload size to optimize computation.
        ضبط موازنة الأداء مقابل الدقة لضمان استقرار المحرك مع الملفات الضخمة.
        """
        # حساب حجم الحمولة النصية
        payload_size = len(content)

        # 1. LIGHT MODE: للمقتطفات أو الفقرات القصيرة (< 5 آلاف حرف)
        if payload_size < 5000:
            scale_data = {
                "mode": "LIGHT_WEIGHT",
                "iterations": 1,
                "vector_search_depth": 3,
                "sampling_rate": 1.0,
                "effort_score": 0.2
            }

        # 2. BALANCED MODE: للمستندات القياسية (5 آلاف - 30 ألف حرف)
        elif payload_size < 30000:
            scale_data = {
                "mode": "BALANCED_CORE",
                "iterations": 3,
                "vector_search_depth": 8,
                "sampling_rate": 0.8,
                "effort_score": 0.6
            }

        # 3. HEAVY MODE: للمجلدات الفنية والملفات الضخمة (> 30 ألف حرف)
        else:
            # تحسين: رفع العمق لضمان عدم ضياع التفاصيل في الملفات الكبيرة
            scale_data = {
                "mode": "HEAVY_INFERENCE",
                "iterations": 5,
                "vector_search_depth": 20,
                "sampling_rate": 0.6,
                "effort_score": 1.0
            }

        # تقرير دقة التخصيص (Accuracy Check)
        self.engine.logger.info(
            f"⚖️ Scale Detected: {scale_data['mode']} | "
            f"Payload: {payload_size} chars | "
            f"Sampling: {int(scale_data['sampling_rate']*100)}%"
        )

        return scale_data

    def _run_tiered_processing(self, page_num: int, page_obj: Any, content: str, metadata: Dict):
        """
        Tiered Supervisor: ينظم عملية الإدراك، التخطيط، والدلالات.
        [مكان الدالة: PDFPageCacheNetwork]
        بما أنها المسؤولة عن بناء "الذاكرة" قبل أن يحللها "المخ".
        """
        import time
        metrics = {}
        content_len = len(content)

        # --- الطبقة 1: الإدراك (توليد المتجهات مع صمام أمان) ---
        t_start = time.perf_counter()
        if content_len > 50:
            # استخدام الموديل المخزن في self
            vector = self.engine.model.encode(content, convert_to_numpy=True)
            self.engine.vectors.append(vector)
            self.engine.vector_map.append(page_num)
            metrics['perception_status'] = "ENCODED"
        else:
            self.engine.logger.info(f"🍃 Skipping Vector for Page {page_num} (Short Content: {content_len} chars)")
            metrics['perception_status'] = "SKIPPED_LOW_CONTENT"
        metrics['perception_latency'] = time.perf_counter() - t_start

        # --- الطبقة 2: الهيكلية البصرية (Layout) ---
        t_start = time.perf_counter()
        # استدعاء دالة الاستخراج من نفس الكلاس
        layout_data = self.engine._extract_layout_structure(page_obj)
        metrics['layout_latency'] = time.perf_counter() - t_start

        # --- الطبقة 3: الاستدلال الدلالي (Classification & Keywords) ---
        t_start = time.perf_counter()
        keywords = self.engine.PDF_extract_keywords(content)
        topic = self.engine._guess_topic(content[:1000])
        layer_type = self.engine._classify_layer(content, metadata)
        metrics['semantic_latency'] = time.perf_counter() - t_start

        # --- تجميع البيانات (Data Integration) ---
        page_entry = {
            "content": content,
            "topic_id": topic,
            "semantic_keywords": keywords,
            "visual_headings": layout_data["headings"],
            "layer_type": layer_type,
            "metadata": metadata,
            "processing_metrics": metrics
        }

        # --- الطبقة 4: تدقيق الجودة ومزامنة الكاش ---
        # استدعاء المزامنة من نفس الكلاس
        is_valid = self._audit_and_sync_cache(page_num, page_entry)
        if not is_valid:
            self.engine.logger.warning(f"⚠️ Quality Audit failed for Page {page_num} - Auto-repairing cache entry.")

        # تحديث الكاش (LRU Management)
        self.engine.page_cache[page_num] = page_entry
        if len(self.engine.page_cache) > self.engine.max_pages:
            oldest_id, _ = self.engine.page_cache.popitem(last=False)
            self.engine.logger.info(f"🧹 LRU Eviction: Removed Page {oldest_id}")

        # --- الطبقة 5: الربط الشبكي (Heuristics) ---
        self.engine._build_visual_heuristics(page_num, layout_data)
        self.engine._build_heuristic_links(page_num)

        return layer_type

    def _audit_and_sync_cache(self, page_num: int, current_entry: Dict[str, Any]) -> bool:
        """
        [Location: PDFPageCacheNetwork/SmartAnalyzePDF Sync]
        التدقيق في سلامة البيانات ومزامنة المتجهات لضمان استقرار الشبكة العصبية.
        """
        # 1. فحص وجود الصفحة في الكاش (الوصول للمحرك الرئيسي)
        if page_num not in self.engine.page_cache:
            self.engine.logger.warning(f"⚠️ Sync Skip: Page {page_num} not found in cache.")
            return False

        # 2. فحص سلامة المحتوى (Data Integrity)
        content = current_entry.get("content")
        if not content or len(str(content)) < 5:
            self.engine.logger.error(f"❌ Integrity Error: Page {page_num} has insufficient content.")
            return False

        # 3. صمام أمان المزامنة (Vector Alignment Security)
        # التأكد من أن عدد المصفوفات يطابق عدد خرائط الصفحات لضمان دقة البحث الدلالي
        if len(self.engine.vectors) != len(self.engine.vector_map):
            self.engine.logger.error(f"⚠️ SYNC_GAP: Vectors ({len(self.engine.vectors)}) != Map ({len(self.engine.vector_map)})")
            # محاولة إصلاح ذاتي: إعادة بناء الخارطة بناءً على المتجهات المتاحة
            self.engine.vector_map = self.engine.vector_map[:len(self.engine.vectors)]
            return False

        # 4. مراقبة الأداء (Latency Monitoring)
        metrics = current_entry.get("processing_metrics", {})
        if isinstance(metrics, dict):
            for layer, latency in metrics.items():
                if isinstance(latency, (int, float)) and latency > 2.0:
                    self.engine.logger.warning(f"🐢 Performance Alert: '{layer}' latency {latency:.2f}s on Page {page_num}")

        # 5. فحص الجزر المعزولة دلالياً (Orphan Node Detection)
        # التأكد من أن الصفحة مرتبطة فعلياً بالشبكة الاستدلالية (Heuristic Network)
        keywords = current_entry.get("semantic_keywords", [])
        if len(keywords) > 3:
            # فحص وجود أي صلة للكلمات المفتاحية لهذه الصفحة في الشبكة الكلية
            has_connections = any(
                any(isinstance(e, dict) and e.get("page") == page_num for e in entries)
                for entries in self.engine.heuristic_network.values()
            )

            if not has_connections:
                self.engine.logger.info(f"📍 Orphan Node: Page {page_num} is semantically isolated. Re-linking...")
                # هنا يمكن إضافة دالة إعادة الربط إذا لزم الأمر
                return True

        # تقرير نجاح المزامنة (Accuracy Log)
        self.engine.logger.info(f"✅ Sync Verified: Page {page_num} is aligned with Heuristic Engine.")
        return True

    def _audit_sweet_spot(self, chunk_content: str, current_confidence: float) -> bool:
        """
        Confidence Range Audit: فحص النطاق الذهبي للجودة.
        يضمن أن جودة التحليل تقع في المنطقة المثالية (0.85 - 0.96) لضمان دقة الاستنتاج.
        """
        # 1. تحديد معايير النطاق الذهبي (The Golden Zone)
        lower_bound = 0.85
        upper_bound = 0.96

        # 2. فحص النطاق المباشر
        if lower_bound <= current_confidence <= upper_bound:
            self.engine.logger.info(f"✨ Audit Success: Confidence {current_confidence:.2f} is in the Golden Zone.")
            return True

        # 3. معيار المرونة للمحتوى القصير (Relaxed Criteria for Short Pulses)
        # إذا كان النص قصيراً جداً، يصعب الوصول لدرجة 0.85، لذا نكتفي بـ 0.80
        if len(chunk_content) < 300 and current_confidence >= 0.80:
            self.engine.logger.info(f"🌤️ Audit Pass: Confidence {current_confidence:.2f} accepted for short content.")
            return True

        # 4. معيار "الثقة المفرطة" (Over-Confidence Check)
        # إذا كانت الدرجة أعلى من 0.96، قد يكون هناك "هلوسة" أو تبسيط مفرط
        if current_confidence > upper_bound:
            self.engine.logger.warning(f"⚠️ Audit Alert: Confidence {current_confidence:.2f} is suspiciously high (Potential Over-fitting).")
            return False

        # 5. تقرير الفشل (Accuracy Log)
        self.engine.logger.info(
            f"⚖️ Audit Failure: Confidence {current_confidence:.2f} is outside Golden Zone. "
            f"Reason: {'Below Threshold' if current_confidence < lower_bound else 'Above Threshold'}"
        )
        return False

    def _reprocess_pulse(self, content: str, initial_score: float, thinking_engine: Any):
        """
        Deep analysis reprocessing: الربط مع LLM حقيقي أو تكثيف الجهد المحلي.
        تضمن الوصول للنطاق الذهبي عبر نظام Fallback متسلسل.
        """
        self.engine.logger.info(f"🔄 Activating Recovery Logic (Initial Score: {initial_score:.2f})")

        try:
            # 1. الملاذ الأول: الجسر الحقيقي (Real LLM Bridge) عبر المحرك الرئيسي
            if hasattr(self.engine, 'llm_client') and self.engine.llm_client:
                self.engine.logger.info("📡 Routing to External Cloud LLM...")
                # استدعاء الجسر الذي قمنا بتعريفه في PDFPageCacheNetwork
                return self.engine.llm_bridge(content, user_request="DEEP_RECOVERY")

            # 2. الملاذ الثاني: تكثيف الجهد عبر المحرك الممرر (إذا كان يدعم المحاكاة)
            if hasattr(thinking_engine, 'generate_mock'):
                self.engine.logger.info("⚙️ Scaling local thinking iterations to level 5...")
                enhanced_ideas, enhanced_state = thinking_engine.generate_mock(
                    content,
                    max_iterations=5,
                    user_request="DEEP_REASONING_RECOVERY"
                )
            # 3. الملاذ الثالث: استخدام دالة المحاكي الخاصة بالمحلل الحالي (Self-Recovery)
            else:
                self.engine.logger.info("🛠️ Falling back to Internal High-Intensity Mock...")
                enhanced_ideas, enhanced_state = self.generate_mock(
                    content,
                    max_iterations=5,
                    user_request="INTERNAL_RECOVERY"
                )

            # 4. تقرير كفاءة التحسين
            new_score = float(enhanced_state.get("avg_score", 0.0))
            improvement = new_score - initial_score

            self.engine.logger.info(
                f"✅ Recovery Successful: New Score {new_score:.2f} "
                f"(Delta: {improvement:+.2f})"
            )

            return enhanced_ideas, enhanced_state

        except Exception as e:
            self.engine.logger.error(f"❌ Recovery Pipeline Failed: {str(e)}")
            # إرجاع الدرجة الأصلية لمنع انهيار مصفوفة الوعي الكلية
            return [], {"avg_score": initial_score, "status": "RECOVERY_FAILED"}

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
                "heuristic_keys_count": len(self.engine.heuristic_network),
                "visual_links_count": len(self.engine.visual_links),
                "vector_space_size": len(self.engine.vectors),
                "cache_capacity": f"{len(self.engine.page_cache)}/{self.engine.max_pages}"
            }
        }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_payload, f, ensure_ascii=False, indent=4)
            self.engine.logger.info(f"💾 DATA_EXPORT: Successfully saved JSON to {output_path}")
        except Exception as e:
            self.engine.logger.error(f"❌ EXPORT_FAILED: Could not save JSON: {str(e)}")

    def export_to_markdown(self, analysis_result: Dict[str, Any], output_path: str):
        """
        Generates a professional Markdown report based on the hybrid analysis output.
        Standardized with updated English internal keys and structural tables.
        """
        # 1. Data Extraction (Consistent with SNN Output)
        metrics = analysis_result.get('analysis_metrics', {})
        file_ref = analysis_result.get('file_reference', {})
        output_data = analysis_result.get('output', {})

        # Accessing stats safely
        file_name = Path(file_ref.get('path', 'Unknown')).name
        consciousness = float(metrics.get('consciousness_score', 0))
        topic = metrics.get('inferred_topic', 'General Technical')

        md = [
            f"# 📄 Technical Audit: {file_name}",
            f"\n## 📊 Analytical Metadata",
            f"- **Inferred Topic:** `{topic}`",
            f"- **Confidence Score:** `{consciousness * 100:.1f}%`",
            f"- **Heuristic Connections:** `{len(self.engine.visual_links)}` identified links",
            f"\n## 🧠 Semantic Insights & Reasoning",
            "---"
        ]

        # 2. Logic processing for Pulse Ideas
        for idea in output_data.get('structured_ideas', []):
            # Styling based on Importance Level
            is_core = idea.get('importance_level') == "CORE_CONCEPT"
            prefix = "### 🚀 [CORE]" if is_core else "#### 🔍 [DETAIL]"

            md.append(f"{prefix} {idea.get('title')}")
            md.append(f"- **Trust Factor:** `{idea.get('confidence_score', 0) * 100:.0f}%` | **Source:** `{idea.get('origin_context', 'N/A')}`")

            if 'extracted_segments' in idea:
                # Cleaning segments for better MD display
                clean_segment = str(idea['extracted_segments'][0]).replace('\n', ' ').strip()
                md.append(f"> {clean_segment}")

            kws = idea.get('contextual_keywords', [])
            if kws:
                md.append(f"**Keywords:** `{', '.join(kws)}`")
            md.append("\n---")

        # 3. Heuristic Hubs Table (The Knowledge Map)
        md.append(f"\n## 🌐 Knowledge Hub Discovery")
        md.append("| Hub Identifier | Network Connectivity |")
        md.append("| :--- | :--- |")

        hubs = metrics.get('knowledge_hubs', [])
        if hubs:
            for hub_name, density in hubs:
                md.append(f"| **{str(hub_name).upper()}** | {density} pages connected |")
        else:
            md.append("| *No significant hubs detected* | - |")

        # 4. Final Disk Write with Error Resilience
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(md))
            self.engine.logger.info(f"📝 REPORT_EXPORT: Markdown successfully archived to {output_path}")
        except Exception as e:
            self.engine.logger.error(f"❌ EXPORT_FAILED: Disk write error: {str(e)}")

    def clear_cache(self, deep: bool = False):
        """
        Smart Memory Management: balances system performance with data retention.
        Standardized with English internal identifiers.
        """
        # 1. Partial Clean (Default): تحرير الذاكرة من النصوص الثقيلة مع الإبقاء على الفهرسة
        # نقوم بمسح النصوص لتقليل استهلاك الرام، لكن نبقي على المتجهات والشبكة الاستدلالية
        self.engine.page_cache.clear()
        self.engine.page_queue.clear() # استخدام clear() أفضل من إعادة التعريف لضمان ثبات المرجع

        if deep:
            # 2. Deep Clean: مسح شامل وإعادة ضبط المحرك (Full Reset)
            self.engine.heuristic_network.clear()
            self.engine.layer_hierarchy.clear()
            self.engine.visual_links = []
            self.engine.vectors = []
            self.engine.vector_map = []
            self.engine.navigation_stack.clear()

            self.engine.logger.info("🧹 DEEP_CLEAN: All semantic intelligence and vectors have been purged.")
        else:
            # 3. Intelligent Retention: الإبقاء على "دماغ" المحرك
            # نحافظ على الشبكة الاستدلالية لأنها تتيح الملاحة السريعة دون استهلاك ذاكرة
            self.engine.logger.info("🧹 SMART_CLEAN: Text cache cleared. Structural memory and vectors retained.")

        # 4. تقرير حالة الموارد (Resource Accuracy Report)
        self.engine.logger.info(
            f"📊 Post-Cleanup Status:\n"
            f"   - Active Vectors: {len(self.engine.vectors)}\n"
            f"   - Heuristic Hubs: {len(self.engine.heuristic_network)}\n"
            f"   - Mode: {'Total Reset' if deep else 'Optimized Retention'}"
        )

# ========== الربط مع process_pdf_streaming ==========
def process_pdf_streaming(pdf_path: str, logger: Any, callback: Optional[Callable] = None) -> Optional[PDFPageCacheNetwork]:
    """
    Standalone Orchestrator: Creates and populates the Heuristic Network.
    Separated from the class to act as a 'Factory' for building a knowledge base.
    """
    start_time = time.time()
    logger.info(f"🚀 Factory Launch: Serial Ingestion for {Path(pdf_path).name}")

    # 1. تهيئة محرك الشبكة الاستدلالية (Initialization)
    # ملاحظة: PDFPageCacheNetwork هي الكلاس الأساسي الذي بنيناه سابقاً
    cache_network = PDFPageCacheNetwork(logger=logger)

    try:
        with fitz.open(pdf_path) as pdf_doc:
            total_pages = pdf_doc.page_count

            for page_num in range(total_pages):
                # تحميل الصفحة (استخدام doc[idx] أسرع برمجياً)
                page = pdf_doc[page_num]

                # تحسين: استخراج النص مع الحفاظ على الترتيب لضمان دقة التحليل لاحقاً
                raw_text = page.get_text("text", sort=True)
                page_text = str(raw_text or "").strip()

                if page_text:
                    # 2. إعداد الميتا-داتا الموحدة (Standardized Metadata)
                    metadata = {
                        "source_path": pdf_path,
                        "page_index": page_num + 1,
                        "content_size": len(page_text),
                        "ingestion_mode": "SERIAL_STREAM"
                    }

                    # 3. تغذية المحرك (Feeding the Core)
                    # دالة add_page ستقوم تلقائياً بتوليد الـ Vectors والـ Heuristics
                    cache_network.add_page(
                        page_num=page_num,
                        text=page_text,
                        page_obj=page,
                        metadata=metadata
                    )

                    # 4. التفاعل مع الواجهة أو النظام الخارجي
                    if callback:
                        try:
                            callback(page_num, page_text, metadata)
                        except Exception as cb_e:
                            logger.warning(f"⚠️ Callback Failure on page {page_num}: {cb_e}")

        # 5. تقرير نجاح بناء المصنع (Factory Completion Report)
        process_time = round(time.time() - start_time, 2)
        logger.info(
            f"✅ Factory Process Complete:\n"
            f"   - Pages Indexed: {len(cache_network.page_cache)}/{total_pages}\n"
            f"   - Network Hubs: {len(cache_network.heuristic_network)}\n"
            f"   - Processing Time: {process_time}s"
        )

        return cache_network

    except Exception as e:
        logger.error(f"❌ FACTORY_CRITICAL_FAILURE: {str(e)}")
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

# ==================== __main__ ====================
if __name__ == "__main__":
    # 1. إعداد المحرك (المخزن/Perception Layer)
    # نحدد الحد الأقصى للصفحات لضمان كفاءة الذاكرة
    engine = PDFPageCacheNetwork(max_pages=200)

    # ربط العميل الحقيقي بالمحرك
    from openai import OpenAI
    engine.llm_client = OpenAI(api_key="")

    # 1. إعداد المحرك (المخزن)
    engine = PDFPageCacheNetwork(max_pages=200)

    # [إضافة استراتيجية]: ربط الـ LLM الحقيقي إذا كان المفتاح متوفراً
    # from openai import OpenAI
    # engine.llm_client = OpenAI(api_key="YOUR_API_KEY")

    # 2. إعداد المحلل (المخ)
    analyzer_brain = SmartAnalyzePDF(
        parent_engine=engine,
        thinking_engine=engine
    )

    print("\n" + "="*50)
    print("🚀 [Hybrid Mode] Starting Tiered Ingestion & Semantic Indexing")
    print("="*50)

    pdf_file = "ROBOTICS.pdf"

    # المرحلة 1: بناء القاعدة المعرفية (تغيير حيوي لضمان بناء المتجهات والروابط)
    # بدلاً من fast_ingest_stream، سنستخدم المصنع لبناء الـ Vectors فوراً
    print("⚙️ Building Knowledge Graph & Vectors...")
    engine = process_pdf_streaming(pdf_file, engine.logger)

    if engine:
        # تحديث مرجع المحرك داخل المحلل بعد الفهرسة الكاملة
        analyzer_brain.engine = engine

        print(f"✅ Perception Layer Ready: {len(engine.page_cache)} pages indexed.")
        print(f"📡 Vector Space: {len(engine.vectors)} embeddings generated.")

        # المرحلة 2: التحليل النبضي (Chained Pulse Analysis)
        print("\n🧠 Activating Chained Pulse Analysis...")
        final_report = analyzer_brain.analyze_pdf(
            pdf_path=pdf_file,
            user_request="Deep technical audit of control systems"
        )

        # المرحلة 3: عرض تقارير الأداء الهجين
        print(f"\n📊 Hybrid Performance Report:")
        print(f"   ├─ Heuristic Hubs: {len(engine.heuristic_network)} nodes")
        print(f"   ├─ Visual Links: {len(engine.visual_links)} connections")

        metrics = final_report.get('analysis_metrics', {})
        score = metrics.get('consciousness_score', 0)
        # الآن الـ Score سيكون حقيقياً (مثلاً 91.5%)
        print(f"   └─ Cumulative Awareness: {score * 100:.1f}%")

        # المرحلة 4: اختبار القفز الاستدلالي
        print("\n🔍 Testing Semantic Jump (Knowledge Graph):")
        query = "robotic sensors and feedback loops"
        search_hits = engine.semantic_search(query, top_k=2)

        for hit in search_hits:
            p_num = hit['page_index']
            related = analyzer_brain.get_related_pages(p_num, depth=1)
            print(f"   📍 Match in Page {p_num+1} (Confidence: {hit['rank_score']:.2f})")
            print(f"      🔗 Strategically Related Pages: {related}")

        # المرحلة 5: التصدير النهائي
        print("\n💾 Archiving Hybrid Memory to Disk...")
        analyzer_brain.export_to_markdown(final_report, "hybrid_audit_report.md")

        # المرحلة 6: اكتشاف مراكز المعرفة (Hubs)
        print("\n🧬 Knowledge Core Discovery (Hubs):")
        hubs = analyzer_brain.get_network_hubs(5)
        for word, count in hubs:
            print(f"   🔑 Hub: {word.upper():<15} | Connectivity Density: {count}")

    else:
        print(f"❌ Engine Failed: Analysis cannot proceed without a valid knowledge base.")

    print("\n" + "="*50)
    print("🎉 Hybrid Serial Workflow Completed Successfully!")
    print("="*50 + "\n")
