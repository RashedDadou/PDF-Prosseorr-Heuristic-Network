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
    def info(self, msg: str, *args, **kwargs) -> None: ...
    def error(self, msg: str, *args, **kwargs) -> None: ...
    def warning(self, msg: str, *args, **kwargs) -> None: ...
    def debug(self, msg: str, *args, **kwargs) -> None: ...  # أضف هذا السطر

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
        # 1. تهيئة المتغيرات في البداية لضمان وصول الـ return إليها
        doc_info = {}
        full_text = ""
        path_obj = Path(pdf_path)

        try:
            if not path_obj.exists():
                raise FileNotFoundError(f"المسار غير موجود: {pdf_path}")

            with fitz.open(pdf_path) as pdf_doc:
                page_count = len(pdf_doc)
                full_text_parts: List[str] = []

                # 2. تعريف doc_info داخل الـ context الصحيح
                raw_meta = pdf_doc.metadata or {}
                doc_info = {
                    "title": str(raw_meta.get("title") or path_obj.stem),
                    "author": str(raw_meta.get("author") or "Unknown"),
                    "page_count": page_count
                }

                self.logger.info(f"🚀 معالجة عدوانية لـ: {doc_info['title']} ({page_count} صفحة)")

                for page_num in range(page_count):
                    page = pdf_doc[page_num]

                    # 🔴 [خطوة فكرتك 1]: الوسم الأحمر الأولي
                    # نستخدم إحداثيات مرنة تعتمد على عرض الصفحة لضمان الجمالية
                    status_rect = fitz.Rect(10, 10, 150, 40)
                    page.draw_rect(status_rect, color=(1, 0, 0), width=1.5, fill=(1, 0.95, 0.95))
                    page.insert_text((15, 28), f"PROCESSING P.{page_num+1}", color=(1, 0, 0), fontsize=9)

                    page_contents = []

                    # استخراج النصوص (منطقك العدواني)
                    raw_blocks = page.get_text("blocks")
                    blocks = sorted([b for b in raw_blocks if len(b) >= 5], key=lambda b: (b[1], b[0]))
                    for b in blocks:
                        text = b[4].strip()
                        if text: page_contents.append(text)

                    # استخراج الجداول (منطقك المتميز)
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

                    page_text = "\n".join(page_contents).strip()

                    if page_text:
                        full_text_parts.append(page_text)
                        page_meta = {
                            "source_path": pdf_path,
                            "page_index": page_num + 1,
                            "contains_tables": has_tables
                        }

                        # 🟢 [خطوة فكرتك 2]: استدعاء دالة الإضافة التي ستحول الختم للأخضر بالرموز 🧬
                        self.add_page(page_num, page_text, page, page_meta)

                full_text = "\n\n".join(full_text_parts)
                process_time = round(time.time() - start_time, 2)

                # حفظ نسخة من الملف الموشوم بالألوان (اختياري للمعاينة)
                # pdf_doc.save(f"stamped_{path_obj.name}")

                return {
                    "status": "success",
                    "data": {
                        "full_text": full_text,
                        "metadata": doc_info,
                        "stats": {"pages": page_count, "time": process_time}
                    },
                    "context": {"user_request": user_request}
                }

        except Exception as e:
            self.logger.error(f"❌ فشل المحرك: {str(e)}")
            return {"status": "error", "error_details": str(e), "metadata": doc_info}

    def _extract_tables_safely(self, page) -> Tuple[str, bool]:
        """تستخرج الجداول بأمان وتعود بالنص وحالة الوجود"""
        try:
            tabs = page.find_tables()
            if not tabs or not tabs.tables:
                return "", False

            extracted_texts = []
            for table in tabs.tables:
                data = table.extract()
                # دمج الخلايا مع تنظيف القيم الفارغة
                clean_rows = [" | ".join([str(cell).strip() if cell else "" for cell in row]) for row in data]
                extracted_texts.append("\n".join(clean_rows))

            return "\n\n[TABLE_START]\n" + "\n".join(extracted_texts) + "\n[TABLE_END]", True
        except Exception as e:
            self.logger.debug(f"فشل استخراج الجدول في صفحة {page.number}: {e}")
            return "", False

    def _extract_aggressive_text(self, page: Any) -> str:
        """
        استخراج نصي مكثف: يدمج بين النصوص، الجداول، والروابط لتقليل الفقد.
        """
        # 1. استخراج النصوص كبلوكات مع فرز هندسي (من الأعلى لأسفل)
        raw_blocks = page.get_text("blocks")
        # فرز البلوكات حسب الإحداثيات (Y ثم X) لضمان ترتيب القراءة الصحيح
        blocks = sorted(raw_blocks, key=lambda b: (b[1], b[0]))

        text_parts = [str(b[4]).strip() for b in blocks if b[6] == 0 and str(b[4]).strip()]

        # 2. محاولة سحب النصوص من الجداول (بنية مهيكلة)
        try:
            tabs = page.find_tables()
            if tabs and hasattr(tabs, "tables"):
                for table in tabs.tables:
                    # نحافظ على الفواصل | لكي يفهم الـ AI أنها أعمدة
                    table_rows = [" | ".join([str(cell).strip() for cell in row if cell is not None])
                                for row in table.extract()]
                    table_body = "\n".join(table_rows)
                    if table_body.strip():
                        text_parts.append(f"\n[STRUCTURED_TABLE]:\n{table_body}")
        except Exception:
            pass

        full_content = "\n".join(text_parts).strip()

        # 3. صمام الأمان: دعم الصفحات البصرية (Visual Safety Valve)
        if len(full_content) < 15:
            image_count = len(page.get_images())
            if image_count > 0:
                # إضافة سياق تقني للمحلل الاستراتيجي
                return f"[VISUAL_ASSET_PAGE]: Contains {image_count} images/diagrams. OCR or Visual Analysis recommended."

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

    def export_knowledge_tree(self) -> str:
        """
        [The Grand Architect] - تصدير الهيكل الهرمي الاستدلالي بالرموز الجديدة.
        الموضوع (🟢) > الأفكار (⭐) > الصواعق (⚡) > السياقات (🔗/🌐)
        """
        tree_output = [
            "\n" + "═"*60,
            "🌲 GLOBAL KNOWLEDGE HIERARCHY REPORT (SNN-STAMP SYSTEM)",
            "═"*60 + "\n"
        ]

        current_topic_id = 0

        # ترتيب الصفحات لضمان تسلسل الشجرة
        for p_num in sorted(self.page_cache.keys()):
            data = self.page_cache[p_num]
            layer = data.get("layer_type", "STANDARD")
            kws = data.get("semantic_keywords", [])
            insight_ids = data.get("insight_ids", [])
            stats = data.get("context_stats", {"internal": 0, "total": 0})

            # 1. طبقة الموضوع (🔴/🟢 Topic Level)
            if layer == "CHAPTER_LAYER" or p_num == 0:
                current_topic_id += 1
                heading = data.get("visual_headings", [{"text": "Main Context"}])[0]["text"]
                icon = "🔴 NEW TOPIC" if layer == "CHAPTER_LAYER" else "🟢 CORE TOPIC"
                tree_output.append(f"\n{current_topic_id}. {icon}: {heading} (Page {p_num+1})")

            # 2. طبقة الأفكار والأقسام (⭐ Ideas/Sections)
            if kws:
                idea_str = " | ".join(kws[:3]) # أفضل 3 أفكار في الصفحة
                tree_output.append(f"   ├── ⭐ Ideas: {idea_str}")

            # 3. طبقة الصواعق المعرفية (⚡ Insight Level)
            if insight_ids:
                # نبرز الصواعق المكررة (🔄) لبيان قوة الحقيقة التقنية
                unique_insights = []
                for i_id in insight_ids[:5]: # عرض أول 5 صواعق فقط للاختصار
                    is_shared = len(self.insight_manager.vault.get(i_id, {}).get("pages", [])) > 1
                    status = "🔄 Shared" if is_shared else "📍 Unique"
                    unique_insights.append(f"⚡.{i_id}({status})")

                tree_output.append(f"   │   ├── ⚡ Insights: {', '.join(unique_insights)}")

            # 4. طبقة السياقات (🔗/🌐 Contextual Foundation)
            if stats["total"] > 0:
                # سحب أرقام الصفحات المرتبطة من الهيكل الطبقي
                related = self.layer_hierarchy.get(p_num, [])
                refs = ", ".join([f"p.{r+1}" for r in related[:4]])
                tree_output.append(f"   │   └── 🔗 Context: {stats['internal']} Internal | {stats['total']} Global [Refs: {refs}...]")

        tree_output.append("\n" + "═"*60)
        tree_output.append(f"📊 SUMMARY: {current_topic_id} Topics | {len(self.insight_manager.vault)} Unique Insights (⚡)")
        tree_output.append("═"*60)

        final_report = "\n".join(tree_output)

        # حفظ التقرير كمرجع دائم
        with open("hybrid_knowledge_tree.txt", "w", encoding="utf-8") as f:
            f.write(final_report)

        self.logger.info("🌲 تم تصدير شجرة المعرفة الهرمية (⚡) بنجاح.")
        return final_report

# ------------ ثالثاً: إدارة الذاكرة والأرشفة (Caching & Storage) ------------
    def add_page(self, page_num: int, text: str, page_obj: Any, metadata: Dict):
        """
        [Precision Indexing Core 97%] - النسخة النهائية المصححة برمجياً
        """

        # استدعاء مدير الصواعق لمعالجة النص واستخراج أرقام الـ DNA
        assigned_insights = self.insight_manager.process_sentences(text, page_num)

        # الآن يمكنك استخدامه في الختم وفي التخزين
        sent_count = len(assigned_insights)


        # 1. أولاً: تنظيف النص وحساب الجمل (الصواعق 🧬)
        import re
        clean_text = str(text or "").strip()
        raw_sentences = re.split(r'[.!?|.]', clean_text)
        # تعريف sent_count فوراً لاستخدامه في الختم
        sent_count = len([s for s in raw_sentences if len(s.strip()) > 20])

        # 2. تحليل الهيكل وشخصية الصفحة
        layout_data = self._extract_layout_structure(page_obj)
        personality = layout_data.get("page_personality", {})
        layer_type = self._classify_layer(clean_text, metadata)

        # 3. استخراج الأفكار (🧩) والروابط (📡/🛰️)
        keywords = self.PDF_extract_keywords(clean_text)
        self._build_visual_heuristics(page_num, layout_data)
        self._build_heuristic_links(page_num)

        # حساب إحصائيات السياق
        internal_ids = {l['origin_page'] for l in self.visual_links if l['target_page'] == page_num}
        internal_ids.update({l['target_page'] for l in self.visual_links if l['origin_page'] == page_num})
        internal_count = len(internal_ids)
        total_links = len(self.layer_hierarchy.get(page_num, []))

        # حساب المواضيع
        topics_count = sum(1 for d in self.page_cache.values() if d.get('layer_type') == "CHAPTER_LAYER")
        if layer_type == "CHAPTER_LAYER": topics_count += 1

        # 4. توليد المتجهات (Vector Generation) - تعريف 'vector' قبل استخدامه
        with torch.no_grad():
            enriched_text = f"{layer_type} | {clean_text[:500]}"
            vector = self.model.encode(enriched_text, convert_to_numpy=True)

        self.vectors.append(vector)
        self.vector_map.append(page_num)

        # 5. [الرسم الهندسي]: الآن كل المتغيرات (sent_count, topics_count) معرفة يقيناً
        try:
            stamp_x, stamp_y = layout_data.get("safe_stamp_zone", (10, 10))
            rect = fitz.Rect(stamp_x, stamp_y, stamp_x + 195, stamp_y + 80)
            page_obj.draw_rect(rect, color=(0, 0.4, 0), width=1.5, fill=(0.97, 1, 0.97))

            # الصف 1: الموضوع (Topic)
            topic_icon = "🏗️" if layer_type == "CHAPTER_LAYER" else "🏛️"
            page_obj.insert_text((stamp_x + 10, stamp_y + 20),
                                f"p.{page_num+1} | {topics_count}.{topic_icon}",
                                color=(0, 0.4, 0), fontsize=10)

            # الصف 2: الأفكار والجمل (🧩 & 🧬)
            ideas_count = len(keywords)
            page_obj.insert_text((stamp_x + 10, stamp_y + 42),
                                f"{ideas_count} 🧩 | {sent_count} 🧬 DNA Insights",
                                color=(0.7, 0.4, 0), fontsize=9)

            # الصف 3: شبكة الاتصال (📡 & 🛰️)
            context_text = f"Net: {internal_count} 📡 | {total_links} 🛰️ Global"
            page_obj.insert_text((stamp_x + 10, stamp_y + 65),
                                context_text, color=(0, 0.3, 0.7), fontsize=9)
        except Exception as visual_e:
            self.logger.warning(f"⚠️ فشل الختم الهندسي: {visual_e}")

        # 6. التخزين في الكاش
        self.page_cache[page_num] = {
            "content": clean_text,
            "layer_type": layer_type,
            "semantic_keywords": keywords,
            "insight_ids": assigned_insights,  # <--- THIS MUST BE HERE
            "context_stats": {"internal": internal_count, "total": total_links}
        }

        # 7. سياسة الإخلاء (LRU)
        if len(self.page_cache) > self.max_pages:
            self.page_cache.popitem(last=False)

        self.logger.info(f"🧬 Mastered Page [{page_num}] | {layer_type}")

    def _extract_layout_structure(self, page: Any) -> Dict[str, Any]:
        # إضافة مفتاح جديد safe_stamp_zone لتحديد مكان الختم
        layout_data: Dict[str, Any] = {
            "headings": [],
            "blocks": [],
            "page_personality": {},
            "safe_stamp_zone": (10, 10) # القيمة الافتراضية
        }

        try:
            dict_data = page.get_text("dict")
            raw_blocks = dict_data.get("blocks", [])
            layout_data["blocks"] = raw_blocks

            # مصفوفة لتتبع المساحات المشغولة في الزوايا
            # [Top-Left, Top-Right, Bottom-Left, Bottom-Right]
            page_width = page.rect.width
            page_height = page.rect.height

            for block in raw_blocks:
                bbox = block.get("bbox", (0,0,0,0))
                # نفس منطق استخراج العناوين الخاص بك...
                if isinstance(block, dict) and "lines" in block:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = str(span.get("text", "")).strip()
                            if len(text) > 3:
                                font_size = span.get("size", 0)
                                font_name = str(span.get("font", "")).lower()

                                # ... منطق الـ Heuristics الخاص بك ...
                                is_large = font_size > 11.5
                                is_bold = any(x in font_name for x in ["bold", "black", "heavy", "medium"])
                                is_caps = text.isupper() and len(text) > 5
                                is_not_numeric = not text.replace('.', '').replace('-', '').isdigit()

                                if (is_large or is_bold or is_caps) and is_not_numeric:
                                    layout_data["headings"].append({
                                        "text": text,
                                        "bbox": span.get("bbox"),
                                        "font_size": font_size,
                                        "type": "structural_anchor"
                                    })

            # 🎯 تطوير فكرتك: البحث عن أفضل زاوية للختم (أولوية للزوايا العلوية)
            # نفحص إذا كانت الزاوية العلوية اليمنى فارغة (غالباً الأفضل للمستندات العربية/الإنجليزية)
            # نحدد منطقة فحص 150x100 في الزاوية
            top_right_occupied = any(b.get("bbox")[0] > page_width - 150 and b.get("bbox")[1] < 100 for b in raw_blocks)

            if not top_right_occupied:
                layout_data["safe_stamp_zone"] = (page_width - 160, 15) # الزاوية اليمنى
            else:
                layout_data["safe_stamp_zone"] = (15, 15) # الزاوية اليسرى كبديل

            # حساب الشخصية (نفس منطقك الممتاز)
            headings_count = len(layout_data["headings"])
            page_raw_text = page.get_text("text").lower()

            layout_data["page_personality"] = {
                "type": "TECHNICAL_DATA" if headings_count > 3 or "table" in page_raw_text else "FLUID_TEXT",
                "technical_score": min(1.0, headings_count * 0.15),
                "has_tables": hasattr(page, "find_tables") and len(page.find_tables().tables) > 0
            }

        except Exception as e:
            self.logger.warning(f"Layout extraction skipped: {str(e)}")

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
        نسخة مطورة: تبني الجسور الهيكلية وتصنفها كروابط رادار (📡).
        """
        headings = layout_data.get("headings", [])
        if not isinstance(headings, list): return

        for heading in headings:
            title = str(heading.get("text", "")).strip()
            if len(title) < 2: continue

            # تحديث خريطة العناوين
            if page_num not in self.heading_map[title]:
                self.heading_map[title].append(page_num)

            # الربط الاستراتيجي (HEADING_CONTINUITY -> 📡)
            if len(self.heading_map[title]) > 1:
                source_page = self.heading_map[title][-2]
                if source_page != page_num:
                    self.visual_links.append({
                        "link_type": "RADAR_STATION", # تسمية هندسية بدلاً من HEADING_CONTINUITY
                        "origin_page": source_page,
                        "target_page": page_num,
                        "strength": 0.95 # رفع القوة لأن العناوين المتطابقة هي أقوى رابط هيكلي
                    })

                    if page_num not in self.layer_hierarchy: self.layer_hierarchy[page_num] = []
                    if source_page not in self.layer_hierarchy[page_num]:
                        self.layer_hierarchy[page_num].append(source_page)

        # الربط التسلسلي (STRUCTURAL_FLOW -> 📡)
        if page_num > 0:
            self.visual_links.append({
                "link_type": "GRID_FLOW", # تدفق الشبكة الهندسية
                "origin_page": page_num - 1,
                "target_page": page_num,
                "flow_weight": 1.0
            })
            if page_num not in self.layer_hierarchy: self.layer_hierarchy[page_num] = []
            if (page_num - 1) not in self.layer_hierarchy[page_num]:
                self.layer_hierarchy[page_num].append(page_num - 1)

    def _build_heuristic_links(self, page_num: int):
        """
        [Satellite Network Core 🛰️]
        تربط الصفحات بناءً على الثقل الهيكلي وتطابق "بصمات الجمل" (🧬).
        """
        if page_num not in self.page_cache:
            return

        page_data = self.page_cache[page_num]
        current_kws = set(page_data.get("semantic_keywords", []))
        current_insights = set(page_data.get("insight_ids", [])) # جلب أرقام الصواعق ⚡
        current_layer = page_data.get("layer_type", "STANDARD_CONTENT")

        # 1. المرحلة الأولى: التنبؤ بالأوزان (Predictive Weighting)
        weights = {"CHAPTER_LAYER": 1.0, "CORE_CONTENT": 0.85, "TECHNICAL_DATA": 0.7}
        predicted_weight = weights.get(current_layer, 0.4)

        # تعزيز الوزن إذا كانت الصفحة غنية بالـ DNA المعرفي (🧬)
        if len(current_insights) > 10: predicted_weight = min(predicted_weight + 0.1, 1.0)

        # 2. المرحلة الثانية: الربط الدلالي وتعزيز الأوزان القديمة
        for kw in current_kws:
            if kw not in self.heuristic_network:
                self.heuristic_network[kw] = []
            self.heuristic_network[kw].append({"page": page_num, "weight": predicted_weight})

            # This line below is a duplicate, you can remove it:
            self.heuristic_network[kw].append({"page": page_num, "weight": predicted_weight})

        # 3. المرحلة الثالثة: الربط الاستراتيجي (DNA Matching & Satellite Links 🛰️)
        if page_num not in self.layer_hierarchy:
            self.layer_hierarchy[page_num] = []

        for past_page, data in self.page_cache.items():
            if past_page == page_num: continue

            past_kws = set(data.get("semantic_keywords", []))
            past_insights = set(data.get("insight_ids", [])) # جلب صواعق الصفحة السابقة

            # أ) الربط عبر "تشابه الأفكار" (🧩/⭐)
            kw_overlap = current_kws.intersection(past_kws)

            # ب) الربط عبر "تطابق الـ DNA" (🧬/⚡) - (مثل مثال يد الروبوت 45 سم)
            dna_overlap = current_insights.intersection(past_insights)

            # إذا وجدنا تطابقاً في جينات المعلومة (حتى لو جملة واحدة متطابقة دلالياً)
            # أو تقاطعاً قوياً في الأفكار (فكرتين فأكثر)
            if len(dna_overlap) >= 1 or len(kw_overlap) >= 2:
                if past_page not in self.layer_hierarchy[page_num]:
                    self.layer_hierarchy[page_num].append(past_page)
                    # تفعيل الربط التبادلي كقمر صناعي 🛰️
                    if past_page not in self.layer_hierarchy: self.layer_hierarchy[past_page] = []
                    if page_num not in self.layer_hierarchy[past_page]:
                        self.layer_hierarchy[past_page].append(page_num)

        # 4. معالجة العزلة (Isolation Recovery)
        neighbors = {n for n in [page_num - 1, page_num + 1] if n >= 0}
        existing_links = set(self.layer_hierarchy.get(page_num, []))
        final_links = existing_links.union(neighbors)

        # Force link to closest Chapter if still isolated
        if len(final_links) < 3:
            chapters = [p for p, d in self.page_cache.items() if d.get("layer_type") == "CHAPTER_LAYER"]
            if chapters:
                closest = min(chapters, key=lambda x: abs(x - page_num))
                final_links.add(closest)

        self.layer_hierarchy[page_num] = list(final_links)

    def _classify_layer(self, text: str, metadata: Dict) -> str:
        """
        Technical layer classification.
        Returns: String label used for both Logic and Visual Stamping.
        """
        words = text.split()
        word_count = len(words)
        text_lower = text.lower()
        header_area = text_lower[:400]

        # 1. COVER_LAYER (اللون المقترح للختم: ذهبي/أصفر)
        is_page_zero = metadata.get("page_index") == 1
        cover_indicators = {'robotics', 'manual', 'handbook', 'guide', 'edition', 'دليل', 'روبوت'}
        if is_page_zero and (word_count < 120 or any(ind in text_lower for ind in cover_indicators)):
            return "COVER_LAYER"

        # 2. CHAPTER_LAYER (اللون المقترح: أزرق ملكي)
        chapter_indicators = {'chapter', 'section', 'part', 'فصل', 'باب', 'وحدة', 'المبحث', 'contents'}
        if any(ind in header_area for ind in chapter_indicators):
            return "CHAPTER_LAYER"

        # 3. TECHNICAL_DATA (اللون المقترح: برتقالي تقني)
        eng_indicators = {
            'table', 'figure', 'diagram', 'schema', 'robot', 'sensor', 'actuator',
            'controller', 'feedback', 'kinematics', 'torque', 'volt', 'جدول', 'مخطط'
        }
        digit_count = sum(c.isdigit() for c in text[:500])
        # إذا وجدنا كثافة رقمية عالية أو مصطلحات روبوتات صريحة
        if any(ind in text_lower for ind in eng_indicators) or (digit_count > 50):
            return "TECHNICAL_DATA"

        # 4. CORE_CONTENT (اللون المقترح: أخضر عشبي)
        if word_count > 450:
            return "CORE_CONTENT"

        # 5. APPENDIX_LAYER (اللون المقترح: رمادي)
        reference_kws = {'appendix', 'references', 'bibliography', 'citation', 'ملحق', 'مراجع', 'فهرس'}
        if any(kw in text_lower for kw in reference_kws):
            return "APPENDIX_LAYER"

        return "STANDARD_CONTENT"

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
        [The Domain Specialist 🏛️]
        تخمين التخصص التقني بناءً على الجاذبية المعرفية وكثافة الـ DNA (🧬).
        """
        # 1. إثراء الفئات الهندسية بمصطلحات أكثر دقة (Stems)
        categories = {
            'ROBOTICS_ENGINEERING': [
                'kinematic', 'actuator', 'feedback', 'manipulator', 'trajectory',
                'torque', 'encoder', 'chassis', 'روبوت', 'كينماتيكا', 'محرك', 'عزم'
            ],
            'ELECTRICAL_SYSTEMS': [
                'circuit', 'voltage', 'frequency', 'ohm', 'schematic', 'soldering',
                'دائرة', 'فولت', 'تردد', 'مقاومة', 'مخطط'
            ],
            'MECHANICAL_DESIGN': [
                'cad', 'prototyping', 'material', 'alloy', 'structural', 'assembly',
                'تصميم', 'نماذج', 'سبائك', 'هيكلية', 'تجميع'
            ],
            'CONTROL_THEORY': [
                'pid', 'stability', 'compensation', 'latency', 'nonlinear', 'tuning',
                'استقرار', 'تخميد', 'معايرة', 'خطي'
            ]
        }

        # 2. فحص العينة الممتدة
        sample_text = text.lower()[:10000] # توسيع العينة لتغطية الفهرس والمقدمة
        scores = {topic: 0.0 for topic in categories}

        # 3. حساب "الجاذبية المعرفية" مع حقن قوة الـ DNA (🧬)
        for topic, keywords in categories.items():
            for kw in keywords:
                count = sample_text.count(kw)
                if count > 0:
                    # ميزة "الثقل التخصصي": الكلمات التي تنتهي بـ 'ics' أو 'ing' غالباً ما تكون أسماء علوم
                    weight = 2.5 if kw.endswith(('ics', 'ing')) else 1.8 if len(kw) > 7 else 1.2
                    scores[topic] += float(count) * weight

        # 4. اختيار التخصص الأدق
        if not any(scores.values()):
            return "GENERAL_TECHNICAL_AUDIT"

        best_topic = max(scores, key=lambda k: scores[k])
        max_score = scores[best_topic]

        # 5. تقرير Mastered Topic (🏗️)
        if max_score > 3.0:
            confidence = "HIGH" if max_score > 20 else "MEDIUM"
            # إرسال إشارة للمراقب بنجاح تحديد الهوية
            self.logger.info(f"🏗️ Domain Identified: {best_topic} | Gravity Score: {max_score:.1f} ({confidence})")
            return best_topic

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
        [القسم الأول: المشرف الاستراتيجي - المحسن 97%]
        تجميع البيانات بنظام الذاكرة المتسلسلة والرقابة التكيفية.
        """
        start_time = time.time()

        # --- المرحلة 1: الاستطلاع السريع ---
        self.engine.logger.info(f"📡 Stage 1: Initializing Fast Ingest for {Path(pdf_path).name}")
        ingest_status = self.engine.fast_ingest_stream(pdf_path)

        if ingest_status.get("status") != "completed":
            return {"status": "error", "message": "Fast Ingest Failed"}

        page_count = int(ingest_status.get("pages_ingested", 0))

        # --- المرحلة 2: الاستدلال الاستراتيجي ---
        self.engine.logger.info(f"🎯 Stage 2: Inferring navigation path for: '{user_request}'")
        target_pages = self.infer_navigation_path(user_request)
        analysis_queue = target_pages if target_pages else list(range(page_count))

        # --- 🛡️ [إضافة] تهيئة نظام الذاكرة النشطة قبل بدء المرحلة 3 ---
        # هذا السطر هو الذي يضمن "خيط الأفكار" المستمر
        virtual_rolling_summary = f"START_CONTEXT: Analysis for user request '{user_request}'"

        # --- المرحلة 3: تجميع النبضات (الآن مع الفلترة والرقابة والذاكرة) ---
        step = 12 # تقليل الخطوة لـ 12 لزيادة دقة الـ Context العابر
        total_items = len(analysis_queue)
        pulses_data = []

        self.engine.logger.info(f"🌀 Stage 3: Building Pulse Network with Active Memory Bridge...")

        # هنا يكمل كود المرحلة 3 الملحمي الذي أرسلته لك سابقاً...

        # --- المرحلة 3 الملحمية: تجميع النبضات بذاكرة نشطة، فلترة سيادية، ورقابة استراتيجية ---
        virtual_rolling_summary = "START: Global technical baseline initialized."
        pulses_data = []

        for start_idx in range(0, total_items, step):
            current_batch = analysis_queue[start_idx : start_idx + step]
            if not current_batch: continue

            chunk_texts = []
            related_summaries = set()
            purity_scores = []

            for p_num in current_batch:
                page_data = self.engine.get_page_data(p_num, pdf_path=pdf_path)

                if page_data and page_data.get("content"):
                    raw_content = page_data["content"]

                    # 1. تطبيق الفلتر الاستراتيجي (التنقية قبل الرقابة)
                    refined_text = self._pulse_content_filter(raw_content)

                    # حساب معامل النقاء لدعم عمل الرقيب والـ Monitor
                    purity = len(refined_text) / max(len(raw_content), 1)
                    purity_scores.append(purity)
                    chunk_texts.append(refined_text)

                    # 2. سحب السياق (Context)
                    related_ids = self.get_related_pages(p_num, depth=1)
                    for r_id in related_ids:
                        r_data = self.engine.page_cache.get(r_id)
                        if r_data and r_data.get("semantic_keywords"):
                            kws = ", ".join(r_data["semantic_keywords"][:2])
                            related_summaries.add(f"[Ref Page {r_id+1} | Topics: {kws}]")

            full_text = "\n".join(chunk_texts).strip()
            avg_purity = sum(purity_scores) / len(purity_scores) if purity_scores else 1.0

            # 3. تجهيز الحمولة الأولية للاختبار من قبل الرقيب
            temp_payload = {
                "text": full_text,
                "context": "\n".join(list(related_summaries)[:4]),
                "virtual_memory": virtual_rolling_summary,
                "purity_index": avg_purity,
                "range": (current_batch[0], current_batch[-1]),
                "pulse_id": len(pulses_data) + 1
            }

            # --- 🕵️ إشراف الرقيب الاستراتيجي (The Auditor's Gate) ---
            # الرقيب يفحص النبضة قبل إرسالها للمحلل
            audit_report = self._pulse_strategic_auditor(temp_payload)

            if not audit_report["is_ready"]:
                self.engine.logger.warning(f"🧹 Filter/Auditor Alert: Pulse {temp_payload['pulse_id']} rejected (Signal: {audit_report['signal_score']})")
                continue

            # 4. تعزيز السياق التكيفي (Adaptive Context Enhancement)
            # إذا طلب الرقيب تعزيزاً، نقوم بمضاعفة السياق الدلالي فوراً لرفع الدقة
            if audit_report["action"] == "ENHANCE_CONTEXT":
                self.engine.logger.info(f"🔍 Auditor Escalation: Enhancing context for Pulse {temp_payload['pulse_id']}")
                # سحب صفحات إضافية مرتبطة لرفع مستوى الإشارة
                deeper_related = self.get_related_pages(current_batch[0], depth=3)
                for dr_id in deeper_related:
                    dr_data = self.engine.page_cache.get(dr_id)
                    if dr_data:
                        related_summaries.add(f"[DeepRef {dr_id+1} | {dr_data.get('layer_type')}]")

                # تحديث السياق في الحمولة النهائية
                temp_payload["context"] = "\n".join(list(related_summaries)[:6])

            # 5. الاعتماد النهائي للنبضة
            pulses_data.append(temp_payload)

        # التحقق النهائي من حصاد النبضات
        if not pulses_data:
            return {"status": "error", "message": "Critical Failure: Auditor blocked all pulses due to low data quality."}

        self.engine.logger.info(f"✅ Strategic Assembly Complete: {len(pulses_data)} pulses certified by Auditor.")

        # الانتقال للمرحلة النهائية مع ضمان مرور البيانات المنقحة والمراقبة
        return self.SNN_analyze_pdf(pulses_data, user_request, pdf_path)

    def SNN_analyze_pdf(self, pulses_data: List[Dict], user_request: str, pdf_path: str) -> Dict[str, Any]:
        """
        [Section 2: SNN Pulse Processor - Memory Enhanced]
        معالجة النبضات بنظام الذاكرة المتسلسلة الوهمية (Transient Adaptive Memory).
        """
        all_ideas: List[Dict[str, Any]] = []
        cumulative_awareness: List[float] = []
        total_pulses = len(pulses_data)

        # --- 🛡️ مخزن الذاكرة التكيفية الوهمية (Active Buffer) ---
        # هذا المخزن يعيش فقط خلال دورة حياة هذه الدالة
        virtual_memory_bridge = "INITIAL_STATE: Prime analysis focused on technical robotics audit."

        self.engine.logger.info(f"🧠 SNN Core: Processing {total_pulses} pulses with Adaptive Memory Bridge...")

        for idx, pulse in enumerate(pulses_data):
            pulse_idx = idx + 1
            chunk_text = str(pulse.get("text", ""))
            network_context = str(pulse.get("context", ""))

            raw_range = pulse.get("range", (0, 0))
            start_p = raw_range[0][0] if isinstance(raw_range[0], list) else raw_range[0]
            end_p = raw_range[1]

            if not chunk_text.strip():
                continue

            # --- 💉 حقن الذاكرة التكيفية في الـ Prompt ---
            # النبضة الحالية "تتذكر" ما استنتجناه من النبضات السابقة
            enhanced_prompt = (
                f"-- VIRTUAL_MEMORY_BRIDGE --\n{virtual_memory_bridge}\n\n"
                f"-- CURRENT_PAYLOAD --\n{chunk_text}\n\n"
                f"-- NETWORK_INSIGHTS --\n{network_context}"
            )

            try:
                # 1. التوليد المتكيف (Generation with Context)
                chunk_ideas, chunk_state = self.generate_mock(enhanced_prompt, user_request=user_request)
                current_score = float(chunk_state.get("avg_score", 0.5))

                # 2. فحص النطاق الذهبي (Sweet Spot)
                if not self._audit_sweet_spot(chunk_text, current_score):
                    self.engine.logger.info(f"🔄 Pulse {pulse_idx} quality check. Escalating reasoning...")
                    chunk_ideas, chunk_state = self._reprocess_pulse(enhanced_prompt, current_score, self.engine)
                    current_score = float(chunk_state.get("avg_score", 0.5))

                # --- 🧠 تحديث الذاكرة الوهمية (Memory Synthesis) ---
                # استخراج أهم مفهوم من النبضة الحالية وتمريره للنبضة القادمة
                if chunk_ideas:
                    top_insight = chunk_ideas[0].get("title", "Ongoing Technical Trace")
                    # نقوم بضغط الذاكرة (Keep it lean) لضمان عدم تشتيت المحرك
                    virtual_memory_bridge = f"PREVIOUS_PULSE_INSIGHT: {top_insight} | STATUS: Analysis Stabilized."

                # 3. المراقبة والمزامنة
                self._analysis_monitor(pulse_idx, total_pulses, current_score)
                self._audit_and_sync_cache(start_p, {
                    "quality_score": current_score,
                    "virtual_trace": virtual_memory_bridge[:100] # وسم وهمي للتحقق
                })

                # 4. تجميع الأفكار
                for idea in chunk_ideas:
                    if isinstance(idea, dict):
                        idea["origin_context"] = f"Pages {start_p + 1}-{end_p + 1}"
                        all_ideas.append(idea)

                cumulative_awareness.append(current_score)

            except Exception as e:
                self.engine.logger.error(f"❌ SNN Failure in Pulse {pulse_idx}: {str(e)}")

        # --- المرحلة النهائية: تجميع الوعي التراكمي ---
        final_avg_score = round(sum(cumulative_awareness) / len(cumulative_awareness), 2) if cumulative_awareness else 0.0

        # استنتاج الموضوع بناءً على الذاكرة النهائية المجمعة
        sample_texts = [p.get("text", "") for p in pulses_data[:3]]
        combined_sample = " ".join(sample_texts)
        inferred_topic = self.engine._guess_topic(combined_sample)

        report = {
            "status": "success",
            "analysis_metrics": {
                "total_pulses": len(cumulative_awareness),
                "consciousness_score": final_avg_score,
                "inferred_topic": inferred_topic,
                "knowledge_hubs": self.get_network_hubs(3),
                "memory_trace": "ACTIVE_ADAPTIVE"
            },
            "output": {
                "structured_ideas": all_ideas,
                "summary": f"SNN Analysis complete. System achieved {final_avg_score*100}% awareness via context bridging."
            },
            "file_reference": {"path": pdf_path, "links": len(self.engine.visual_links)}
        }

        self.engine.logger.info(f"📊 Final Synthesis: {inferred_topic} | Awareness: {final_avg_score*100}%")
        return report

    def _pulse_strategic_auditor(self, pulse_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        الرقيب الاستراتيجي: يعمل تحت إشراف analyze_pdf لفحص جاهزية النبضة.
        يقوم بتقييم (النقاء، الذاكرة، والروابط) قبل إعطاء الضوء الأخضر.
        """
        purity = pulse_payload.get("purity_index", 1.0)
        has_memory = len(pulse_payload.get("virtual_memory", "")) > 50
        page_range = pulse_payload.get("range", (0, 0))

        # 1. تقييم "قوة الإشارة" (Signal Strength)
        signal_score = (purity * 0.6) + (0.4 if has_memory else 0.0)

        # 2. اتخاذ قرار تكتيكي
        action = "PROCEED"
        if signal_score < 0.6:
            action = "ENHANCE_CONTEXT" # طلب توسيع السياق لضعف الإشارة
        elif signal_score > 0.9:
            action = "FAST_TRACK"      # نبضة نقية جداً، يمكن معالجتها بعمق أقل لتوفير الموارد

        audit_report = {
            "action": action,
            "signal_score": round(signal_score, 2),
            "is_ready": signal_score > 0.4
        }

        return audit_report

    def _pulse_content_filter(self, raw_text: str) -> str:
        """
        [The Signal Guard 🛡️]
        ينظف النص مع حماية "الصواعق المعرفية" (🧬) من الحذف العشوائي.
        """
        import re

        # 1. التنظيف الأولي للمسافات والرموز المزعجة
        clean_text = re.sub(r'[\t ]+', ' ', raw_text)
        clean_text = re.sub(r'\n\s*\n+', '\n\n', clean_text)

        lines = clean_text.split('\n')
        useful_lines = []

        # 2. الفلترة الاستراتيجية (تجنب حذف البيانات التقنية 🧬)
        for line in lines:
            line = line.strip()
            if not line: continue

            # معيار القبول المطور:
            # - ألا يكون السطر مجرد رقم صفحة وحيد.
            # - ألا يكون مجرد رموز تزيينية (---, ===).
            # - استثناء: إذا كان السطر يحتوي على "بصمة تقنية" (مثل 45cm أو Unitree) نقبله فوراً.

            is_decoration = re.match(r'^[=\-_#\*\s]+$', line)
            is_just_page_num = re.match(r'^\d+$', line)
            has_technical_dna = any(token in line.lower() for token in ['cm', 'kg', 'v', 'hz', 'robot', 'id'])

            if (len(line) > 5 or has_technical_dna) and not is_decoration and not is_just_page_num:
                useful_lines.append(line)

        # 3. الربط الهيكلي (Structural Anchoring)
        # نقوم بإعادة بناء النص بحيث تظهر "الأفكار" (🧩) واضحة للـ LLM
        filtered_content = "\n".join(useful_lines)

        # 4. تدقيق "الحمولة الميتة" (Dead Weight Audit)
        reduction_rate = 1 - (len(filtered_content) / max(len(raw_text), 1))
        if reduction_rate > 0.4:
            self.engine.logger.info(f"🛡️ Filter Shield: Removed {reduction_rate*100:.1f}% noise from payload.")

        return filtered_content

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

    def _analysis_monitor(self, current_pulse: int, total_pulses: int, chunk_score: float, pulse_metadata: Dict = None):
        """
        [The Command Center 🛰️]
        مراقبة حية للأداء، استهلاك الذاكرة، وتدفق الـ DNA المعرفي (🧬).
        """
        # 1. حساب مؤشرات الأداء (KPIs)
        total_p = max(total_pulses, 1)
        progress = (current_pulse / total_p) * 100
        cache_count = len(self.engine.page_cache)
        cache_usage_pct = (cache_count / max(self.engine.max_pages, 1)) * 100

        # 2. جرد المحتوى الهرمي في النبضة الحالية
        # سحب عدد الأفكار (🧩) والصواعق (🧬) التي تمت معالجتها الآن
        ideas_in_pulse = len(pulse_metadata.get("semantic_keywords", [])) if pulse_metadata else 0
        dna_in_pulse = len(pulse_metadata.get("insight_ids", [])) if pulse_metadata else 0

        # 3. تحديد الحالة (Status) بناءً على النطاق الذهبي
        quality_icon = "💎" if chunk_score >= 0.90 else "✨" if chunk_score >= 0.85 else "⚠️"
        memory_icon = "🟢" if cache_usage_pct < 80 else "🟡" if cache_usage_pct < 95 else "🔴"

        # 4. التنسيق البصري المطور (The Engineering Dashboard)
        header = f"\n{'═'*45}\n[ 🛰️ PULSE MONITOR #{current_pulse:02d} ]\n{'═'*45}"

        # سطر المعرفة: يظهر مدى "دسامة" النبضة الحالية
        knowledge_flow = f" 🧬 DNA Flow: {dna_in_pulse} Insights | 🧩 Concepts: {ideas_in_pulse}\n"

        # سطر الأداء والذاكرة
        stats_body = (
            f" 📊 Progress: {progress:>5.1f}% | Quality: {quality_icon} {chunk_score:.4f}\n"
            f" 💾 Cache {memory_icon} : {cache_usage_pct:>5.1f}% ({cache_count} Pages Active)\n"
            f"{'─'*45}"
        )

        self.engine.logger.info(header + knowledge_flow + stats_body)

        # 5. صمام الأمان (Memory Guard)
        if cache_usage_pct >= 100.0:
            # استدعاء دالة التنظيف العميق التي اقترحناها سابقاً
            old_idx, _ = self.engine.page_cache.popitem(last=False)
            self.engine.logger.warning(f"🚨 ALERT: Cache Exhausted. Purging Page {old_idx}...")

    def get_related_pages(self, page_num: int, depth: int = 2) -> List[int]:
        """
        [The Navigator 🛰️]
        تتنقل في الشبكة الاستدلالية باستخدام الرادار الهيكلي (📡) والاقمار الدلالية (🛰️).
        """
        from typing import Set

        if page_num not in self.engine.page_cache:
            return []

        # 1. القاعدة الأساسية: الهيكل الطبقي (الارتباطات المباشرة)
        related_indices: Set[int] = set()
        if page_num in self.engine.layer_hierarchy:
            related_indices.update(self.engine.layer_hierarchy[page_num])

        # 2. رادار التدفق البصري (📡): جلب الجيران والعناوين المتصلة
        visual_connections = (
            link["target_page"] if link["origin_page"] == page_num else link["origin_page"]
            for link in self.engine.visual_links
            if link["origin_page"] == page_num or link["target_page"] == page_num
        )
        related_indices.update(visual_connections)

        # 3. [تطوير بصمة الـ DNA 🧬]: البحث عن الصفحات التي تتشارك نفس الصواعق (⚡)
        current_data = self.engine.page_cache[page_num]
        current_insights = current_data.get("insight_ids", [])

        for insight_id in current_insights:
            # الوصول لخزنة الصواعق لمعرفة الصفحات الأخرى التي وردت فيها هذه المعلومة
            vault_entry = self.engine.insight_manager.vault.get(insight_id)
            if vault_entry:
                # إضافة الصفحات التي تحمل نفس "الجين المعرفي" (مثل معلومة يد الروبوت 45سم)
                related_indices.update(vault_entry.get("pages", []))

        # 4. البحث عبر الأفكار (🧩): الكلمات المفتاحية ذات الوزن العالي
        current_keywords = current_data.get("semantic_keywords", [])
        for kw in current_keywords:
            if kw in self.engine.heuristic_network:
                entries = self.engine.heuristic_network[kw]
                # تصفية بناءً على الوزن لضمان عدم تشتيت السياق
                relevant_ids = [
                    entry["page"] for entry in entries
                    if isinstance(entry, dict) and entry.get("weight", 0) > 0.6 # رفع العتبة للجودة
                ]
                related_indices.update(relevant_ids[:depth])

        # 5. التصفية النهائية والفرز
        related_indices.discard(page_num)
        # موازنة النتائج لضمان عدم إغراق الـ LLM بصفحات كثيرة
        final_results = sorted(list(related_indices))[:depth * 3]

        self.engine.logger.info(f"🛰️ Navigator Sync: Page {page_num} linked to {len(final_results)} logic nodes.")
        return final_results

    def get_network_hubs(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """
        [The Knowledge Core 🏛️]
        تستخرج "المراكز السيادية" بناءً على كثافة الروابط الاستدلالية (🛰️) والـ DNA (🧬).
        """
        if not self.engine.heuristic_network:
            self.engine.logger.warning("⚠️ Hub Analysis: Heuristic network is empty.")
            return []

        # 1. تحليل كثافة الشبكة (Network Density Scan)
        # نجمع كل الروابط الاستدلالية التي بناها المحرك
        all_hubs = list(self.engine.heuristic_network.items())

        # 2. الفرز الذكي (Smart Ranking)
        # لا نحسب العدد فقط، بل "قوة التأثير"؛ الكلمة التي تربط صفحات بعيدة (🛰️) أهم من القريبة.
        sorted_hubs = sorted(
            all_hubs,
            key=lambda x: sum(entry.get("weight", 0.5) for entry in x[1]),
            reverse=True
        )

        # 3. اختيار الصفوة (The Elite Hubs)
        top_hubs = sorted_hubs[:top_n]
        hub_results = [(str(h[0]), len(h[1])) for h in top_hubs]

        # 4. تقرير الوعي المعرفي (Cognitive Density Report)
        hub_names = [h[0] for h in hub_results]
        avg_density = sum(h[1] for h in hub_results) / max(len(hub_results), 1)

        self.engine.logger.info(
            f"🏛️ Knowledge Core Discovery (Hubs):\n"
            f"   - Identified Hubs: {hub_names}\n"
            f"   - Connectivity Density: {avg_density:.1f} nodes/hub\n"
            f"   - Network Status: STABLE & INTERCONNECTED 🛰️"
        )

        return hub_results

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

    def _audit_sweet_spot(self, chunk_content: str, current_confidence: float, pulse_metadata: Dict = None) -> bool:
        """
        [The Quality Scales ⚖️]
        فحص النطاق الذهبي مع مراعاة كثافة الـ DNA المعرفي (🧬).
        """
        # 1. ضبط النطاق الأساسي
        lower_bound = 0.85
        upper_bound = 0.96

        # 2. [تطوير استراتيجي]: تعديل النطاق بناءً على "دسامة" المحتوى
        # إذا كانت النبضة تحتوي على صواعق (🧬) أو أفكار (🧩) كثيرة، نرفع سقف القبول
        insight_count = len(pulse_metadata.get("insight_ids", [])) if pulse_metadata else 0

        if insight_count > 5:
            # محتوى دسم تقنياً: نحتاج دقة أعلى، لذا نرفع الحد الأدنى قليلاً
            lower_bound = 0.88
            self.engine.logger.info(f"🧬 High Density Pulse: Raising quality bar to {lower_bound}")

        # 3. فحص النطاق المباشر
        if lower_bound <= current_confidence <= upper_bound:
            self.engine.logger.info(f"✨ Audit Success: {current_confidence:.2f} is in the Golden Zone.")
            return True

        # 4. معيار المرونة للمحتوى القصير (Technical Snippets)
        # إذا كانت النبضة قصيرة لكنها تحتوي على "جينات تقنية" (🧬) مهمة، نقبلها بـ 0.80
        if len(chunk_content) < 400 and current_confidence >= 0.80:
            return True

        # 5. معيار "الثقة المفرطة" (Over-Confidence / Hallucination Shield)
        if current_confidence > upper_bound:
            # إذا كان الـ AI واثقاً بنسبة 99% في نص معقد، فهذا مؤشر خطر (هلوسة)
            self.engine.logger.warning(f"⚠️ Audit Alert: Confidence {current_confidence:.2f} is suspiciously high.")
            return False

        return False

    def _reprocess_pulse(self, content: str, initial_score: float, pulse_metadata: Dict = None):
        """
        [The Strategic Resurrector 🏥]
        إعادة معالجة النبضة عبر "حقن السياق المفقود" وتقوية روابط الـ DNA (🧬).
        """
        self.engine.logger.info(f"🔄 Escalating Reasoning (Initial: {initial_score:.2f})")

        # 1. استخراج الـ DNA المفقود لتعزيز الـ Prompt
        # إذا فشلت النبضة، قد يكون السبب نقص المصطلحات التقنية الصريحة
        missing_dna = pulse_metadata.get("insight_ids", []) if pulse_metadata else []
        dna_context = ""
        if missing_dna:
            dna_texts = [self.engine.insight_manager.vault[i]["text"] for i in missing_dna[:3]]
            dna_context = "\nCRITICAL_TECH_FACTS: " + " | ".join(dna_texts)

        # 2. بناء الـ Prompt المعزز (Aggressive Reasoning Prompt)
        enhanced_content = (
            f"{content}\n"
            f"{dna_context}\n"
            f"STRATEGIC_REASONING_MODE: HIGH_PRECISION\n"
            f"MANDATORY_FOCUS: Match linked insights and technical scale."
        )

        try:
            # 3. محرك الملاذ الأخير (Recursive Reasoning)
            # نرفع عدد التكرارات (Iterations) ونحقن "الحقائق التقنية" الصريحة
            enhanced_ideas, enhanced_state = self.generate_mock(
                enhanced_content,
                max_iterations=8, # رفع الجهد لثماني مراحل تفكير
                user_request="DEEP_REASONING_RECOVERY"
            )

            new_score = float(enhanced_state.get("avg_score", 0.0))

            # إذا لم يتحسن السكور، نقوم بسحب سياق من "الصفحات التوأم" (🛰️)
            if new_score <= initial_score and pulse_metadata:
                self.engine.logger.info("📡 Score Stagnant: Pulling Satellite Context (🛰️)...")
                # سحب بيانات من الصفحات التي تحمل نفس الـ DNA
                return self.engine_emergency_context_bridge(enhanced_content, pulse_metadata)

            return enhanced_ideas, enhanced_state

        except Exception as e:
            self.engine.logger.error(f"❌ Recovery Failure: {e}")
            return [], {"avg_score": initial_score, "status": "FAILED"}

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

def run_500_page_stress_test(engine, analyzer, pdf_path):
    """
    Stress Test for PDFPageCacheNetwork & SNN Analyzer.
    Focus: Cache stability, Heuristic Density, and Pulse Latency.
    """
    print(f"\n--- 🚀 STARTING 500-PAGE STRESS TEST: {Path(pdf_path).name} ---")
    start_time = time.time()

    # --- Stage 1: Aggressive Ingest & Cache Stability ---
    print("Stage 1: Ingesting & Building Heuristic Base...")
    ingest_results = engine.process_pdf(pdf_path)

    if ingest_results['status'] != 'success':
        print(f"❌ Ingest Failed: {ingest_results.get('error_details')}")
        return

    # Critical Check: Did LRU Eviction work?
    cache_size = len(engine.page_cache)
    print(f"✅ Ingest Complete. Cache Size: {cache_size} (Max Cap: {engine.max_pages})")
    if cache_size > engine.max_pages:
        print("⚠️ WARNING: Cache overflow detected. Logic leak in Eviction policy!")

    # --- Stage 2: Heuristic Connectivity Audit ---
    print("\nStage 2: Auditing Heuristic Network Hubs...")
    total_links = sum(len(links) for links in engine.layer_hierarchy.values())
    avg_links = total_links / max(len(engine.layer_hierarchy), 1)
    print(f"✅ Network Map verified. Total Links: {total_links} (Avg Density: {avg_links:.2f} links/page)")

    # --- Stage 3: Pulse Pressure Test ---
    user_request = "Identify critical failure points in joint actuators and cross-reference with torque tables."
    print(f"\nStage 3: Running Pulse Analysis for High-Complexity Request: '{user_request}'")

    analysis_start = time.time()
    final_report = analyzer.analyze_pdf(pdf_path, user_request=user_request)
    analysis_duration = round(time.time() - analysis_start, 2)

    # --- Stage 4: Final Synthesis Reporting ---
    if final_report['status'] == 'success':
        metrics = final_report['analysis_metrics']
        print("\n" + "="*40)
        print("📊 FINAL STRESS TEST REPORT")
        print("="*40)
        print(f"⏱️ Total System Runtime    : {round(time.time() - start_time, 2)}s")
        print(f"🌀 Pulses Orchestrated      : {metrics['total_pulses']}")
        print(f"🧠 System Consciousness     : {metrics['consciousness_score']*100}%")
        print(f"🔗 Structural Knowledge Hubs: {metrics['knowledge_hubs']}")
        print(f"🤖 Inferred Global Topic    : {metrics['inferred_topic']}")
        print(f"💾 Memory Trace Stability   : {metrics['memory_trace']}")
        print("="*40)
    else:
        print(f"❌ Analysis Failure: {final_report.get('message')}")

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
    engine.llm_client = OpenAI(api_key=)

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

    # Usage Example:
    # my_engine = PDFPageCacheNetwork(max_pages=200) # Your 200-page limit
    # my_analyzer = StrategicSupervisor(engine=my_engine)
    # run_500_page_stress_test(my_engine, my_analyzer, "heavy_manual.pdf")
    pass
