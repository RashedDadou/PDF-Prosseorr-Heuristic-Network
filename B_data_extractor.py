# B_data_extractor.py

import fitz  # PyMuPDF
import re
from typing import List, Dict, Any
from collections import OrderedDict

class SovereignDataExtractor:
    """
    [المستخلص السيادي]: متخصص في تشريح ملفات PDF وتحويلها إلى كتل معرفية.
    صُمم للتعامل مع الأحمال العالية والمحتوى التقني المكثف.
    """
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_document(self, pdf_path: str) -> List[Dict[str, Any]]:
        """المحرك الرئيسي لتحويل الملف إلى كتل نصية موشومة."""
        document_structure = []
        try:
            with fitz.open(pdf_path) as doc:
                self.total_pages = len(doc)
                # إضافة سطر تتبع للتأكد من عدد الصفحات التي يراها النظام فعلياً
                print(f"DEBUG: الملف المكتشف يحتوي على {self.total_pages} صفحة.")

                for page_num in range(self.total_pages):
                    page = doc[page_num]
                    raw_text = page.get_text("text")

                    # حماية النوع
                    text = str(raw_text) if raw_text else ""

                    # تطهير النص
                    clean_text = self._purify_text(text)

                    # [إجراء وقائي]: إذا كانت الصفحة تحتوي على نص قليل جداً (أقل من 5 أحرف)
                    # سنتخطى عملية التقطيع ونأخذ النص كما هو لضمان عدم ضياع الصفحات شبه الفارغة
                    if len(clean_text) < 10 and len(clean_text) > 0:
                         document_structure.append({
                            "id": f"p{page_num}_minimal",
                            "content": clean_text,
                            "metadata": {"page": page_num + 1, "keywords": []}
                        })
                         continue

                    chunks = self._apply_overlap_chunking(clean_text)

                    for i, chunk_content in enumerate(chunks):
                        document_structure.append({
                            "id": f"p{page_num}_c{i}",
                            "content": chunk_content,
                            "metadata": {
                                "page": page_num + 1,
                                "keywords": self._extract_technical_tokens(chunk_content)
                            }
                        })

            # سطر تتبع نهائي
            print(f"DEBUG: تم توليد {len(document_structure)} كتلة معرفية (Nodes).")
            return document_structure
        except Exception as e:
            raise RuntimeError(f"Extraction Failure: {str(e)}")

    def _purify_text(self, text: str) -> str:
        """تطهير النص مع الحفاظ على هيكلية الأسطر للمصفوفات والمعادلات."""
        if not text:
            return ""
        # بدلاً من تحويل كل المسافات إلى مسافة واحدة، سنحافظ على فاصل الأسطر
        # لضمان بقاء المصفوفات (4x4) مفهومة للنظام
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    def _apply_overlap_chunking(self, text: str) -> List[str]:
        """تقطيع استراتيجي يحترم نهاية الفقرات لضمان سلامة المعادلات والمصفوفات."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        # تقسيم النص إلى فقرات أولاً للحفاظ على تماسك المعلومات التقنية
        paragraphs = text.split('\n')
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += para + "\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # الحفاظ على التداخل (Overlap) من خلال أخذ آخر جزء من الكتلة السابقة
                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else ""
                current_chunk = overlap_text + para + "\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _extract_technical_tokens(self, text: str) -> List[str]:
        """استخراج احترافي للمصطلحات التقنية، المصفوفات، والمتغيرات الهندسية."""
        # 1. نمط متطور لاصطياد:
        # - الكلمات التقنية (Kinematics, Trajectory)
        # - المتغيرات الهندسية (theta1, d1, L2)
        # - الرموز الرياضية والمصفوفات (T4x4, Cos, Sin)
        patterns = [
            r'\b[A-Z][a-z]{3,}\b',             # كلمات تقنية تبدأ بحرف كبير
            r'\b[a-z]+\d+\b',                  # متغيرات مثل q1, d2, theta1
            r'\b[A-Z]{1,2}\d{1,2}\b',          # مصفوفات أو نقاط مثل T1, P2
            r'[\d\.]+\s*[\+\-\*\/=]\s*[\d\.]+', # معادلات بسيطة
            r'sin|cos|tan|atan2'               # دالات مثلثية أساسية في الروبوتات
        ]

        combined_pattern = "|".join(patterns)
        tokens = re.findall(combined_pattern, text, re.IGNORECASE)

        # 2. تنظيف التوكنز من التكرار مع الحفاظ على الترتيب
        unique_tokens = list(OrderedDict.fromkeys(tokens))

        # 3. رفع السقف الذكي: السماح بـ 15 توكن بدلاً من 5 لضمان شمولية البيانات
        return unique_tokens[:15]
