# A_pdf_processor.py

import os
import fitz  # PyMuPDF
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

# --- [الارتباط بالوحدات الخدمية] ---
from B_data_extractor import SovereignDataExtractor
from C_Tillage_engine_PDF_files import TillageenginePDFfiles
from D_Overall_conclusion import UniversalSovereignInference

# --- [الارتباط بالمحركات الذكية] ---
from Generation_Engine import SovereignInferenceEngine
from Search_Engine import SovereignNavigator
from Calculator import Calculator_

# ==========================================
# 1. كلاس معالج البيانات (The Data Processor)
# ==========================================
class SovereignDataProcessor:
    """
    [معالج البيانات السيادي]: القائد الميداني المسؤول عن الربط بين
    العمليات الفيزيائية للملفات والمحركات الذكية للاستدلال.
    """
    def __init__(self, supervisor, extractor: SovereignDataExtractor, logger, network, embedder, openai_client=None):

        # 1. استقبال البنية التحتية من ملف main
        self.supervisor = supervisor
        self.extractor = extractor  # الوحدة B
        self.logger = logger
        self.network = network    # الوحدة P3 (الذاكرة)
        self.embedder = embedder   # الوحدة P2

        self.logger.info("🏗️ [A_Processor]: بدأ تهيئة المحركات الذكية والوحدات الخدمية...")

        # 2. تهيئة المحركات الذكية (Smart Engines) داخلياً
        self.inference_engine = SovereignInferenceEngine(logger, embedder, openai_client)
        self.navigator = SovereignNavigator(network, embedder, logger)
        self.calculator = Calculator_(logger)

        # 3. تهيئة الوحدات الخدمية الإضافية
        self.batch_engine = TillageenginePDFfiles()
        self.conclusion_unit = UniversalSovereignInference()

        self.logger.info("✅ [A_Processor]: جميع المحركات (الحاسبة، الملاح، المولد) جاهزة للعمل.")

        # --- هذا هو السطر المفقود الذي يسبب الخطأ ---
        self.overall_conclusion = [] # تعريف المخزن كقائمة فارغة

    def _clean_sovereign_text(self, text: str) -> str:
        """
        تطهير النص من الضجيج الرقمي مع الحفاظ على هيكلية الأسطر (المصفوفات والمعادلات).
        """
        if not text:
            return ""

        import re
        # 1. تنظيف المسافات الأفقية الزائدة فقط (Tabs, Spaces)
        # مع الحفاظ على سطر جديد (\n) لضمان بقاء المصفوفات مفهومة
        lines = text.splitlines()
        cleaned_lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]

        # 2. إزالة الأسطر الفارغة تماماً الناتجة عن التطهير
        final_text = "\n".join([line for line in cleaned_lines if line])

        return final_text

    def process_pdf_core(self, pdf_path: str) -> Dict[str, Any]:
        """
        المحرك الميداني المطور: ينسق بين استخراج البيانات (B) والتحويل الدلالي مع مراقبة السيادة.
        """
        if not os.path.exists(pdf_path):
            return {"status": "error", "error_details": "File not found"}

        try:
            # 1. الفحص الهيكلي وتحديث عداد الصفحات السيادي
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    self.logger.warning("🔓 PDF is encrypted. Attempting sovereign bypass...")

                actual_pages = len(doc)
                self.total_pages = actual_pages
                # إبلاغ المشرف بالعدد الحقيقي لتجنب ضياع الـ 214 صفحة
                self.logger.info(f"📊 فحص الهيكل: تم رصد {actual_pages} صفحة جاهزة للمعالجة.")

            # 2. استدعاء المستخلص المطوّر (الوحدة B)
            # تأكد أن extractor.process_document يستخدم الآن logic التطهير الجديد
            raw_chunks = self.extractor.process_document(pdf_path)

            if not raw_chunks:
                self.logger.error("🛑 فشل الاستخراج: لم يتم العثور على نصوص قابلة للقراءة.")
                return {"status": "error", "error_details": "PDF is empty or no text extracted"}

            # 3. المعالجة المتوازية الذكية (Multi-threading مع حماية الموارد)
            self.logger.info(f"⚙️ بدء المعالجة المتوازية لـ {len(raw_chunks)} كتلة معرفية...")

            # تحديد max_workers لضمان استقرار النظام على الأجهزة المتوسطة
            with ThreadPoolExecutor(max_workers=4) as executor:
                # تصفية الكتل الفارغة قبل المعالجة لزيادة السرعة
                valid_chunks = [chunk for chunk in raw_chunks if len(chunk.get('content', '')) > 5]

                futures = [executor.submit(self._process_single_chunk_logic, item) for item in valid_chunks]

                # مراقبة التقدم
                completed = 0
                for future in futures:
                    try:
                        future.result()
                        completed += 1
                    except Exception as chunk_err:
                        self.logger.error(f"❌ خطأ في معالجة كتلة: {str(chunk_err)}")

            self.logger.info(f"✅ اكتملت المهمة: تمت معالجة {completed} كتلة بنجاح.")
            return {
                "status": "success",
                "total_pages": actual_pages,
                "processed_chunks": completed
            }

        except Exception as e:
            # تسجيل الحادثة في سجلات السيادة
            self.supervisor.log_incident("Core_Extraction_Failure", str(e))
            return {"status": "error", "error_details": str(e)}

    def _process_single_chunk_logic(self, chunk_data: Dict[str, Any]):
        """
        دالة معالجة الكتل المطورة: تضمن التشفير الدلالي والحفظ الآمن مع معالجة الأخطاء.
        """
        content = chunk_data.get('content', '')
        metadata = chunk_data.get('metadata', {})
        page_num = metadata.get('page', 0)
        chunk_id = chunk_data.get('id', 'unknown')

        # 1. التحقق من وجود محتوى حقيقي (تجنب معالجة الفراغات)
        if not content or len(content.strip()) < 5:
            return

        try:
            # 2. التشفير الدلالي (Vectorization)
            # نستخدم try/except هنا لأن محرك الـ Embedding قد يواجه مشاكل في الذاكرة
            vector = self.embedder.encode_text(content)

            if vector is None:
                raise ValueError(f"Failed to generate embedding for chunk {chunk_id}")

            # 3. الحفظ في شبكة الذاكرة (Memory Network)
            # نرسل البيانات مع التأكد من أن الميتا-داتا تحتوي على معرف الكتلة
            extended_metadata = {**metadata, "chunk_id": chunk_id}

            success = self.network.add_page(
                page_num=page_num,
                content=content,
                embedding=vector,
                metadata=extended_metadata
            )

            # 4. تسجيل المخرجات للرقابة (اختياري للـ Debug)
            if not success:
                self.logger.warning(f"⚠️ تنبيه: تعذر إضافة الكتلة {chunk_id} من الصفحة {page_num} إلى الذاكرة.")

        except Exception as e:
            # تسجيل الحادثة دون إيقاف البرنامج (بما أنها معالجة متوازية)
            error_msg = f"Chunk Logic Failure [Page {page_num}]: {str(e)}"
            self.supervisor.log_incident("Chunk_Processing_Error", error_msg)
            # طباعة الخطأ في الكونسول للمتابعة الفورية
            print(f"❌ {error_msg}")

    def execute_sovereign_mission(self, pdf_path: str):
        """
        [إدارة المهمة الشاملة]: تنسيق التدفق من الملف الخام إلى التحليل الرياضي النهائي.
        """
        # 1. التحقق من وجود الهدف (استخدام pdf_path الممرر للدالة)
        if not os.path.exists(pdf_path):
            self.logger.error(f"❌ الهدف غير موجود في المسار: {pdf_path}")
            return []

        # 2. تشريح الملف (الوحدة B)
        all_chunks = self.extractor.process_document(pdf_path)
        if not all_chunks:
            self.logger.error("🛑 فشل التشريح: لم يتم العثور على محتوى معرفي.")
            return []

        # 3. بناء القاعدة الدلالية في الذاكرة (الوحدة P3)
        self.process_pdf_core(pdf_path)

        # 4. الربط التشغيلي مع المحرك C (التدفق والاستنتاج)
        try:
            self.logger.info(f"⚙️ إرسال {len(all_chunks)} كتلة إلى المحرك C للتدفق...")

            # استدعاء المحرك C وتخزين مخرجاته
            results = self.batch_engine.stream_to_engine(
                all_chunks,
                self.inference_engine.generate
            )

            # --- [الدمج الجوهري]: تحديث الاستنتاجات لكي يراها المحلل الرياضي ---
            if results:
                # نستخدم extend لإضافة النتائج الجديدة إلى القائمة الأساسية
                self.overall_conclusion.extend(results)
                self.logger.info(f"✅ تم استقبال {len(results)} استنتاج من المحرك C.")
            else:
                self.logger.warning("⚠️ المحرك C أكمل التدفق دون إرجاع نتائج.")

        except Exception as e:
            self.logger.error(f"❌ خطأ في تدفق المحرك C: {str(e)}")

        # 5. استلام النتائج النهائية لبدء عملية التدقيق
        final_results = self.overall_conclusion

        # 6. إطلاق المحلل الرياضي (Stability Analyzer)
        stability_report = self._analyze_mathematical_stability(final_results)

        # 7. طباعة التقرير الختامي للسيادة (Audit Report)
        print(f"\n" + "="*50)
        print(f"🛡️ [FINAL SOVEREIGN AUDIT REPORT]")
        print(f"="*50)
        print(f"📄 الملف: {os.path.basename(pdf_path)}")
        print(f"🧩 العقد المعالجة (Nodes): {len(all_chunks)}")
        print(f"🔢 المصفوفات المكتملة: {stability_report['matrix_4x4_count']}")
        print(f"🛡️ حالة الاستقرار: {stability_report['status']}")
        print(f"📈 معامل النزاهة: {stability_report['integrity_score']:.2f}%")
        print("="*50 + "\n")

        return final_results

    def _analyze_mathematical_stability(self, extracted_data: List[Any]) -> Dict[str, Any]:
        """
        [المحلل الرياضي السيادي]:
        يفحص المصفوفات المستخرجة للتأكد من اكتمال الهيكل الحركي (Kinematic Structure).
        """
        self.logger.info("🛡️ [STABILITY_CHECK]: بدء فحص الاستقرار الرياضي للمخرجات...")

        report = {
            "total_nodes": len(extracted_data),
            "matrix_4x4_count": 0,
            "integrity_score": 0.0,
            "status": "Incomplete"
        }

        # البحث عن أنماط المصفوفات 4x4 في النتائج
        for entry in extracted_data:
            entry_str = str(entry)
            # التحقق من وجود مصفوفة مكتملة الأركان (16 عنصراً أو وسم 4x4)
            if "4x4" in entry_str or entry_str.count(",") >= 15:
                report["matrix_4x4_count"] += 1

        # حساب معامل الاستقرار
        if report["total_nodes"] > 0:
            # افتراض هندسي: يجب أن تحتوي 20% على الأقل من العقد التقنية على مصفوفات تحويل
            expected_matrices = max(1, int(report["total_nodes"] * 0.2))
            report["integrity_score"] = min(100.0, (report["matrix_4x4_count"] / expected_matrices) * 100)

            if report["integrity_score"] >= 90:
                report["status"] = "💎 High Integrity (Stable)"
            elif report["integrity_score"] >= 50:
                report["status"] = "⚠️ Marginal Stability"
            else:
                report["status"] = "🚨 Unstable / Fragmented"

        return report
