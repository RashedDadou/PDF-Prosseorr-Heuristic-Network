# main.py

import os
from dotenv import load_dotenv

# 1. استيراد الوحدة الخدمية الرئيسية (الرأس)
from A_pdf_processor import SovereignDataProcessor

# 2. استيراد مستودع الأسلحة (Infrastructure)
from P1_sovereign_utils import SovereignSupervisorySystem
from P2_embedding_logic import EmbeddingManager
from P3_memory import PDFpageCacheNetwork
from B_data_extractor import SovereignDataExtractor

# تحميل إعدادات البيئة (API Keys, URLs)
load_dotenv()

def run_sovereign_mission():
    """
    [إطلاق المهمة السيادية]:
    ربط الوحدات بنظام الدفعات الجديد والأرشفة التلقائية.
    """
    # --- المرحلة 1: تهيئة مستودع الأسلحة (Infrastructure) ---
    supervisor = SovereignSupervisorySystem(name="SOVEREIGN_CORE")
    logger = supervisor.logger

    # تهيئة المحرك الدلالي (مع تمرير الـ logger)
    embedder = EmbeddingManager(logger=logger)

    # تهيئة شبكة الذاكرة (التي تدعم الآن نظام الـ 20 صفحة)
    network_memory = PDFpageCacheNetwork(supervisor)

    # تهيئة مستخلص البيانات
    extractor = SovereignDataExtractor(chunk_size=500, overlap=50)

    # --- المرحلة 2: بناء المعالج الرئيسي ---
    commander = SovereignDataProcessor(
        supervisor=supervisor,
        extractor=extractor,
        logger=logger,
        network=network_memory, # تمرير الكائن بالكامل
        embedder=embedder
    )

    # --- المرحلة 3: التنفيذ والدمج النهائي ---
    pdf_target = "ROBOTICS.pdf"

    if os.path.exists(pdf_target):
        try:
            logger.info(f"🚀 بدء معالجة الهدف السيادي: {pdf_target}")

            # تنفيذ المهمة
            final_results = commander.execute_sovereign_mission(pdf_target)

            # --- التحديث الأهم: الدمج النهائي للشبكة ---
            # هذا السطر يضمن أرشفة آخر صفحات الملف ودمج الخرائط
            network_memory.finalize_network()

            if final_results:
                logger.info(f"🏁 تمت المهمة بنجاح. وقت التشغيل: {supervisor.get_uptime()}")

        except Exception as e:
            logger.critical(f"🚨 خطأ في العمليات الميدانية: {str(e)}")
        finally:
            supervisor.shutdown_sequence()
    else:
        logger.error(f"❌ الملف {pdf_target} مفقود.")

if __name__ == "__main__":
    # --- المرحلة 1: تهيئة مستودع الأسلحة (Infrastructure) ---
    supervisor = SovereignSupervisorySystem(name="SOVEREIGN_CORE")
    logger = supervisor.logger

    # 2. تهيئة الأدوات الأساسية (بشكل صحيح)
    embedder = EmbeddingManager(logger=logger)
    network_memory = PDFpageCacheNetwork(supervisor) # الكائن كامل هنا
    extractor = SovereignDataExtractor(chunk_size=400, overlap=50)

    # --- المرحلة 2: بناء القائد الميداني وتزويده بالأسلحة ---
    commander = SovereignDataProcessor(
        supervisor=supervisor,
        extractor=extractor,
        logger=logger,
        network=network_memory, # التصحيح: نمرر الكائن بالكامل وليس الدالة فقط
        embedder=embedder
    )

    # --- المرحلة 3: تحديد الهدف والتنفيذ ---
    pdf_target = "ROBOTICS.pdf"

    if os.path.exists(pdf_target):
        logger.info(f"🚀 تم رصد الهدف: {pdf_target} - إطلاق المهمة السيادية...")

        try:
            # 1. تنفيذ المهمة (القائد الميداني يعالج 214 صفحة)
            final_results = commander.execute_sovereign_mission(pdf_target)

            # 2. تأمين آخر دفعة من الذاكرة (لا تترك أثراً خلفك)
            network_memory.finalize_network()

            # 3. عرض التقارير
            print("\n" + "="*50)
            print("📊 تقرير الاستدلال الهندسي النهائي:")
            print("="*50)

            if final_results:
                # --- [محلل الاستقرار اللحظي] ---
                valid_matrices = [r for r in final_results if "4x4" in str(r)]
                stability_index = (len(valid_matrices) / len(final_results)) * 100

                print("\n" + "🛡️ [SOVEREIGN STABILITY AUDIT]:")
                print("-" * 40)
                print(f"📊 معامل الاستقرار الهيكلي: {stability_index:.2f}%")
                print(f"✅ تم تأمين {len(valid_matrices)} مصفوفة حركية بنجاح.")

                if stability_index > 90:
                    print("💎 الحالة: استقرار رياضي ممتاز (High Integrity)")
                else:
                    print("⚠️ الحالة: استقرار متوسط - يوجد فجوات في تسلسل البيانات")

                print("-" * 40)
                # عرض الأفكار المستخلصة
                for i, insight in enumerate(final_results, 1):
                    print(f"{i}. {insight}")

                logger.info(f"🏁 تمت المهمة بنجاح. وقت التشغيل: {supervisor.get_uptime()}")

            else:
                print("❌ لم يتم استخراج بيانات كافية لتحليل الاستقرار.")
                logger.warning("⚠️ لم يتم استخلاص نتائج واضحة من الملف.")

        except Exception as e:
            logger.critical(f"🚨 خطأ ميداني أثناء التشغيل: {str(e)}")
