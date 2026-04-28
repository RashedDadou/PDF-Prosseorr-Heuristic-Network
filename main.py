# main.py

import os
import sys
import traceback
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# 1. استيراد الوحدة الخدمية الرئيسية
from A_pdf_processor import SovereignDataProcessor

# 2. استيراد مستودع الأسلحة (Infrastructure)
from P1_sovereign_utils import SovereignSupervisorySystem
from P2_embedding_logic import EmbeddingManager
from P3_memory import PDFpageCacheNetwork
from B_data_extractor import SovereignDataExtractor

# تحميل إعدادات البيئة (API Keys, URLs)
load_dotenv()

# ==========================================
# تعريفات الإعدادات (Configuration Constants)
# ==========================================
CONFIG = {
    "pdf_target": os.getenv("PDF_TARGET", "ROBOTICS.pdf"),
    "chunk_size": int(os.getenv("CHUNK_SIZE", "400")),
    "overlap": int(os.getenv("OVERLAP", "50")),
    "batch_size": int(os.getenv("BATCH_SIZE", "20")),
    "stability_threshold": float(os.getenv("STABILITY_THRESHOLD", "90")),
    "archive_dir": os.getenv("ARCHIVE_DIR", "sovereign_knowledge_base"),
}

# ==========================================
# دوال المساعدة والتكامل
# ==========================================
def _create_infrastructure(logger) -> tuple:
    """
    [تهيئة البنية التحتية]:
    إنشاء جميع المكونات الأساسية بإعدادات موحدة.
    
    Args:
        logger: نظام تسجيل السجلات
        
    Returns:
        tuple: (supervisor, embedder, network_memory, extractor, commander)
    """
    logger.info("🏗️ [INFRASTRUCTURE]: جاري بناء البنية التحتية...")
    
    # تهيئة المحرك الدلالي
    embedder = EmbeddingManager(logger=logger)
    
    # تهيئة شبكة الذاكرة
    network_memory = PDFpageCacheNetwork(
        supervisor=None,  # سيتم تعيينه لاحقاً
        batch_size=CONFIG["batch_size"],
        archive_dir=CONFIG["archive_dir"]
    )
    
    # تهيئة مستخلص البيانات
    extractor = SovereignDataExtractor(
        chunk_size=CONFIG["chunk_size"],
        overlap=CONFIG["overlap"]
    )
    
    logger.info("✅ [INFRASTRUCTURE]: اكتملت البنية التحتية بنجاح.")
    return embedder, network_memory, extractor


def _create_commander(supervisor, embedder, network_memory, extractor, logger) -> SovereignDataProcessor:
    """
    [بناء القائد الميداني]:
    إنشاء معالج البيانات الرئيسي مع جميع المكونات.
    
    Args:
        supervisor: نظام الإشراف
        embedder: مدير التشفير المتجهي
        network_memory: شبكة الذاكرة
        extractor: مستخلص البيانات
        logger: نظام التسجيل
        
    Returns:
        SovereignDataProcessor: القائد الميداني المجهز
    """
    logger.info("⚙️ [COMMANDER]: جاري تأسيس القائد الميداني...")
    
    commander = SovereignDataProcessor(
        supervisor=supervisor,
        extractor=extractor,
        logger=logger,
        network=network_memory,
        embedder=embedder
    )
    
    logger.info("✅ [COMMANDER]: القائد الميداني جاهز للعمل.")
    return commander


def _print_stability_report(
    final_results: List[Any],
    logger,
    valid_matrix_pattern: str = "4x4"
) -> float:
    """
    [تقرير الاستقرار]:
    عرض تقرير شامل عن استقرار المعالجة.
    
    Args:
        final_results: النتائج النهائية من المحرك
        logger: نظام التسجيل
        valid_matrix_pattern: النمط المستخدم للتحقق من المصفوفات الصحيحة
        
    Returns:
        float: معامل الاستقرار (0-100)
    """
    if not final_results:
        logger.warning("⚠️ لا توجد نتائج للتقرير.")
        return 0.0
    
    # تحديد المصفوفات الصحيحة
    valid_matrices = [r for r in final_results if valid_matrix_pattern in str(r)]
    stability_index = (len(valid_matrices) / len(final_results)) * 100 if final_results else 0.0
    
    # طباعة التقرير باستخدام Logger فقط
    logger.info("=" * 50)
    logger.info("📊 تقرير الاستدلال الهندسي النهائي")
    logger.info("=" * 50)
    logger.info(f"🛡️ [SOVEREIGN STABILITY AUDIT]")
    logger.info("-" * 40)
    logger.info(f"📊 معامل الاستقرار الهيكلي: {stability_index:.2f}%")
    logger.info(f"✅ تم تأمين {len(valid_matrices)} مصفوفة حركية بنجاح.")
    
    # تقييم الحالة
    if stability_index >= CONFIG["stability_threshold"]:
        logger.info("💎 الحالة: استقرار رياضي ممتاز (High Integrity)")
    else:
        logger.info("⚠️ الحالة: استقرار متوسط - يوجد فجوات في تسلسل البيانات")
    
    logger.info("-" * 40)
    
    # عرض الأفكار المستخلصة (أول 10 فقط لتجنب الفوضى)
    logger.info(f"📋 عرض أول {min(10, len(final_results))} من {len(final_results)} استنتاج:")
    for i, insight in enumerate(final_results[:10], 1):
        logger.info(f"   {i}. {insight}")
    
    if len(final_results) > 10:
        logger.info(f"   ... و{len(final_results) - 10} استنتاجات أخرى")
    
    return stability_index


def run_sovereign_mission(pdf_target: Optional[str] = None) -> bool:
    """
    [إطلاق المهمة السيادية]:
    ربط الوحدات بنظام الدفعات الجديد والأرشفة التلقائية.
    
    Args:
        pdf_target: مسار الملف المراد معالجته (يس��خدم الإعدادات إذا لم يتم تحديده)
        
    Returns:
        bool: True إذا نجحت المهمة، False إذا فشلت
    """
    # تحديد الملف المستهدف
    pdf_file = pdf_target or CONFIG["pdf_target"]
    
    # --- المرحلة 1: تهيئة نظام الإشراف ---
    supervisor = SovereignSupervisorySystem(name="SOVEREIGN_CORE")
    logger = supervisor.logger
    
    logger.info(f"🚀 بدء المهمة السيادية في: {pd_file if pdf_target else '(من الإعدادات)'}")
    logger.info(f"⏱️ الوقت: {__import__('datetime').datetime.now().isoformat()}")
    
    try:
        # التحقق من وجود الملف
        if not os.path.exists(pdf_file):
            logger.error(f"❌ الملف {pdf_file} غير موجود.")
            logger.error(f"📂 المسار المطلوب: {os.path.abspath(pdf_file)}")
            return False
        
        # --- المرحلة 2: بناء البنية التحتية ---
        embedder, network_memory, extractor = _create_infrastructure(logger)
        
        # --- المرحلة 3: بناء القائد الميداني ---
        commander = _create_commander(supervisor, embedder, network_memory, extractor, logger)
        
        # --- المرحلة 4: التنفيذ ---
        logger.info(f"⚙️ جاري معالجة الملف: {os.path.basename(pdf_file)}")
        final_results = commander.execute_sovereign_mission(pdf_file)
        
        # --- المرحلة 5: الدمج النهائي للشبكة ---
        logger.info("🔗 جاري دمج الشبكات الاستدلالية...")
        network_memory.finalize_network()
        
        # --- المرحلة 6: عرض التقارير ---
        if final_results:
            _print_stability_report(final_results, logger)
            logger.info(f"🏁 تمت المهمة بنجاح. وقت التشغيل: {supervisor.get_uptime()}")
            return True
        else:
            logger.warning("⚠️ لم يتم استخراج بيانات كافية للتحليل.")
            return False
            
    except Exception as e:
        logger.critical(f"🚨 خطأ ميداني أثناء التشغيل: {str(e)}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        return False
        
    finally:
        supervisor.shutdown_sequence()


def print_usage():
    """طباعة معلومات الاستخدام."""
    print("""
    ╔════════════════════════════════════════════╗
    ║     🛡️ SOVEREIGN PDF PROCESSOR 🛡️          ║
    ╚════════════════════════════════════════════╝
    
    الاستخدام:
    ─────────
    python main.py [pdf_file]
    
    الأمثلة:
    ───────
    python main.py                          # استخدام إعدادات البيئة
    python main.py ROBOTICS.pdf             # معالجة ملف محدد
    
    متغيرات البيئة:
    ──────────────
    PDF_TARGET          الملف المراد معالجته (افتراضي: ROBOTICS.pdf)
    CHUNK_SIZE          حجم القطع النصية (افتراضي: 400)
    OVERLAP             التداخل بين القطع (افتراضي: 50)
    BATCH_SIZE          حجم الدفعات (افتراضي: 20)
    STABILITY_THRESHOLD حد الاستقرار (افتراضي: 90)
    ARCHIVE_DIR         مجلد الأرشفة (افتراضي: sovereign_knowledge_base)
    """)


if __name__ == "__main__":
    # معالجة معاملات سطر الأوامر
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help", "?"]:
            print_usage()
            sys.exit(0)
        pdf_target = sys.argv[1]
    else:
        pdf_target = None
    
    # تنفيذ المهمة
    success = run_sovereign_mission(pdf_target)
    sys.exit(0 if success else 1)