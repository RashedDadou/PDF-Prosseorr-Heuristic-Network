# P1_sovereign_utils.py

import logging
import sys
import time
import functools
from typing import Protocol, Any, Union, Optional, Callable, runtime_checkable

# ==========================================
# 1. مزخرف قياس الأداء (Performance Decorator)
# ==========================================
def profile_performance(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    [مزخرف قياس الأداء السيادي]:
    يقيس زمن تنفيذ الدالة ويقوم بتسجيله عبر الـ Logger المتاح في الكلاس.
    تم وضعه خارج الكلاس لضمان توافق الأنواع وإسكات تنبيهات Pylance.
    """
    @functools.wraps(func)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            return func(self, *args, **kwargs)
        finally:
            duration = time.time() - start_time
            logger = getattr(self, 'logger', None)

            func_name = getattr(func, '__name__', str(func))
            msg = f"⏱️ العملية [{func_name}] استغرقت: {duration:.4f} ثانية"

            if logger and hasattr(logger, 'info'):
                logger.info(msg)
            else:
                print(msg)
    return wrapper

# ==========================================
# 2. بروتوكول الرقابة الموحد (Logger Protocol)
# ==========================================
@runtime_checkable
class LoggerProtocol(Protocol):
    """
    [بروتوكول الرقابة الموحد V2.0]:
    يضمن التوافق التام بين المحرك السيادي، نظام إدارة العمل، والمزخرفات.
    """
    level: int
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def error(self, msg: str, *args: Any, exc_info: Optional[Union[bool, Any]] = None, **kwargs: Any) -> None: ...
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def critical(self, msg: str, *args: Any, exc_info: Optional[Union[bool, Any]] = True, **kwargs: Any) -> None: ...
    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def isEnabledFor(self, level: int) -> bool: ...

# ==========================================
# 3. نظام الرقابة السيادي (Supervisory System)
# ==========================================
class SovereignSupervisorySystem:
    """
    [نظام الرقابة السيادي]: المسؤول عن تتبع العمليات وإدارة السجلات ورصد الحوادث.
    """
    def __init__(self, name: str = "SOVEREIGN_CORE"):
        self.logger = self._setup_logger(name)
        self.start_time = time.time()
        self.incident_log = []
        self.level = self.logger.level

    def _setup_logger(self, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    # تنفيذ دوال البروتوكول لضمان التوافق
    def info(self, msg: str, *args, **kwargs): self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs): self.logger.warning(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs): self.logger.debug(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self.log_incident("General_Error", msg)
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self.log_incident("CRITICAL_FAILURE", msg)
        self.logger.critical(msg, *args, **kwargs)

    def log(self, level: int, msg: str, *args, **kwargs): self.logger.log(level, msg, *args, **kwargs)

    def isEnabledFor(self, level: int) -> bool: return self.logger.isEnabledFor(level)

    def log_incident(self, category: str, details: str):
        """توثيق سيادي للحوادث التقنية مع طابع زمني دقيق."""
        timestamp = time.strftime('%H:%M:%S')
        incident = {
            "time": timestamp,
            "category": category.upper(),
            "details": details
        }
        self.incident_log.append(incident)
        # لا نضع self.logger.error هنا لتجنب التكرار اللانهائي (Recursion)
        # إذا تم استدعاؤها من داخل دالة error

    def get_audit_summary(self):
        """توليد ملخص نهائي لعملية الرقابة."""
        duration = time.time() - self.start_time
        return {
            "uptime": f"{duration:.2f}s",
            "total_incidents": len(self.incident_log),
            "status": "SECURE" if not self.incident_log else "AUDIT_REQUIRED"
        }

    def get_uptime(self) -> str:
        """حساب مدة تشغيل النظام بتنسيق قراءة بشري."""
        duration = time.time() - self.start_time
        if duration < 60:
            return f"{duration:.2f}s"
        minutes, seconds = divmod(duration, 60)
        return f"{int(minutes)}m {seconds:.2f}s"

    def shutdown_sequence(self):
        """إجراءات الإغلاق الآمن مع توثيق الحوادث في الأرشيف."""
        uptime = self.get_uptime()
        self.logger.info("="*50)
        self.logger.info(f"🏁 [SHUTDOWN]: إغلاق المحرك السيادي... مدة التشغيل: {uptime}")

        if self.incident_log:
            self.logger.warning(f"⚠️ تم رصد {len(self.incident_log)} حادثة تقنية أثناء المهمة.")

            # اختياري: حفظ الحوادث في ملف خارجي للمراجعة
            try:
                with open("sovereign_incidents_audit.log", "a", encoding="utf-8") as f:
                    f.write(f"\n--- SESSION {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                    for incident in self.incident_log:
                        # بما أن incident أصبح قاموساً في التطوير السابق:
                        f.write(f"[{incident['time']}] [{incident['category']}]: {incident['details']}\n")
                self.logger.info("💾 تم حفظ سجل الحوادث في sovereign_incidents_audit.log")
            except Exception as e:
                self.logger.error(f"❌ فشل حفظ سجل الحوادث: {str(e)}")

        self.logger.info("✅ النظام الآن في وضع الاستعداد. وداعاً.")
        self.logger.info("="*50)
