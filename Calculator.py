# Calculator.py

import numpy as np
import math
from typing import List, Dict, Any

class Calculator_:
    """
    [الحاسب الآلي]: المحرك الرياضي الصلب المسؤول عن الحسابات الكينماتيكية،
    الدقة الميكانيكية، وتخطيط المسارات الانسيابية.
    """
    def __init__(self, logger):
        self.logger = logger

    def calculate_robot_precision(self, bits: int, range_max: float) -> float:
        """
        حساب الدقة (Precision) بناءً على عدد بتات التشفير والمدى الحركي.
        تمت إضافة التحقق من المنطق الرياضي لضمان عدم وجود قيم سالبة أو صفرية.
        """
        try:
            if bits <= 0:
                self.logger.warning("⚠️ [CALCULATOR]: Bits must be greater than 0. Resetting to 1.")
                bits = 1

            # حساب الدقة: المدى تقسيم 2 أس عدد البتات
            resolution = range_max / (2**bits)
            return float(resolution)
        except Exception as e:
            self.logger.error(f"❌ [CALCULATOR]: Precision Calculation Error: {e}")
            return 0.0

    def compute_joint_velocity(self, delta_theta: float, delta_t: float) -> float:
        """
        حساب السرعة الزاوية للمفاصل (Angular Velocity).
        """
        if delta_t <= 0: return 0.0
        return delta_theta / delta_t

    def verify_dh_matrix(self, matrix: List[List[float]]) -> bool:
        """
        التحقق من صحة مصفوفة D-H (Denvait-Hartenberg).
        تتأكد من أن المصفوفة مربعة (4x4) كما هو معيار الروبوتات.
        """
        if len(matrix) == 4 and all(len(row) == 4 for row in matrix):
            return True
        return False

    def calculate_joint_transformation(self, theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        """
        [المعالج الكينماتيكي]: حساب مصفوفة التحويل 4x4 باستخدام معاملات D-H القياسية.
        تستخدم لربط الإحداثيات بين المفصل (i-1) والمفصل (i).
        """
        import math # ضمان الاستيراد داخل الدالة أو في بداية الملف

        # حساب الجيب والتمام للزوايا (Theta و Alpha)
        c_th = math.cos(theta)
        s_th = math.sin(theta)
        c_al = math.cos(alpha)
        s_al = math.sin(alpha)

        # المصفوفة السيادية للتحويل (D-H Standard Matrix)
        # السطر الأول: تمثيل الدوران حول Z ثم الإزاحة على X
        # السطر الثاني: تمثيل الدوران حول X (Alpha)
        return np.array([
            [c_th, -s_th * c_al,  s_th * s_al, a * c_th],
            [s_th,  c_th * c_al, -c_th * s_al, a * s_th],
            [0,     s_al,         c_al,         d     ],
            [0,     0,            0,            1     ]
        ])

    def calculate_sovereign_resolution(self, motor_steps: int, gear_ratio: float, arm_length: float) -> Dict[str, Any]:
        """
        [الحاسبة السيادية]: حساب الدقة النظرية (Theoretical Resolution) بدقة ميكرونية.
        تربط بين حركة الموتور، نسبة التخفيض، وطول الذراع الروبوتي.
        """
        try:
            # 1. فحص سلامة المدخلات لمنع الانهيار (Safety Guard)
            if motor_steps <= 0 or gear_ratio <= 0 or arm_length <= 0:
                raise ValueError("يجب أن تكون قيم الخطوات، التروس، والذراع أكبر من الصفر.")

            # 2. حساب الزاوية لكل خطوة للموتور (Step Angle in Radians)
            step_angle = (2 * math.pi) / motor_steps

            # 3. حساب زاوية الخرج بعد صندوق التروس (Output Resolution)
            output_angle = step_angle / gear_ratio

            # 4. حساب الإزاحة الخطية عند نهاية الذراع (Arc Resolution: S = r * θ)
            # نفترض أن arm_length بالمتر، وسنقوم بحسابها بدقة ميكرونية
            linear_resolution_m = arm_length * output_angle

            # 5. بناء التقرير الحسابي النهائي
            return {
                "status": "SUCCESS",
                "angular_resolution_rad": round(output_angle, 10),
                "angular_resolution_deg": round(math.degrees(output_angle), 6),
                "linear_resolution_mm": round(linear_resolution_m * 1000, 6),
                "linear_resolution_micron": round(linear_resolution_m * 1000000, 2),
                "metadata": {
                    "motor_steps": motor_steps,
                    "effective_gear_ratio": gear_ratio
                }
            }

        except Exception as e:
            self.logger.error(f"❌ [CALCULATOR_ERROR]: فشل الحساب الهندسي: {str(e)}")
            return {"status": "ERROR", "message": str(e)}

    def trajectory_cubic_spline(self, t: List[float], q: List[float]) -> Dict[str, Any]:
        """
        [منعم الحركة السيادي]: توليد مسار انسيابي من الدرجة الثالثة (Cubic Spline).
        يضمن استمرارية الموقع والسرعة لمنع الصدمات الميكانيكية (Jerks).
        """
        try:
            if len(t) < 2 or len(q) < 2:
                return {"status": "ERROR", "message": "نقاط غير كافية لتوليد المسار"}

            # تحديد نقطة البداية والنهاية
            t0, tf = t[0], t[-1]
            q0, qf = q[0], q[-1]

            T = tf - t0
            if T <= 0:
                raise ValueError("يجب أن يكون زمن النهاية أكبر من زمن البداية")

            # حساب المعاملات (Coefficients) بفرض سرعة البداية والنهاية = 0
            # المعادلات السيادية للمسار: q(t) = a0 + a1τ + a2τ² + a3τ³ حيث τ هو الزمن النسبي
            a0 = q0
            a1 = 0.0  # السرعة الابتدائية
            a2 = 3 * (qf - q0) / (T**2)
            a3 = -2 * (qf - q0) / (T**3)

            # توليد نقاط المسار (السرعة والتسارع أيضاً لأهميتهما في الرقابة)
            time_steps = np.linspace(0, T, 50)
            position_path = []
            velocity_path = []
            acceleration_path = []

            for tau in time_steps:
                # الموقع
                pos = a0 + a1*tau + a2*(tau**2) + a3*(tau**3)
                # السرعة (المشتقة الأولى)
                vel = a1 + 2*a2*tau + 3*a3*(tau**2)
                # التسارع (المشتقة الثانية)
                acc = 2*a2 + 6*a3*tau

                position_path.append(round(pos, 6))
                velocity_path.append(round(vel, 6))
                acceleration_path.append(round(acc, 6))

            self.logger.info(f"✅ [TRAJECTORY]: تم توليد مسار انسيابي لـ {len(position_path)} نقطة.")

            return {
                "status": "SUCCESS",
                "time_axis": (time_steps + t0).tolist(), # العودة للزمن المطلق
                "position_path": position_path,
                "velocity_path": velocity_path,
                "acceleration_path": acceleration_path,
                "metadata": {"total_time": T, "start_pos": q0, "end_pos": qf}
            }

        except Exception as e:
            self.logger.error(f"🚨 [TRAJECTORY_ERROR]: فشل توليد المسار: {e}")
            return {"status": "ERROR", "message": str(e)}
