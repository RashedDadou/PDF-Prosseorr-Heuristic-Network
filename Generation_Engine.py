# Generation_Engine.py

import re
from typing import List, Dict, Any, Tuple

class SovereignInferenceEngine:
    """
    [المحرك الاستدلالي]: المسؤول عن تحليل الأنماط الهندسية، استخراج المعادلات الكينماتيكية،
    وتوليد الاستنتاجات التقنية عالية الدقة.
    """
    def __init__(self, logger, embedder, openai_client=None):
        self.logger = logger
        self.embedder = embedder
        self.client = openai_client  # للربط مع OpenAI أو LM Studio

    def extract_kinematic_equations(self, text_content: str) -> List[str]:
        """
        [المجس الرياضي السيادي]: استخراج المعادلات مع محيطها المنطقي
        لضمان عدم فقدان سياق المتغيرات.
        """
        # تحسين الأنماط لتشمل المساحات، الأقواس، والرموز اللاتينية المتقدمة
        math_patterns = [
            r"T\d+\s*=\s*\[[\s\S]*?\]",               # مصفوفات التحويل (بما في ذلك المتعددة الأسطر)
            r"[θφαβγ]\d*\s*=\s*[^.!?\n]+",           # قيم الزوايا والمعادلات المثلثية
            r"J\(q\)\s*=\s*[^.!?\n]+",               # مصفوفات الياكوبي (Jacobian)
            r"[xyz]\s*=\s*f\([θq]\d*\)",             # معادلات التحويل الأمامي
            r"τ\d*\s*=\s*[^.!?\n]+",                 # معادلات العزم والديناميكا
            r"\d+\s*=\s*2\^n\s*Where\s*n\s*=\s*\d+"   # أنماط الذاكرة والتحكم التي ظهرت في تقريرك
        ]

        found_equations = []
        for pattern in math_patterns:
            # استخدام DOTALL للتعامل مع المعادلات التي تمتد لأكثر من سطر
            matches = re.findall(pattern, text_content, re.IGNORECASE | re.DOTALL)
            for m in matches:
                clean_match = " ".join(m.split()) # تنظيف المسافات الزائدة
                if len(clean_match) > 5:
                    found_equations.append(f"📐 [MATH_MODEL]: {clean_match}")

        return list(set(found_equations))

    def analyze_trajectory_logic(self, text_content: str) -> Dict[str, str]:
        """
        [محلل الخوارزميات السيادي]: يربط بين نوع المسار (Path Planning)
        والجملة التقنية التي تشرحه.
        """
        logic_map = {
            "PTP (Point-to-Point)": r"(point-to-point|PTP|joint-space move|trapezoidal velocity)",
            "CP (Continuous Path)": r"(continuous path|CP|arc welding|path control)",
            "Interpolation": r"(linear interpolation|circular interpolation|S-curve|spline)",
            "Control Logic": r"(PID control|feedback loop|control memory|increments)",
            "Obstacle Avoidance": r"(potential field|A\* algorithm|collision-free|path planning)"
        }

        results = {}
        for algo_name, pattern in logic_map.items():
            # البحث عن النمط واستخراج الجملة الكاملة التي ورد فيها (Context-Aware)
            search_pattern = rf"([^.!?\n]*?{pattern}[^.!?\n]*[\.!?])"
            match = re.search(search_pattern, text_content, re.IGNORECASE | re.DOTALL)

            if match:
                # تنظيف الجملة المستخرجة من الفواصل والرموز المزعجة
                context = match.group(1).strip()
                results[algo_name] = f"⚙️ [LOGIC_FOUND]: {context}"

        return results

    def process_high_precision(self, chunk: str) -> Dict[str, Any]:
        """
        [بروتوكول الدقة المجهرية]: المترجم السيادي الذي يحول المعادلات الرياضية
        إلى خوارزميات برمجية جاهزة للتنفيذ.
        """
        # 1. استخراج المعادلات الكينماتيكية باستخدام المحسن الرياضي
        equations = self.extract_kinematic_equations(chunk)

        # 2. تحليل منطق المسار لتحديد نوع الخوارزمية (PTP, CP, etc.)
        trajectory_logic = self.analyze_trajectory_logic(chunk)

        # 3. بناء الـ Prompt السيادي (هندسة الأوامر المتقدمة)
        # جعلنا الأوامر تفرض على النموذج اللغوي تقديم كود Python دقيق
        prompt = (
            f"--- [SOVEREIGN AUDIT DATA] ---\n"
            f"TECHNICAL CONTEXT: {chunk[:2000]}\n\n"
            f"EXTRACTED MATH: {', '.join(equations) if equations else 'None identified'}\n"
            f"IDENTIFIED LOGIC: {trajectory_logic}\n\n"
            f"TASK FOR ROBOTICS ENGINEER:\n"
            f"1. Convert the identified equations into a high-performance Python function.\n"
            f"2. Use NumPy for matrix operations if T-matrices are present.\n"
            f"3. Ensure the code follows Kinematic principles (Forward/Inverse).\n"
            f"4. Provide only the Python block and a brief technical justification."
        )

        self.logger.info(f"⚡ [HIGH_PRECISION]: Analyzing {len(equations)} equations and {len(trajectory_logic)} logic patterns.")

        # 4. استدعاء محرك التوليد (سواء كان محلياً أو عبر API)
        # نمرر الـ Prompt المصمم بعناية
        response, status = self.generate(prompt, "System: Senior Robotics Software Architect")

        # 5. بناء الحزمة المعرفية النهائية
        return {
            "status": status.get("engine_status", "PROCESSED"),
            "equations_found": equations,
            "logic_detected": list(trajectory_logic.keys()),
            "python_implementation": response,
            "complexity_level": "HIGH" if len(equations) > 3 else "STANDARD"
        }

    def generate(self, context: str, user_request: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        [المحلل الاستدلالي المطور]: يقوم بتحليل النص تقنياً واستنتاج الأفكار
        بدلاً من مجرد مطابقة الأنماط الثابتة.
        """
        try:
            # 1. إذا كان العميل (OpenAI/Local LLM) مهيأ، نستخدم الاستدلال العميق
            if self.client:
                return self._ai_inference(context, user_request)

            # 2. إذا كان العمل محلياً (Regex)، نستخدم نظام الأنماط الديناميكية
            technical_insights = []

            # أنماط متنوعة تغطي (الرياضيات، الخوارزميات، الحساسات، والنتائج الهندسية)
            dynamic_patterns = {
                "kinematics": r"(θ|φ|α|β|γ|matrix|transform|joint|link)",
                "algorithms": r"(algorithm|procedure|steps|calculation|computation)",
                "results": r"(yields|results|equals|approx|found to be)",
                "components": r"(sensor|actuator|motor|encoder|controller)"
            }

            sentences = re.split(r'(?<=[.!?])\s+', context)
            seen_insights = set() # لمنع التكرار الممل في التقرير

            for sent in sentences:
                sent_clean = sent.strip()

                # التحقق من وجود أنماط تقنية مع تنويع المصادر
                matches = [label for label, pat in dynamic_patterns.items() if re.search(pat, sent_clean, re.IGNORECASE)]

                # شرط القبول: أن تحتوي الجملة على فكرة تقنية، لم تظهر سابقاً، وطولها مناسب
                if matches and sent_clean not in seen_insights and len(sent_clean) > 35:
                    label_str = f"[{'|'.join(matches).upper()}]"
                    technical_insights.append(f"🔍 {label_str}: {sent_clean}")
                    seen_insights.add(sent_clean)

                # التوقف عند حد معين لضمان جودة التقرير وعدم الحشو
                if len(technical_insights) >= 12: break

            return technical_insights, {"engine_status": "DYNAMIC_LOCAL_ANALYSIS"}

        except Exception as e:
            self.logger.error(f"Error in Sovereign Generation: {e}")
            return [f"Inference interrupted for: {user_request}"], {"engine_status": "ERROR"}

    def _ai_inference(self, context: str, user_request: str) -> Tuple[List[str], Dict[str, Any]]:
        """دالة مساعدة للاستدلال عبر النماذج اللغوية الكبيرة"""
        prompt = f"""
        تحليلي هندسي لمستند روبوتات:
        النص: {context[:2000]}
        المطلوب: استخرج أهم 5 استنتاجات تقنية تتعلق بـ {user_request} بأسلوب مهني دقيق.
        """
        # هنا يتم استدعاء self.client.chat.completions.create ...
        # (يفترض وجود إعدادات العميل مسبقاً)
        return ["AI Insight 1...", "AI Insight 2..."], {"engine_status": "AI_POWERED"}
