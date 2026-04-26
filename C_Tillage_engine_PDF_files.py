# C_Tillage_engine_PDF_files.py

import ast
import operator
import math
from typing import Dict, Any, Union, Sequence

class SafeSovereignEvaluator:
    # 1. تحديد العمليات الحسابية المسموح بها
    ALLOWED_OPERATORS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg
    }

    # 2. تحديد الدوال الرياضية المسموح بها (السيادة الهندسية)
    ALLOWED_FUNCTIONS = {
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'sqrt': math.sqrt,
        'radians': math.radians,
        'pi': math.pi
    }

    def safe_eval(self, node, var_map):
        if isinstance(node, ast.Constant): # للأرقام (بايثون 3.8+)
            return node.value
        elif isinstance(node, ast.Num):    # للأرقام (الإصدارات الأقدم)
            return node.n
        elif isinstance(node, ast.Name):   # للمتغيرات (L1, theta1)
            if node.id in var_map:
                return var_map[node.id]
            elif node.id in self.ALLOWED_FUNCTIONS:
                return self.ALLOWED_FUNCTIONS[node.id]
            raise NameError(f"Undefined variable: {node.id}")

        elif isinstance(node, ast.BinOp):  # (+, -, *, /)
            op_type = type(node.op)
            if op_type in self.ALLOWED_OPERATORS:
                return self.ALLOWED_OPERATORS[op_type](
                    self.safe_eval(node.left, var_map),
                    self.safe_eval(node.right, var_map)
                )
            raise TypeError(f"Unsupported operator: {op_type}")

        elif isinstance(node, ast.UnaryOp): # (-5)
            op_type = type(node.op)
            if op_type in self.ALLOWED_OPERATORS:
                return self.ALLOWED_OPERATORS[op_type](
                    self.safe_eval(node.operand, var_map)
                )
            raise TypeError(f"Unsupported unary operator: {op_type}")

        elif isinstance(node, ast.Call):   # استدعاء الدوال مثل sin(theta)
            func = self.safe_eval(node.func, var_map)
            args = [self.safe_eval(arg, var_map) for arg in node.args]

            # --- الإصلاح: التأكد من أن الكائن قابل للاستدعاء (Callable) ---
            if callable(func):
                return func(*args)
            raise TypeError(f"Object of type {type(func)} is not callable")

        else:
            raise TypeError(f"Security Alert: Unsupported operation {type(node)}")

    def simulate_extracted_logic(self, var_map: dict, formula_str: str):
        """
        تنفيذ المنطق المستخرج بأمان تام ودعم كامل للكينماتيكا.
        """
        try:
            # تنظيف المعادلة إذا كانت تحتوي على طرفين (x = L1 * cos...)
            if '=' in formula_str:
                formula_str = formula_str.split('=')[1].strip()

            evaluator = SafeSovereignEvaluator()
            # تحويل النص إلى شجرة تحليلية (AST)
            tree = ast.parse(formula_str, mode='eval')

            result = evaluator.safe_eval(tree.body, var_map)
            return round(result, 5) if isinstance(result, (int, float)) else result

        except Exception as e:
            return f"Logic Execution Error: {str(e)}"

class TillageenginePDFfiles:
    """
    هذا هو 'منظم الدفعات السيادي'.
    وظيفتة تقسيم الـ 217 كتلة إلى مجموعات صغيرة وإرسالها لـ LM Studio
    مع الحفاظ على 'خريطة معرفية' تربط أجزاء الكتاب ببعضها.
    """
    def __init__(self, batch_size=4): # تقليل العدد من 8 إلى 4 لزيادة التركيز
        self.batch_size = batch_size
        self.global_ledger = "" # هذه هي الشبكة الاستدلالية التي اقترحتها أنت

    def stream_to_engine(self, all_chunks: Sequence[Union[Dict[str, Any], str]], generate_func):
        """
        [النسخة السيادية الملحمية]: معالجة دفعات متسلسلة مع الحفاظ على سلامة المعادلات الهندسية.
        """
        final_insights = []
        total_chunks = len(all_chunks)
        # ذاكرة سياقية قصيرة للحفاظ على ترابط المعادلات بين الدفعات
        contextual_bridge = ""

        for i in range(0, total_chunks, self.batch_size):
            batch = all_chunks[i:i + self.batch_size]

            # 1. استخراج النص بذكاء (التوافق مع Dict أو str)
            batch_texts = []
            for c in batch:
                content = c.get('content', '') if isinstance(c, dict) else str(c)
                if content.strip():
                    batch_texts.append(content)

            # دمج السياق السابق مع الدفعة الحالية لمنع قطع المصفوفات في المنتصف
            current_raw_context = "\n".join(batch_texts)
            current_full_context = contextual_bridge + "\n" + current_raw_context

            # 2. تحديث الجسر السياقي (أخذ آخر 500 حرف لضمان عدم قطع مصفوفة 4x4)
            contextual_bridge = current_raw_context[-500:] if len(current_raw_context) > 500 else current_raw_context

            # 3. صياغة الأوامر العملياتية (Operational Commands)
            extraction_targets = (
                "STRICT TARGET: Extract the 4x4 Transformation Matrices for the 2 DOF robot configuration.\n"
                "TECHNICAL FOCUS: Precisely identify variables (L1, L2, theta1, theta2) and the final position equations (X, Y).\n"
                "MATHEMATICAL INTEGRITY: Ensure no matrix row is skipped or fragmented."
            )

            # 4. بناء "المحفز السيادي" (Sovereign Prompt)
            # تم تحسين الهيكل ليكون أكثر صرامة مع المحرك
            sovereign_prompt = (
                f"### [STRATEGIC KNOWLEDGE MAP (PREVIOUS)]:\n{self.global_ledger[-2000:]}\n\n"
                f"### [OPERATIONAL COMMANDS]:\n{extraction_targets}\n\n"
                f"### [CURRENT DATA CORE - BATCH {i//self.batch_size + 1}]:\n{current_full_context}\n\n"
                f"### [MISSION]: Execute a critical technical audit on the data core. Identify and extract kinematic patterns."
            )

            print(f"🚀 [TURBO-MODE] Processing Batch {i//self.batch_size + 1}/{ (total_chunks // self.batch_size) + 1}...")

            # 5. استدعاء محرك التوليد
            try:
                ideas, _ = generate_func(sovereign_prompt, "Critical Technical Audit")

                if ideas:
                    final_insights.extend(ideas)

                    # تحديث السجل العالمي بنقاط ارتكاز ذكية (Nodes)
                    summary_concepts = [text.strip() for text in ideas if len(text) > 20][:2]
                    node_entry = f"\n📍 Node {i//self.batch_size + 1}: {'. '.join(summary_concepts)}"
                    self.global_ledger += node_entry

            except Exception as e:
                print(f"⚠️ [ENGINE_SKIP] Error in Batch {i//self.batch_size + 1}: {str(e)}")
                continue

            # داخل الملف C - الجزء 5
            if ideas:
                final_insights.extend(ideas)
                # فحص فوري سريع: هل تحتوي الدفعة على مصفوفة كاملة؟
                matrix_found = any(str(idea).count(',') >= 15 for idea in ideas)
                if matrix_found:
                    print(f"💎 [STABILITY_SYNC]: تم تأمين مصفوفة مكتملة في الدفعة {i//self.batch_size + 1}")

        return final_insights
