# D_Overall_conclusion.py

import re

class UniversalSovereignInference:
    """
    [المستنتج الشامل]: محرك مرن يستخلص المتغيرات (Variables) والعلاقات (Formulas)
    من أي مستند تقني، ويقوم بمحاكاة النتائج بناءً على المنطق الرياضي المستخرج.
    """
    def __init__(self, data_type: str = "Technical"):
        self.data_type = data_type
        self.knowledge_base = {}

    def extract_logic_pattern(self, batch_content: str):
        """
        [المستنتج السيادي]: استخراج الأنماط المنطقية، المصفوفات، والمعادلات الهندسية.
        تم تطويرها لدعم مصفوفات الـ 4x4 ومعادلات الروبوتات (2 DOF).
        """
        # 1. استخراج المتغيرات التقنية (مثل: theta1 = 45, L1 = 0.5)
        # يدعم الأرقام الموجبة، السالبة، والعشرية
        variables = re.findall(r'(\w+)\s*[:=]\s*(-?\d+\.?\d*)', batch_content)

        # 2. استخراج مصفوفات التحويل (Transformation Matrices)
        # البحث عن الأنماط التي تشبه المصفوفات البرمجية أو الرياضية [a, b, c, d]
        matrix_pattern = re.findall(r'(\[[\d\s\.,\-]{5,}\])', batch_content)

        # 3. استخراج العلاقات الرياضية المعقدة (Formulas)
        # تم توسيع النمط ليشمل الدوال المثلثية (sin, cos, atan2) والعمليات الحسابية المتقدمة
        formulas = re.findall(
            r'(\w+\s*=\s*[\w\s\*\/\+\-\(\)\.]*(?:sin|cos|tan|atan2|sqrt)[\w\s\*\/\+\-\(\)\.]*)',
            batch_content,
            re.IGNORECASE
        )

        # 4. تنظيف وتدقيق النتائج المستخرجة
        return {
            "vars": dict(variables), # تحويلها لقاموس لسهولة الوصول
            "matrices": matrix_pattern,
            "formulas": list(set(formulas)), # إزالة التكرار
            "status": "Logic_Extracted" if (variables or formulas or matrix_pattern) else "No_Pattern_Found"
        }

    def simulate_extracted_logic(self, var_map: dict, formula_str: str):
        """
        يقوم بتنفيذ المنطق المستخرج برمجياً مع دعم الدوال المثلثية والحماية السيادية.
        """
        import math
        try:
            # 1. تجهيز بيئة حسابية غنية بالدوال الرياضية الضرورية للكينماتيكا
            safe_dict = {
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'sqrt': math.sqrt,
                'pi': math.pi,
                'radians': math.radians,
                'degrees': math.degrees,
                'abs': abs
            }

            # 2. تنظيف المعادلة: التأكد من أنها تحتوي فقط على الجزء الأيمن (بعد علامة =)
            if '=' in formula_str:
                formula_str = formula_str.split('=')[1].strip()

            # 3. التنفيذ الديناميكي الآمن
            # ندمج قيم المتغيرات المستخرجة (var_map) مع الدوال الرياضية
            execution_context = {**safe_dict, **var_map}

            result = eval(formula_str, {"__builtins__": None}, execution_context)

            # تقريب النتائج لضمان الدقة الهندسية (مثلاً 5 أرقام عشرية)
            return round(result, 5) if isinstance(result, (int, float)) else result

        except ZeroDivisionError:
            return "Execution Error: Division by zero"
        except NameError as ne:
            return f"Execution Error: Missing Variable ({str(ne)})"
        except Exception as e:
            return f"Logic Complexity: {str(e)}"
