# Orchestrator.py

class Smart_PDF_inference_network:
    """
    [القائد الميداني - النسخة المجزأة]: يعمل كمنسق (Orchestrator)
    يربط بين معالجة البيانات، الاستدلال، الملاحة، والحسابات.
    """
    def __init__(self, supervisor, extractor, logger, embedder, openai_client=None):
        self.logger = logger
        # ربط الوحدات الداخلية (التي قمنا بتقسيمها)
        self.data_processor = SovereignDataProcessor(supervisor, extractor, logger, self, embedder)
        self.inference_engine = SovereignInferenceEngine(logger, embedder, openai_client)
        self.navigator = SovereignNavigator(self, embedder, logger)
        self.calculator = Calculator_(logger)

        # ربط وحدة الذاكرة (D)
        self.cache_network = PDFpageCacheNetwork(supervisor)

    # --- [الروابط الذكية - الحفاظ على أسماء الدوال] ---

    def process_pdf_core(self, pdf_path: str):
        # توجيه المهمة لمعالج البيانات
        return self.data_processor.process_pdf_core(pdf_path)

    def generate(self, context: str, user_request: str):
        # توجيه المهمة لمحرك الاستدلال
        return self.inference_engine.generate(context, user_request)

    def semantic_search(self, query: str, top_k: int = 3):
        # توجيه الاستعلام للملاح
        return self.navigator.semantic_search(query, top_k)

    def execute_sovereign_mission(self, pdf_path: str, batch_engine):
        # إدارة المهمة عبر معالج البيانات مع ربطه بمحرك التوليد
        return self.data_processor.execute_sovereign_mission(
            pdf_path, batch_engine, self.inference_engine.generate
        )

    # --- [روابط الحاسب الآلي] ---

    def calculate_robot_precision(self, bits, range_max):
        return self.calculator.calculate_robot_precision(bits, range_max)

    def calculate_joint_transformation(self, theta, d, a, alpha):
        return self.calculator.calculate_joint_transformation(theta, d, a, alpha)
