from Classifier1 import FaBERTClassifier
from Classifier2 import FaBERTSubClassifier
from SiteSupportRag import SiteSupportRAG
from Product_InfoDirectAsk import ProductInfoRAG
from ComparisonRag import ComparisonRAG
from APIArgsFillerRag import ParameterizedRAG

classifier_a = FaBERTClassifier()
classifier_b = FaBERTSubClassifier()
site_rag = SiteSupportRAG()
product_rag = ProductInfoRAG()
comparison_rag = ComparisonRAG()


def process_query(query: str) -> str:
    label_a = classifier_a.classify(query)
    print(f"[Classifier A] → {label_a}")

    if label_a == "پشتیبانی سایت":
        return site_rag.generate_answer(query)

    elif label_a == "راهنمایی در مورد کالا":
        return product_rag.generate_answer(query)

    elif label_a == "مقایسه":
        return comparison_rag.generate_answer(query)

    elif label_a == "جستجو":
        label_b = classifier_b.classify(query)
        print(f"[Classifier B] → {label_b}")

        if label_b in ["ساده", "پارامتری", "غیر پارامتری"]:
            rag = ParameterizedRAG(class_name=label_b)
            return rag.generate_answer(query)
        else:        
            return f"طبقه‌بندی دوم نامشخص: {label_b}"

    else:
        return f"طبقه‌بندی اول نامشخص: {label_a}"
