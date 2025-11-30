# xai_explainer.py
from lime.lime_text import LimeTextExplainer
import torch

explainer = LimeTextExplainer(class_names=["negative", "positive"])

def explain(text, model, tokenizer):
    def pred_proba(text_list):
        enc = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
        logits = model(**enc).logits
        return torch.softmax(logits, dim=1).detach().numpy()
    return explainer.explain_instance(text, pred_proba, num_features=8).as_list()
