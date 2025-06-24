# services/resume_parser_service.py

import os
import re
import json
import tempfile
import logging
from typing import List, Dict, Optional

import pytesseract
import spacy
from PIL import Image, ImageOps
from docx import Document
from pdfminer.high_level import extract_text as extract_pdf_text
from langdetect import detect, LangDetectException
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger("resume_parser_service")

# === Constants ===
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".png", ".jpg", ".jpeg"]
SKILL_BANK_PATH = "skill_bank.json"

# === Models ===
spacy_models = {
    "en": spacy.load("en_core_web_sm"),
    "ru": spacy.load("ru_core_news_sm"),
    "uz": spacy.load("en_core_web_sm"),  # fallback
}
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

class ResumeParserService:
    def parse_resume(self, file_path: str, pinfl: str = "unknown") -> Dict:
        ext = os.path.splitext(file_path)[-1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        # === Extract text ===
        text = self.extract_text(file_path, ext)
        cleaned = self.clean_text(text)

        # === Detect language ===
        try:
            lang = detect(cleaned)
            lang_code = lang[:2] if lang[:2] in spacy_models else "en"
        except LangDetectException:
            lang_code = "en"

        nlp = spacy_models[lang_code]
        doc = nlp(cleaned)

        # === Extract structured fields ===
        new_skills = self.extract_candidate_skills(doc)
        self.save_skill_bank(new_skills)

        parsed = {
            "name": self.extract_name(doc),
            "email": self.extract_email(cleaned),
            "phone": self.extract_phone(cleaned),
            "links": self.extract_links(cleaned),
            "location": self.extract_location(doc),
            "skills": self.semantic_skill_match(cleaned),
            "education": self.extract_education(doc, lang_code),
            "experience": self.extract_experience(doc, lang_code),
            "certifications": self.extract_certifications(doc, lang_code),
            "languages": self.extract_languages(cleaned),
            "summary": self.extract_summary(doc),
            "job_history": self.extract_job_history(cleaned),
            "language": lang_code,
            "resume_embedding": self.get_resume_embedding(cleaned),
            "raw_text": cleaned,
            "parse_confidence": self.estimate_confidence(cleaned),
            "fields_found": self.count_fields_present(cleaned),
        }

        self.store_feedback(pinfl, parsed)
        return parsed

    def extract_text(self, file_path: str, ext: str) -> str:
        if ext == ".pdf":
            return extract_pdf_text(file_path)
        elif ext == ".docx":
            return "\n".join([p.text for p in Document(file_path).paragraphs])
        elif ext in [".png", ".jpg", ".jpeg"]:
            try:
                img = Image.open(file_path)
                img = ImageOps.exif_transpose(img).convert("L")  # rotate & grayscale
                return pytesseract.image_to_string(img)
            except Exception as e:
                logger.warning(f"OCR failed: {e}")
                return ""
        return ""

    def clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def extract_name(self, doc) -> str:
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "PER") and 1 < len(ent.text.split()) <= 4:
                return ent.text.strip()
        return ""

    def extract_email(self, text: str) -> str:
        match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
        return match.group(0) if match else ""

    def extract_phone(self, text: str) -> str:
        match = re.search(r"\+?\d[\d \-]{8,}\d", text)
        return match.group(0) if match else ""

    def extract_links(self, text: str) -> List[str]:
        return re.findall(r"https?://[^\s]+", text)

    def extract_location(self, doc) -> str:
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC"):
                return ent.text.strip()
        return ""

    def extract_summary(self, doc) -> str:
        return " ".join([s.text for s in list(doc.sents)[:5] if len(s.text.strip()) > 20])

    def extract_languages(self, text: str) -> List[str]:
        known = ["english", "russian", "uzbek", "french", "turkish", "german", "arabic"]
        return list({lang.title() for lang in known if lang in text.lower()})

    def extract_education(self, doc, lang: str) -> str:
        keywords = {
            "en": ["university", "bachelor", "master", "phd", "college"],
            "ru": ["университет", "бакалавр", "магистр", "институт"],
            "uz": ["universitet", "bakalavr", "magistr", "litsey"]
        }
        kwords = keywords.get(lang, keywords["en"])
        return "\n".join([s.text for s in doc.sents if any(kw in s.text.lower() for kw in kwords)])

    def extract_experience(self, doc, lang: str) -> str:
        keywords = {
            "en": ["experience", "intern", "worked", "responsibilities", "role", "job"],
            "ru": ["опыт", "работал", "должность", "обязанности"],
            "uz": ["tajriba", "ish", "lavozim", "majburiyat"]
        }
        kwords = keywords.get(lang, keywords["en"])
        return "\n".join([s.text for s in doc.sents if any(kw in s.text.lower() for kw in kwords)])

    def extract_certifications(self, doc, lang: str) -> str:
        keywords = {
            "en": ["certificate", "certification", "training", "completed"],
            "ru": ["сертификат", "курсы", "обучение"],
            "uz": ["sertifikat", "kurs", "trening"]
        }
        kwords = keywords.get(lang, keywords["en"])
        return "\n".join([s.text for s in doc.sents if any(kw in s.text.lower() for kw in kwords)])

    def extract_job_history(self, text: str) -> List[str]:
        pattern = r"(?:[A-ZА-Я][\w\s&\-,.]+)\s+at\s+[\w&\-,.\s]+\s+\d{4}"
        return re.findall(pattern, text)

    def extract_candidate_skills(self, doc) -> List[str]:
        skills = set()
        for chunk in doc.noun_chunks:
            txt = chunk.text.strip().lower()
            if 2 <= len(txt.split()) <= 4 and not txt.endswith(('.', ',')):
                if re.search(r"(development|analysis|engineering|design|data|project|machine|learning|software|sales|support|backend|frontend)", txt):
                    skills.add(txt.title())
        return list(skills)

    def semantic_skill_match(self, text: str) -> List[str]:
        bank = self.load_skill_bank()
        if not bank:
            return []
        text_embed = sbert_model.encode(text, convert_to_tensor=True)
        skill_embeds = sbert_model.encode(bank, convert_to_tensor=True)
        scores = util.cos_sim(text_embed, skill_embeds)[0]
        top_indices = scores.argsort(descending=True)[:10]
        return [bank[i] for i in top_indices]

    def get_resume_embedding(self, text: str) -> List[float]:
        return sbert_model.encode(text).tolist()

    def estimate_confidence(self, text: str) -> float:
        return min(1.0, 0.5 + 0.0001 * len(text))  # naive heuristic

    def count_fields_present(self, text: str) -> Dict[str, bool]:
        return {
            "email": bool(self.extract_email(text)),
            "phone": bool(self.extract_phone(text)),
            "links": bool(self.extract_links(text)),
        }

    def load_skill_bank(self) -> List[str]:
        if os.path.exists(SKILL_BANK_PATH):
            with open(SKILL_BANK_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_skill_bank(self, new_skills: List[str]):
        try:
            existing = set(self.load_skill_bank())
            existing.update(new_skills)
            deduped = self.deduplicate_skills(list(existing))
            with open(SKILL_BANK_PATH, "w", encoding="utf-8") as f:
                json.dump(sorted(deduped), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[SkillBank] Failed to save skill bank: {e}")

    def deduplicate_skills(self, skills: List[str]) -> List[str]:
        if len(skills) < 2:
            return skills
        embeddings = sbert_model.encode(skills)
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.4).fit(embeddings)
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            clusters.setdefault(label, []).append(skills[idx])
        return [sorted(group, key=len)[0] for group in clusters.values()]

    def store_feedback(self, pinfl: str, parsed: Dict):
        try:
            with open("feedback.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"pinfl": pinfl, "parsed": parsed}, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"[Feedback] Failed to store feedback: {e}")
