import os
from paddleocr import PaddleOCR
from pdf2image import convert_from_path

class OCRExtractor:
    def __init__(self, lang="en"):
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)

    def pdf_to_text(self, pdf_path: str):
        """Run OCR on a PDF -> return list of per-page text."""
        pages = convert_from_path(pdf_path)
        page_texts = []
        for idx, page in enumerate(pages):
            img_path = f"temp_page_{idx}.png"
            page.save(img_path, "PNG")

            result = self.ocr.ocr(img_path, cls=True)
            text_lines = [line[1][0] for line in result[0]]
            page_texts.append("\n".join(text_lines))

            os.remove(img_path)
        return page_texts
