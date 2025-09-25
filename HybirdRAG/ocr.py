import os
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PaddleOCR = None
    PADDLEOCR_AVAILABLE = False
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    convert_from_path = None
    PDF2IMAGE_AVAILABLE = False

class OCRExtractor:
    def __init__(self, lang="en"):
        if PADDLEOCR_AVAILABLE:
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        else:
            self.ocr = None

    def pdf_to_text(self, pdf_path: str):
        """Run OCR on a PDF -> return list of per-page text."""
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR is required for OCR functionality. Please install it with: pip install paddleocr")
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError("pdf2image is required for PDF processing. Please install it with: pip install pdf2image")
        
        pages = convert_from_path(pdf_path)
        page_texts = []
        for idx, page in enumerate(pages):
            img_path = f"temp_page_{idx}.png"
            page.save(img_path, "PNG")

            result = self.ocr.ocr(img_path)
            text_lines = [line[1][0] for line in result[0]]
            page_texts.append("\n".join(text_lines))

            os.remove(img_path)
        return page_texts
