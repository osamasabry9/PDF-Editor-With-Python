import sys
import os
import io
import threading
import platform
import shutil
import time
import tempfile
import requests
import numpy as np
import fitz
import cv2
import pytesseract
import arabic_reshaper
from bidi.algorithm import get_display
from typing import List, Dict
from dataclasses import dataclass
from PIL import Image

# PyQt5 imports organized by functionality
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QPushButton,
    QTextEdit, QProgressBar, QFrame, QSpinBox,
    QGroupBox, QGridLayout, QCheckBox, QComboBox,
    QGraphicsView, QGraphicsScene, QGraphicsTextItem, QMenu, QAction, QInputDialog, QFontDialog, QColorDialog,
    QListWidget, QListWidgetItem,
    QToolBar, QStatusBar, QMainWindow, QLineEdit, QDialog, QDialogButtonBox,
    QGraphicsDropShadowEffect, QProgressDialog
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSettings
)
from PyQt5.QtGui import (
    QPixmap, QFont, QColor, QPainter, QImage, QTransform, QKeySequence, QIcon
)

# ==============================================================
# CONSTANTS AND CONFIGURATION
# ==============================================================
AI_CONFIGS = {
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4-turbo-preview",
        "headers": {"Authorization": "Bearer YOUR_OPENAI_API_KEY"}
    },
    "claude": {
        "url": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-opus-20240229",
        "headers": {"x-api-key": "YOUR_CLAUDE_API_KEY"}
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
        "api_key": "YOUR_GEMINI_API_KEY"
    }
}

class ArabicTextHandler:
    """معالج متخصص للنصوص العربية في PDF"""

    def __init__(self):
        self.arabic_fonts_system = [
            'Tahoma', 'Arial Unicode MS', 'Calibri',
            'Traditional Arabic', 'Simplified Arabic',
            'Arial', 'Times New Roman'
        ]

    def is_mixed_text(self, text):
        """فحص النص المختلط (عربي + إنجليزي)"""
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        return arabic_chars > 0 and latin_chars > 0

    def process_mixed_text(self, text):
        """معالجة النص المختلط"""
        try:
            if self.is_mixed_text(text):
                # تقسيم النص إلى أجزاء عربية وإنجليزية
                parts = []
                current_part = ""
                current_is_arabic = None

                for char in text:
                    char_is_arabic = '\u0600' <= char <= '\u06FF'

                    if current_is_arabic is None:
                        current_is_arabic = char_is_arabic
                        current_part = char
                    elif current_is_arabic == char_is_arabic:
                        current_part += char
                    else:
                        # حفظ الجزء الحالي ومعالجته
                        if current_is_arabic:
                            reshaped = arabic_reshaper.reshape(current_part)
                            parts.append(get_display(reshaped))
                        else:
                            parts.append(current_part)

                        # بدء جزء جديد
                        current_part = char
                        current_is_arabic = char_is_arabic

                # معالجة الجزء الأخير
                if current_part:
                    if current_is_arabic:
                        reshaped = arabic_reshaper.reshape(current_part)
                        parts.append(get_display(reshaped))
                    else:
                        parts.append(current_part)

                return ''.join(parts)
            else:
                # نص عربي فقط أو إنجليزي فقط
                if any('\u0600' <= c <= '\u06FF' for c in text):
                    reshaped = arabic_reshaper.reshape(text)
                    return get_display(reshaped)
                return text

        except Exception as e:
            print(f"خطأ في معالجة النص المختلط: {e}")
            return text

# ==============================================================
# تحسين إضافي: حل مشكلة الترميز
# ==============================================================

class EncodingFixer:
    """حل مشاكل الترميز في النصوص"""

    @staticmethod
    def fix_text_encoding(text):
        """إصلاح مشاكل الترميز الشائعة"""
        try:
            # إصلاح الترميز UTF-8
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')

            # إصلاح الأحرف المكسورة الشائعة
            fixes = {
                'Ø£': 'أ',
                'Ù†': 'ن',
                'Ø§': 'ا',
                'ÙŠ': 'ي',
                'Ø©': 'ة',
                'Ø±': 'ر'
            }

            for broken, fixed in fixes.items():
                text = text.replace(broken, fixed)

            return text.strip()

        except Exception as e:
            print(f"خطأ في إصلاح الترميز: {e}")
            return text
class ImprovedPDFSaver:
    """محسن حفظ PDF مع دعم أفضل للنصوص العربية والإنجليزية"""

    def __init__(self):
        # خريطة الخطوط المدعومة في PyMuPDF
        self.font_mapping = {
            'Arial': 'helv',
            'Helvetica': 'helv',
            'Times': 'tiro',
            'Times New Roman': 'tiro',
            'Courier': 'cour',
            'Courier New': 'cour',
            'Symbol': 'symb',
            'ZapfDingbats': 'zadb'
        }

        # الخطوط العربية المدعومة
        self.arabic_fonts = {
            'Tahoma': 'taho',
            'Arial Unicode MS': 'helv',
            'Calibri': 'cali',
            'Traditional Arabic': 'taha'
        }

    def is_arabic_text(self, text: str) -> bool:
        """فحص النص إذا كان يحتوي على أحرف عربية"""
        if not text:
            return False
        arabic_char_count = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        return arabic_char_count > 0

    def process_arabic_text(self, text: str) -> str:
        """معالجة النص العربي للعرض الصحيح"""
        try:
            if self.is_arabic_text(text):
                reshaped_text = arabic_reshaper.reshape(text)
                return get_display(reshaped_text)
            return text
        except Exception as e:
            print(f"خطأ في معالجة النص العربي: {e}")
            return text

    def save_pdf_with_modifications(self, original_doc, output_path, change_tracker, current_page, text_blocks):
        """حفظ PDF مع التعديلات المحسن"""
        try:
            print(f"🔄 بدء عملية الحفظ المحسنة...")

            # إنشاء مستند جديد
            new_doc = fitz.open()

            # نسخ جميع الصفحات
            for page_num in range(len(original_doc)):
                original_page = original_doc[page_num]
                new_page = new_doc.new_page(
                    width=original_page.rect.width,
                    height=original_page.rect.height
                )

                if page_num == current_page:
                    # تطبيق التعديلات على الصفحة الحالية
                    success = self.apply_comprehensive_changes(
                        original_page, new_page, page_num,
                        change_tracker, text_blocks
                    )
                    if not success:
                        print(f"⚠️ فشل في تطبيق التعديلات، استخدام الصفحة الأصلية")
                        new_page.show_pdf_page(original_page.rect, original_doc, page_num)
                else:
                    # نسخ الصفحة بدون تعديل
                    new_page.show_pdf_page(original_page.rect, original_doc, page_num)

            # حفظ المستند
            success = self.safe_document_save(new_doc, output_path)
            new_doc.close()

            if success:
                print(f"✅ تم الحفظ بنجاح: {output_path}")
                return True
            else:
                print(f"❌ فشل في الحفظ")
                return False

        except Exception as e:
            print(f"❌ خطأ عام في الحفظ: {e}")
            import traceback
            traceback.print_exc()
            return False


    def apply_comprehensive_changes(self, original_page, new_page, page_num, change_tracker, text_blocks):
        """
        ننسخ الصفحة الأصلية كصورة، ثم نطبّق التعديلات عليها.
        """
        try:
            print(f"📝 تطبيق التعديلات على الصفحة {page_num}")

            # 1) أرسم الصفحة الأصلية كخلفية (كصورة)
            pix = original_page.get_pixmap()
            new_page.insert_image(
                fitz.Rect(0, 0, original_page.rect.width, original_page.rect.height),
                pixmap=pix
            )

            # 2) اجلب التعديلات المسجلة
            changes = change_tracker.get_changes_for_page(page_num)

            # 3) طبّق التعديلات فقط
            self.comprehensive_text_replacement(new_page, changes)

            return True

        except Exception as e:
            print(f"❌ خطأ في apply_comprehensive_changes: {e}")
            return False

    def comprehensive_text_replacement(self, page, changes):
        """
        ندرج النصوص المعدّلة فقط فوق الخلفية،
        باستعمال insert_textbox لضبط الموضع تمامًا.
        """
        try:
            print("🔄 بدء إدراج التعديلات المعدّلة فقط...")

            for change in changes:
                ctype = change['type']
                data = change['data']
                bbox = data.get('bbox')
                if not bbox:
                    continue
                rect = fitz.Rect(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])

                raw = data.get('new_text', data.get('text', ''))
                # معالجة عربية إن لزم
                if any('\u0600' <= c <= '\u06FF' for c in raw):
                    reshaped = arabic_reshaper.reshape(raw)
                    text = get_display(reshaped)
                    align = fitz.TEXT_ALIGN_RIGHT
                else:
                    text = raw
                    align = fitz.TEXT_ALIGN_LEFT

                font_name = self.get_safe_font_name(data.get('font_family', 'Arial'))
                font_size = max(8, min(data.get('font_size', 12), 72))
                color = self.hex_to_rgb(data.get('color', '#000000'))

                # إدراج ضمن البوكس
                page.insert_textbox(
                    rect,
                    text,
                    fontname=font_name,
                    fontsize=font_size,
                    color=color,
                    align=align,
                    overlay=True  # لا يمسح ما وراءه
                )
                print(f"✅ {ctype} على {change['block_id']} في {rect}")

            print("✅ انتهى إدراج التعديلات")
            return True

        except Exception as e:
            print(f"❌ خطأ في comprehensive_text_replacement: {e}")
            return False
    def insert_modified_text(self, page, mod_data):
        """إدراج نص معدل"""
        try:
            text = mod_data['new_text']
            bbox = mod_data['bbox']
            font_size = mod_data.get('font_size', 12)
            color = mod_data.get('color', '#000000')
            font_family = mod_data.get('font_family', 'Arial')

            # معالجة النص العربي
            processed_text = self.process_arabic_text(text)

            # تحديد موقع الإدراج
            insert_point = fitz.Point(bbox[0], bbox[1] + bbox[3] * 0.8)

            # تحديد الخط
            font_name = self.get_safe_font_name(font_family)

            # تحويل اللون
            rgb_color = self.hex_to_rgb(color)

            # إدراج النص
            result = page.insert_text(
                insert_point,
                processed_text,
                fontsize=max(8, min(font_size, 72)),
                fontname=font_name,
                color=rgb_color,

            )

            return True

        except Exception as e:
            print(f"❌ خطأ في إدراج النص المعدل: {e}")
            return False

    def insert_change_text(self, page, change):
        """إدراج نص من تغيير"""
        try:
            change_data = change['data']

            if change['type'] == 'move':
                text = change_data.get('text', '')
                new_pos = change_data.get('new_pos', [0, 0])
                bbox = change_data.get('bbox', [0, 0, 100, 20])

                insert_point = fitz.Point(new_pos[0], new_pos[1] + bbox[3] * 0.8)

            elif change['type'] == 'add':
                text = change_data.get('text', '')
                bbox = change_data.get('bbox', [0, 0, 100, 20])
                insert_point = fitz.Point(bbox[0], bbox[1] + bbox[3] * 0.8)

            else:
                return True  # تخطي أنواع التغييرات الأخرى

            # معالجة النص
            processed_text = self.process_arabic_text(text)

            # إعدادات الخط
            font_size = change_data.get('font_size', 12)
            color = change_data.get('color', '#000000')
            font_family = change_data.get('font_family', 'Arial')

            font_name = self.get_safe_font_name(font_family)
            rgb_color = self.hex_to_rgb(color)

            # إدراج النص
            result = page.insert_text(
                insert_point,
                processed_text,
                fontsize=max(8, min(font_size, 72)),
                fontname=font_name,
                color=rgb_color,

            )

            return True

        except Exception as e:
            print(f"❌ خطأ في إدراج نص التغيير: {e}")
            return False

    def insert_original_text(self, page, block):
        """إدراج نص أصلي غير معدل"""
        try:
            # معالجة النص
            processed_text = self.process_arabic_text(block.text)

            # تحديد موقع الإدراج
            bbox = block.bbox
            insert_point = fitz.Point(bbox[0], bbox[1] + bbox[3] * 0.8)

            # إعدادات الخط
            font_name = self.get_safe_font_name(block.font_family)
            rgb_color = self.hex_to_rgb(block.color)

            # إدراج النص
            result = page.insert_text(
                insert_point,
                processed_text,
                fontsize=max(8, min(block.font_size, 72)),
                fontname=font_name,
                color=rgb_color,

            )

            return True

        except Exception as e:
            print(f"❌ خطأ في إدراج النص الأصلي: {e}")
            return False

    def get_safe_font_name(self, font_family):
        """الحصول على اسم خط آمن"""
        # فحص الخطوط العربية أولاً
        if font_family in self.arabic_fonts:
            return self.arabic_fonts[font_family]

        # ثم الخطوط العادية
        if font_family in self.font_mapping:
            return self.font_mapping[font_family]

        # خط افتراضي آمن
        return 'helv'

    def hex_to_rgb(self, hex_color: str) -> tuple:
        """تحويل اللون من hex إلى RGB"""
        try:
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
        except:
            return (0, 0, 0)  # أسود افتراضي

    def safe_document_save(self, doc, file_path):
        """حفظ آمن للمستند"""
        try:
            # طريقة 1: الحفظ المباشر
            doc.save(
                file_path,
                garbage=4,
                deflate=True,
                clean=True,
                ascii=False,
                expand=255,

            )

            # فحص أن الملف تم إنشاؤه وله حجم معقول
            if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:
                print("✅ تم الحفظ المباشر بنجاح")
                return True
            else:
                print("⚠️ فشل الحفظ المباشر، جرب الطريقة البديلة...")
                return False

        except Exception as e:
            print(f"❌ خطأ في الحفظ المباشر: {e}")

            # طريقة 2: الحفظ عبر ملف مؤقت
            try:
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_path = temp_file.name

                doc.save(temp_path, garbage=4, deflate=True)

                # فحص الملف المؤقت
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1000:
                    # نقل الملف إلى الهدف
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    shutil.move(temp_path, file_path)
                    print("✅ تم الحفظ عبر الملف المؤقت")
                    return True
                else:
                    print("❌ فشل إنشاء الملف المؤقت")
                    return False

            except Exception as e2:
                print(f"❌ فشل الحفظ عبر الملف المؤقت: {e2}")
                return False



# ==============================================================
# DATA STRUCTURES
# ==============================================================
@dataclass
class TextBlock:
    """Enhanced text block with complete formatting information"""
    text: str
    bbox: List[float]  # [x, y, width, height]
    font_family: str
    font_size: float
    font_weight: str
    font_style: str
    color: str
    alignment: str
    line_height: float
    char_spacing: float
    word_spacing: float
    original_bbox: List[float]  # Original position for restoration
    page_number: int
    block_id: str
    rotation: float = 0.0
    opacity: float = 1.0
    background_color: str = "transparent"
    border_color: str = "transparent"
    border_width: float = 0.0
        # Arabic/RTL specific properties
    is_rtl: bool = False
    text_direction: str = "ltr"  # ltr, rtl, auto
    original_font_name: str = ""  # Original font from PDF
    font_flags: int = 0  # Font flags from PDF
    transform_matrix: List[float] = None  # Transformation matrix
    # Layout preservation
    line_blocks: List[Dict] = None  # Individual line information
    word_positions: List[Dict] = None  # Word-level positioning
    preserve_layout: bool = True

# ==============================================================
# FONT HANDLING AND UTILITIES
# ==============================================================
def setup_tesseract():
    """Setup Tesseract executable path"""
    system = platform.system().lower()
    tesseract_cmd = shutil.which("tesseract")

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        return tesseract_cmd

    if system == "windows":
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return path
    return None
# ==============================================================
# AI INTEGRATION
# ==============================================================
class AITextProcessor:
    """AI-powered text processing and enhancement"""
    def __init__(self):
        self.current_provider = "openai"  # Default
        self.settings = QSettings()

    def set_api_key(self, provider: str, api_key: str):
        """Set API key for AI provider"""
        self.settings.setValue(f"ai/{provider}/api_key", api_key)

    def get_api_key(self, provider: str) -> str:
        """Get API key for AI provider"""
        return self.settings.value(f"ai/{provider}/api_key", "")

    def enhance_ocr_text(self, text: str, context: str = "") -> str:
        """Use AI to enhance OCR results"""
        if not text.strip():
            return text

        prompt = f"""
        يرجى تحسين النص التالي الذي تم استخراجه بـ OCR. قم بتصحيح الأخطاء الإملائية والنحوية والحفاظ على المعنى الأصلي:

        النص الأصلي: {text}
        السياق: {context}

        أعد النص المحسن فقط بدون تفسيرات إضافية.
        """

        try:
            enhanced_text = self._call_ai_api(prompt)
            return enhanced_text.strip() if enhanced_text else text
        except Exception as e:
            print(f"AI enhancement failed: {e}")
            return text

    def translate_text(self, text: str, target_language: str) -> str:
        """Translate text using AI"""
        prompt = f"""
        ترجم النص التالي إلى {target_language}:

        {text}

        قدم الترجمة فقط بدون تفسيرات.
        """

        try:
            translated = self._call_ai_api(prompt)
            return translated.strip() if translated else text
        except Exception as e:
            print(f"Translation failed: {e}")
            return text

    def suggest_improvements(self, text: str) -> List[str]:
        """Get AI suggestions for text improvement"""
        prompt = f"""
        اقترح تحسينات للنص التالي. قدم 3-5 اقتراحات محددة:

        {text}

        اكتب كل اقتراح في سطر منفصل.
        """

        try:
            suggestions = self._call_ai_api(prompt)
            return [s.strip() for s in suggestions.split('\n') if s.strip()]
        except Exception as e:
            print(f"Suggestions failed: {e}")
            return []

    def _call_ai_api(self, prompt: str) -> str:
        """Call the configured AI API"""
        api_key = self.get_api_key(self.current_provider)
        if not api_key:
            raise Exception("API key not configured")

        config = AI_CONFIGS[self.current_provider].copy()

        if self.current_provider == "openai":
            return self._call_openai(prompt, api_key, config)
        elif self.current_provider == "claude":
            return self._call_claude(prompt, api_key, config)
        elif self.current_provider == "gemini":
            return self._call_gemini(prompt, api_key, config)

        raise Exception(f"Unsupported AI provider: {self.current_provider}")

    def _call_openai(self, prompt: str, api_key: str, config: dict) -> str:
        """Call OpenAI API"""
        config["headers"]["Authorization"] = f"Bearer {api_key}"

        data = {
            "model": config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.3
        }

        response = requests.post(config["url"], headers=config["headers"], json=data)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def _call_claude(self, prompt: str, api_key: str, config: dict) -> str:
        """Call Claude API"""
        config["headers"]["x-api-key"] = api_key
        config["headers"]["content-type"] = "application/json"
        config["headers"]["anthropic-version"] = "2023-06-01"

        data = {
            "model": config["model"],
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(config["url"], headers=config["headers"], json=data)
        response.raise_for_status()

        result = response.json()
        return result["content"][0]["text"]

    def _call_gemini(self, prompt: str, api_key: str, config: dict) -> str:
        """Call Gemini API"""
        url = f"{config['url']}?key={api_key}"

        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        response = requests.post(url, json=data)
        response.raise_for_status()

        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]

# ==============================================================
# GUI COMPONENTS
# ==============================================================
class AdvancedTextItem(QGraphicsTextItem):
    """Advanced editable text item with full formatting support"""
    def __init__(self, text_block: TextBlock, parent=None):
        super().__init__(text_block.text, parent)
        self.text_block = text_block
        self.is_editing = False
        self.original_transform = QTransform()
        self.setOpacity(0.0)
        self.text_block.opacity = 0.0
        self.is_modified = False
        self.original_text = text_block.text
        self.setup_item()

    def setup_item(self):
        """Initialize item properties and formatting"""
        self.setPos(self.text_block.bbox[0], self.text_block.bbox[1])
        self.setTextWidth(self.text_block.bbox[2])
        self.apply_formatting()
        self.setup_interaction()
        self.add_visual_effects()
        editor = self.get_editor()
        if editor:
            editor.change_tracker.save_original_state(
                self.text_block.page_number,
                self.text_block.block_id,
                {
                    'text': self.original_text,
                    'bbox': self.text_block.original_bbox.copy(),
                    'font_size': self.text_block.font_size,
                    'font_family': self.text_block.font_family,
                    'color': self.text_block.color
                }
            )
    def get_editor(self):
        """الحصول على المحرر الرئيسي"""
        try:
            return self.scene().views()[0].window()
        except:
            return None
    def apply_formatting(self):
        """Apply complete text formatting"""
        # Create font
        font = QFont(self.text_block.font_family, int(self.text_block.font_size))

        if self.text_block.font_weight == "bold":
            font.setBold(True)
        if self.text_block.font_style == "italic":
            font.setItalic(True)

        self.setFont(font)

        # Set color
        color = QColor(self.text_block.color) if self.text_block.color != "transparent" else QColor("black")
        self.setDefaultTextColor(color)

        # Set alignment
        cursor = self.textCursor()
        text_format = cursor.blockFormat()

        if self.text_block.alignment == "center":
            text_format.setAlignment(Qt.AlignCenter)
        elif self.text_block.alignment == "right":
            text_format.setAlignment(Qt.AlignRight)
        elif self.text_block.alignment == "justify":
            text_format.setAlignment(Qt.AlignJustify)
        else:
            text_format.setAlignment(Qt.AlignLeft)

        cursor.setBlockFormat(text_format)

        # Set rotation
        if self.text_block.rotation != 0:
            transform = QTransform()
            transform.rotate(self.text_block.rotation)
            self.setTransform(transform)

        # Set opacity
        self.setOpacity(self.text_block.opacity)

    def setup_interaction(self):
        """Set interaction flags"""
        self.setFlags(
            QGraphicsTextItem.ItemIsSelectable |
            QGraphicsTextItem.ItemIsFocusable |
            QGraphicsTextItem.ItemIsMovable |
            QGraphicsTextItem.ItemSendsGeometryChanges
        )

    def add_visual_effects(self):
        """Add visual effects for better editing experience"""
        # Add shadow effect when selected
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(5)
        shadow.setColor(QColor(0, 123, 255, 100))
        shadow.setOffset(2, 2)
        self.setGraphicsEffect(shadow)

    def contextMenuEvent(self, event):
        """Enhanced context menu"""
        menu = QMenu()

        # Text formatting
        format_menu = menu.addMenu("تنسيق النص")

        font_action = QAction("تغيير الخط", self)
        font_action.triggered.connect(self.change_font)
        format_menu.addAction(font_action)

        color_action = QAction("تغيير اللون", self)
        color_action.triggered.connect(self.change_color)
        format_menu.addAction(color_action)

        align_menu = format_menu.addMenu("المحاذاة")

        align_left = QAction("يسار", self)
        align_left.triggered.connect(lambda: self.set_alignment("left"))
        align_menu.addAction(align_left)

        align_center = QAction("وسط", self)
        align_center.triggered.connect(lambda: self.set_alignment("center"))
        align_menu.addAction(align_center)

        align_right = QAction("يمين", self)
        align_right.triggered.connect(lambda: self.set_alignment("right"))
        align_menu.addAction(align_right)

        # AI enhancements
        ai_menu = menu.addMenu("تحسينات AI")

        enhance_action = QAction("تحسين النص", self)
        enhance_action.triggered.connect(self.enhance_with_ai)
        ai_menu.addAction(enhance_action)

        translate_action = QAction("ترجمة", self)
        translate_action.triggered.connect(self.translate_with_ai)
        ai_menu.addAction(translate_action)

        suggestions_action = QAction("اقتراحات", self)
        suggestions_action.triggered.connect(self.get_ai_suggestions)
        ai_menu.addAction(suggestions_action)

        menu.addSeparator()

        # Position controls
        reset_action = QAction("إعادة للموضع الأصلي", self)
        reset_action.triggered.connect(self.reset_position)
        menu.addAction(reset_action)

        duplicate_action = QAction("تكرار", self)
        duplicate_action.triggered.connect(self.duplicate_item)
        menu.addAction(duplicate_action)

        delete_action = QAction("حذف", self)
        delete_action.triggered.connect(self.delete_item)
        menu.addAction(delete_action)

        menu.exec_(event.screenPos())

    def change_font(self):
        """Change font with dialog"""
        font, ok = QFontDialog.getFont(self.font(), self.scene().views()[0])
        if ok:
            self.setFont(font)
            self.text_block.font_family = font.family()
            self.text_block.font_size = font.pointSize()
            self.text_block.font_weight = "bold" if font.bold() else "normal"
            self.text_block.font_style = "italic" if font.italic() else "normal"

    def change_color(self):
        """Change text color"""
        color = QColorDialog.getColor(self.defaultTextColor(), self.scene().views()[0])
        if color.isValid():
            self.setDefaultTextColor(color)
            self.text_block.color = color.name()

    def set_alignment(self, alignment: str):
        """Set text alignment"""
        cursor = self.textCursor()
        text_format = cursor.blockFormat()

        if alignment == "center":
            text_format.setAlignment(Qt.AlignCenter)
        elif alignment == "right":
            text_format.setAlignment(Qt.AlignRight)
        elif alignment == "justify":
            text_format.setAlignment(Qt.AlignJustify)
        else:
            text_format.setAlignment(Qt.AlignLeft)

        cursor.setBlockFormat(text_format)
        self.text_block.alignment = alignment

    def enhance_with_ai(self):
        """Enhance text using AI"""
        progress = QProgressDialog("جاري تحسين النص...", "إلغاء", 0, 0, self.scene().views()[0])
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        def enhance_thread():
            try:
                ai_processor = AITextProcessor()
                enhanced_text = ai_processor.enhance_ocr_text(self.toPlainText())

                # Update in main thread
                QTimer.singleShot(0, lambda: self.update_text_safely(enhanced_text, progress))
            except Exception as e:
                QTimer.singleShot(0, lambda: self.show_error(str(e), progress))

        threading.Thread(target=enhance_thread, daemon=True).start()

    def translate_with_ai(self):
        """Translate text using AI"""
        languages = ["الإنجليزية", "الفرنسية", "الألمانية", "الإسبانية", "الإيطالية"]
        language, ok = QInputDialog.getItem(
            self.scene().views()[0], "اختر اللغة", "ترجمة إلى:", languages, 0, False
        )

        if ok and language:

            progress = QProgressDialog("جاري الترجمة...", "إلغاء", 0, 0, self.scene().views()[0])
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            def translate_thread():
                try:
                    ai_processor = AITextProcessor()
                    translated_text = ai_processor.translate_text(self.toPlainText(), language)

                    QTimer.singleShot(0, lambda: self.update_text_safely(translated_text, progress))
                except Exception as e:
                    QTimer.singleShot(0, lambda: self.show_error(str(e), progress))

            threading.Thread(target=translate_thread, daemon=True).start()

    def get_ai_suggestions(self):
        """Get AI suggestions for improvement"""

        progress = QProgressDialog("جاري الحصول على اقتراحات...", "إلغاء", 0, 0, self.scene().views()[0])
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        def suggestions_thread():
            try:
                ai_processor = AITextProcessor()
                suggestions = ai_processor.suggest_improvements(self.toPlainText())

                QTimer.singleShot(0, lambda: self.show_suggestions(suggestions, progress))
            except Exception as e:
                QTimer.singleShot(0, lambda: self.show_error(str(e), progress))

        threading.Thread(target=suggestions_thread, daemon=True).start()

    def update_text_safely(self, new_text: str, progress_dialog):
        progress_dialog.close()
        if new_text.strip() and new_text != self.toPlainText():
            old_text = self.toPlainText()
            self.setPlainText(new_text)
            self.text_block.text = new_text
            self.is_modified = True

            # تسجيل التغيير
            editor = self.scene().views()[0].window()
            editor.change_tracker.add_change(
                self.text_block.page_number,
                self.text_block.block_id,
                'edit',
                {
                    'original_text': old_text,
                    'new_text': new_text,
                    'bbox': self.text_block.original_bbox.copy() if self.text_block.original_bbox else self.text_block.bbox.copy(),
                    'font_size': self.text_block.font_size,
                    'font_family': self.text_block.font_family,
                    'color': self.text_block.color
                }
            )
    def show_suggestions(self, suggestions: List[str], progress_dialog):
        """Show AI suggestions dialog"""
        progress_dialog.close()

        if not suggestions:
            QMessageBox.information(self.scene().views()[0], "اقتراحات", "لا توجد اقتراحات متاحة")
            return

        # Create suggestions dialog
        dialog = QDialog(self.scene().views()[0])
        dialog.setWindowTitle("اقتراحات التحسين")
        dialog.resize(400, 300)

        layout = QVBoxLayout()

        suggestions_list = QListWidget()
        for suggestion in suggestions:
            suggestions_list.addItem(suggestion)

        layout.addWidget(QLabel("اقتراحات لتحسين النص:"))
        layout.addWidget(suggestions_list)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.setLayout(layout)
        dialog.exec_()

    def show_error(self, error_msg: str, progress_dialog):
        """Show error message"""
        progress_dialog.close()
        QMessageBox.warning(self.scene().views()[0], "خطأ", f"حدث خطأ: {error_msg}")

    def reset_position(self):
        """Reset to original position"""
        if self.text_block.original_bbox:
            self.setPos(self.text_block.original_bbox[0], self.text_block.original_bbox[1])
        self.setTransform(self.original_transform)

    def duplicate_item(self):
        """Create a duplicate of this item"""
        # Create new text block
        new_block = TextBlock(
            text=self.text_block.text,
            bbox=[self.pos().x() + 20, self.pos().y() + 20,
                  self.text_block.bbox[2], self.text_block.bbox[3]],
            font_family=self.text_block.font_family,
            font_size=self.text_block.font_size,
            font_weight=self.text_block.font_weight,
            font_style=self.text_block.font_style,
            color=self.text_block.color,
            alignment=self.text_block.alignment,
            line_height=self.text_block.line_height,
            char_spacing=self.text_block.char_spacing,
            word_spacing=self.text_block.word_spacing,
            original_bbox=self.text_block.original_bbox.copy() if self.text_block.original_bbox else self.text_block.bbox.copy(),
            page_number=self.text_block.page_number,
            block_id=f"{self.text_block.block_id}_copy_{int(time.time())}",
            rotation=self.text_block.rotation,
            opacity=self.text_block.opacity
        )

        # Add to scene
        new_item = AdvancedTextItem(new_block)
        self.scene().addItem(new_item)

    def delete_item(self):
        """حذف العنصر مع تسجيل التغيير"""
        try:
            editor = self.scene().views()[0].window()
            editor.change_tracker.add_change(
                self.text_block.page_number,
                self.text_block.block_id,
                'delete',
                {
                    'original_text': self.text_block.text,
                    'bbox': self.text_block.original_bbox.copy() if self.text_block.original_bbox else self.text_block.bbox.copy(),
                    'font_size': self.text_block.font_size,
                    'font_family': self.text_block.font_family,
                    'color': self.text_block.color
                }
            )
            print(f"✓ تم تسجيل حذف النص: {self.text_block.block_id}")
        except Exception as e:
            print(f"❌ خطأ في تسجيل الحذف: {e}")

        self.scene().removeItem(self)

    def mousePressEvent(self, event):
        self.setOpacity(1.0)
        self.text_block.opacity = 1.0
        """Start editing on single click"""
        if event.button() == Qt.LeftButton:
            # Start editing immediately
            self.setTextInteractionFlags(Qt.TextEditorInteraction)
            self.setFocus(Qt.MouseFocusReason)
            self.is_editing = True
            event.accept()
            return

        super().mousePressEvent(event)

    def paint(self, painter, option, widget):
        """تخصيص الرسم لجعل النص مرئياً فقط عند التحديد أو التحرير"""
        if self.is_editing or self.isSelected() or self.hasFocus():
            # جعل العنصر مرئياً عند التحديد أو التحرير
            self.setOpacity(1.0)
        else:
            # إرجاع الشفافية عندما لا يكون محدداً
            editor = self.get_editor()
            if editor and not editor.edit_mode:
                self.setOpacity(0.0)

        super().paint(painter, option, widget)

    def mouseDoubleClickEvent(self, event):
        """Enter edit mode on double click"""
        # Already handled by mousePressEvent
        event.accept()

    def focusOutEvent(self, event):
        """معالجة محسنة لخروج التركيز مع حفظ أفضل"""
        if self.is_editing:
            self.setTextInteractionFlags(Qt.NoTextInteraction)
            new_text = self.toPlainText().strip()

            # تطبيق معالجة النص العربي
            arabic_handler = ArabicTextHandler()
            processed_text = arabic_handler.process_mixed_text(new_text)

            # إصلاح الترميز
            fixed_text = EncodingFixer.fix_text_encoding(processed_text)

            # التحقق من وجود تغيير فعلي
            if fixed_text != self.original_text:
                print(f"📝 تم اكتشاف تغيير في النص:")
                print(f"   النص الأصلي: '{self.original_text[:50]}...'")
                print(f"   النص الجديد: '{fixed_text[:50]}...'")

                # تحديث بيانات الكتلة
                self.text_block.text = fixed_text
                self.is_modified = True

                try:
                    editor = self.scene().views()[0].window()

                    # حفظ معلومات شاملة للتغيير
                    change_data = {
                        'original_text': self.original_text,
                        'new_text': fixed_text,
                        'bbox': self.text_block.original_bbox.copy() if self.text_block.original_bbox else self.text_block.bbox.copy(),
                        'font_size': self.text_block.font_size,
                        'font_family': self.text_block.font_family,
                        'color': self.text_block.color,
                        'position': [self.pos().x(), self.pos().y()],
                        'text_direction': 'rtl' if arabic_handler.is_mixed_text(fixed_text) or any('\u0600' <= c <= '\u06FF' for c in fixed_text) else 'ltr',
                        'encoding_fixed': processed_text != new_text,  # هل تم إصلاح الترميز
                        'timestamp': time.time()
                    }

                    # تسجيل التغيير بالمعلومات الشاملة
                    editor.change_tracker.add_change(
                        self.text_block.page_number,
                        self.text_block.block_id,
                        'edit',
                        change_data
                    )

                    # تحديث النص الأصلي
                    self.original_text = fixed_text
                    print(f"✅ تم تسجيل التغيير بنجاح للكتلة: {self.text_block.block_id}")

                except Exception as e:
                    print(f"❌ خطأ في تسجيل التغيير: {e}")

            self.is_editing = False

        super().focusOutEvent(event)


    def itemChange(self, change, value):
        """معالجة محسنة لتغيير الموقع"""
        if change == QGraphicsTextItem.ItemPositionChange and self.scene():
            new_pos = value
            old_x, old_y = self.text_block.bbox[0], self.text_block.bbox[1]

            # التحقق من وجود تحريك فعلي
            if abs(old_x - new_pos.x()) > 2 or abs(old_y - new_pos.y()) > 2:
                # تحديث موقع الكتلة
                self.text_block.bbox[0] = new_pos.x()
                self.text_block.bbox[1] = new_pos.y()

                try:
                    editor = self.scene().views()[0].window()

                    # معلومات شاملة للتحريك
                    move_data = {
                        'old_pos': [old_x, old_y],
                        'new_pos': [new_pos.x(), new_pos.y()],
                        'bbox': self.text_block.original_bbox.copy() if self.text_block.original_bbox else [old_x, old_y, self.text_block.bbox[2], self.text_block.bbox[3]],
                        'text': self.text_block.text,
                        'font_size': self.text_block.font_size,
                        'font_family': self.text_block.font_family,
                        'color': self.text_block.color,
                        'movement_distance': ((old_x - new_pos.x()) ** 2 + (old_y - new_pos.y()) ** 2) ** 0.5,
                        'timestamp': time.time()
                    }

                    editor.change_tracker.add_change(
                        self.text_block.page_number,
                        self.text_block.block_id,
                        'move',
                        move_data
                    )
                    print(f"✅ تم تسجيل تحريك الكتلة: {self.text_block.block_id}")
                except Exception as e:
                    print(f"❌ خطأ في تسجيل تحريك الكتلة: {e}")

        return super().itemChange(change, value)


class EnhancedOCRThread(QThread):
    """Enhanced OCR processing with better text detection"""
    finished = pyqtSignal(list, QPixmap, int)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, pdf_path: str, page_index: int, use_ai_enhancement: bool = True):
        super().__init__()
        self.pdf_path = pdf_path
        self.page_index = page_index
        self.use_ai_enhancement = use_ai_enhancement
        self.ai_processor = AITextProcessor() if use_ai_enhancement else None

    def run(self):
        try:
            self.progress.emit(5)

            doc = fitz.open(self.pdf_path)
            page = doc[self.page_index]

            self.progress.emit(15)

            # تحويل صفحة PDF إلى صورة لاستخراج OCR
            mat = fitz.Matrix(3.0, 3.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("ppm")
            pil_image = Image.open(io.BytesIO(img_data))

            self.progress.emit(40)

            # استخراج النصوص باستخدام OCR فقط
            ocr_blocks = self.perform_advanced_ocr(pil_image)

            self.progress.emit(95)

            qimage = self.pil_to_qimage(pil_image)
            pixmap = QPixmap.fromImage(qimage)

            self.progress.emit(100)
            self.finished.emit(ocr_blocks, pixmap, len(doc))

        except Exception as e:
            import traceback
            self.error.emit(f"خطأ في معالجة الصفحة: {str(e)}\n{traceback.format_exc()}")
    def extract_pdf_text_blocks(self, page) -> List[TextBlock]:
        """Extract text blocks from PDF with formatting information"""
        blocks = []
        text_dict = page.get_text("dict")

        block_id = 0
        for block in text_dict["blocks"]:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                for span in line["spans"]:
                    text_content = span["text"].strip()
                    if not text_content:
                        continue

                    # Extract and clean font information
                    raw_font = span.get("font", "Arial")
                    font_family = self.clean_font_name(raw_font)

                    # Get bbox
                    bbox = span["bbox"]  # [x0, y0, x1, y1]
                    width = max(bbox[2] - bbox[0], 10)
                    height = max(bbox[3] - bbox[1], 10)

                    # Create text block
                    text_block = TextBlock(
                        text=text_content,
                        bbox=[bbox[0], bbox[1], width, height],
                        font_family=font_family,
                        font_size=max(6, min(span.get("size", 12), 72)),
                        font_weight="bold" if "Bold" in raw_font else "normal",
                        font_style="italic" if "Italic" in raw_font else "normal",
                        color=self.color_to_hex(span.get("color", 0)),
                        alignment="right" if ImprovedPDFSaver.is_arabic_text(text_content) else "left",
                        line_height=1.2,
                        char_spacing=0.0,
                        word_spacing=0.0,
                        original_bbox=[bbox[0], bbox[1], width, height],
                        page_number=self.page_index,
                        block_id=f"pdf_{block_id}",
                        rotation=0.0,
                        opacity=1.0
                    )

                    blocks.append(text_block)
                    block_id += 1

        return blocks

    def clean_font_name(self, raw_font: str) -> str:
        """Clean font name for safe usage"""
        if not raw_font:
            return "Arial"

        font_name = raw_font.split("-")[0].split("+")[-1].split(",")[0].strip()
        font_name = font_name.replace("Bold", "").replace("Italic", "").strip()

        return font_name if font_name else "Arial"

    def perform_advanced_ocr(self, image: Image.Image) -> List[TextBlock]:
        """Perform advanced OCR with better text detection"""
        cv_image = np.array(image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        denoised = cv2.fastNlMeansDenoising(gray)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        setup_tesseract()

        try:
            ocr_data = pytesseract.image_to_data(
                enhanced,
                lang='ara+eng',
                output_type=pytesseract.Output.DICT,
                config='--oem 3 --psm 6'
            )
        except:
            ocr_data = pytesseract.image_to_data(
                enhanced,
                output_type=pytesseract.Output.DICT
            )

        blocks = []
        block_id = 0

        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            if text and int(ocr_data['conf'][i]) > 30:
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]

                font_size = max(8, h * 0.7)

                text_block = TextBlock(
                    text=text,
                    bbox=[x, y, w, h],
                    font_family="Arial",
                    font_size=font_size,
                    font_weight="normal",
                    font_style="normal",
                    color="#000000",
                    alignment="left",
                    line_height=1.2,
                    char_spacing=0.0,
                    word_spacing=0.0,
                    original_bbox=[x, y, w, h],
                    page_number=self.page_index,
                    block_id=f"ocr_{block_id}",
                    rotation=0.0,
                    opacity=1.0
                )

                blocks.append(text_block)
                block_id += 1

        return blocks


    def merge_text_blocks(self, pdf_blocks: List[TextBlock], ocr_blocks: List[TextBlock]) -> List[TextBlock]:
        """دمج مواقع PDF مع نصوص OCR"""
        merged_blocks = []
        used_ocr_blocks = set()

        for pdf_block in pdf_blocks:
            best_match = None
            best_overlap = 0

            for i, ocr_block in enumerate(ocr_blocks):
                if i in used_ocr_blocks:
                    continue

                overlap = self.calculate_bbox_overlap(pdf_block.bbox, ocr_block.bbox)
                if overlap > best_overlap and overlap > 0.3:  # تقليل عتبة التداخل
                    best_overlap = overlap
                    best_match = i

            if best_match is not None:
                ocr_block = ocr_blocks[best_match]

                # تحديث النص باستخدام OCR مع الحفاظ على موقع وخصائص PDF
                pdf_block.text = ocr_block.text
                merged_blocks.append(pdf_block)
                used_ocr_blocks.add(best_match)
            else:
                # إذا لم نجد تطابقاً، نستخدم النص من PDF
                merged_blocks.append(pdf_block)

        # إضافة كتل OCR المتبقية (نادراً ما يحدث)
        for i, ocr_block in enumerate(ocr_blocks):
            if i not in used_ocr_blocks:
                merged_blocks.append(ocr_block)

        return merged_blocks

    def calculate_bbox_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1

        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def enhance_blocks_with_ai(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Enhance text blocks using AI"""
        enhanced_blocks = []

        for block in blocks:
            try:
                context = self.get_block_context(block, blocks)
                enhanced_text = self.ai_processor.enhance_ocr_text(block.text, context)

                if enhanced_text and enhanced_text.strip():
                    block.text = enhanced_text

                enhanced_blocks.append(block)
            except Exception as e:
                print(f"AI enhancement failed for block: {e}")
                enhanced_blocks.append(block)

        return enhanced_blocks

    def get_block_context(self, target_block: TextBlock, all_blocks: List[TextBlock]) -> str:
        """Get context text from surrounding blocks"""
        context_blocks = []
        target_center = (
            target_block.bbox[0] + target_block.bbox[2] / 2,
            target_block.bbox[1] + target_block.bbox[3] / 2
        )

        for block in all_blocks:
            if block.block_id == target_block.block_id:
                continue

            block_center = (
                block.bbox[0] + block.bbox[2] / 2,
                block.bbox[1] + block.bbox[3] / 2
            )

            distance = ((target_center[0] - block_center[0]) ** 2 +
                       (target_center[1] - block_center[1]) ** 2) ** 0.5

            if distance < 200:
                context_blocks.append((distance, block.text))

        context_blocks.sort(key=lambda x: x[0])
        return " ".join([block[1] for block in context_blocks[:3]])

    def color_to_hex(self, color_value) -> str:
        """Convert color value to hex string"""
        if isinstance(color_value, int):
            return f"#{color_value:06x}"
        elif isinstance(color_value, (list, tuple)) and len(color_value) >= 3:
            r, g, b = color_value[:3]
            return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        else:
            return "#000000"

    def pil_to_qimage(self, pil_image: Image.Image) -> QImage:
        """Convert PIL image to QImage"""
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        np_array = np.array(pil_image)
        height, width, channel = np_array.shape
        bytes_per_line = 3 * width

        return QImage(np_array.data, width, height, bytes_per_line, QImage.Format_RGB888)

class AIConfigDialog(QDialog):
    """Dialog for configuring AI settings"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("إعدادات AI")
        self.setModal(True)
        self.resize(500, 400)

        self.ai_processor = AITextProcessor()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Provider selection
        provider_group = QGroupBox("مزود الخدمة")
        provider_layout = QVBoxLayout()

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["OpenAI GPT-4", "Claude 3", "Google Gemini"])
        self.provider_combo.currentTextChanged.connect(self.on_provider_changed)

        provider_layout.addWidget(QLabel("اختر مزود AI:"))
        provider_layout.addWidget(self.provider_combo)
        provider_group.setLayout(provider_layout)
        layout.addWidget(provider_group)

        # API Key configuration
        api_group = QGroupBox("مفتاح API")
        api_layout = QVBoxLayout()

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("أدخل مفتاح API...")

        self.test_button = QPushButton("اختبار الاتصال")
        self.test_button.clicked.connect(self.test_connection)

        api_layout.addWidget(QLabel("مفتاح API:"))
        api_layout.addWidget(self.api_key_input)
        api_layout.addWidget(self.test_button)
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)

        # Feature settings
        features_group = QGroupBox("الميزات")
        features_layout = QVBoxLayout()

        self.auto_enhance_cb = QCheckBox("تحسين النص تلقائياً")
        self.auto_enhance_cb.setChecked(True)

        self.context_aware_cb = QCheckBox("تحسين حسب السياق")
        self.context_aware_cb.setChecked(True)

        self.preserve_formatting_cb = QCheckBox("المحافظة على التنسيق")
        self.preserve_formatting_cb.setChecked(True)

        features_layout.addWidget(self.auto_enhance_cb)
        features_layout.addWidget(self.context_aware_cb)
        features_layout.addWidget(self.preserve_formatting_cb)
        features_group.setLayout(features_layout)
        layout.addWidget(features_group)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)
        self.load_settings()

    def on_provider_changed(self, provider_text: str):
        """Handle provider change"""
        provider_map = {
            "OpenAI GPT-4": "openai",
            "Claude 3": "claude",
            "Google Gemini": "gemini"
        }

        provider = provider_map.get(provider_text, "openai")
        self.ai_processor.current_provider = provider

        # Load existing API key for this provider
        api_key = self.ai_processor.get_api_key(provider)
        self.api_key_input.setText(api_key)

    def test_connection(self):
        """Test AI connection"""
        provider = self.ai_processor.current_provider
        api_key = self.api_key_input.text().strip()

        if not api_key:
            QMessageBox.warning(self, "تحذير", "يرجى إدخال مفتاح API")
            return

        # Save API key temporarily
        self.ai_processor.set_api_key(provider, api_key)

        # Test with simple prompt
        test_prompt = "مرحبا"

        progress = QProgressDialog("جاري اختبار الاتصال...", "إلغاء", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        def test_thread():
            try:
                result = self.ai_processor._call_ai_api(test_prompt)
                QTimer.singleShot(0, lambda: self.show_test_result(True, result, progress))
            except Exception as e:
                QTimer.singleShot(0, lambda: self.show_test_result(False, str(e), progress))

        threading.Thread(target=test_thread, daemon=True).start()

    def show_test_result(self, success: bool, message: str, progress_dialog):
        """Show test connection result"""
        progress_dialog.close()

        if success:
            QMessageBox.information(self, "نجح الاختبار", "تم الاتصال بنجاح!")
        else:
            QMessageBox.critical(self, "فشل الاختبار", f"فشل في الاتصال:\n{message}")

    def load_settings(self):
        """Load saved settings"""
        settings = QSettings()

        # Load provider
        provider = settings.value("ai/current_provider", "openai")
        provider_map = {
            "openai": "OpenAI GPT-4",
            "claude": "Claude 3",
            "gemini": "Google Gemini"
        }

        provider_text = provider_map.get(provider, "OpenAI GPT-4")
        index = self.provider_combo.findText(provider_text)
        if index >= 0:
            self.provider_combo.setCurrentIndex(index)

        # Load features
        self.auto_enhance_cb.setChecked(
            settings.value("ai/auto_enhance", True, type=bool)
        )
        self.context_aware_cb.setChecked(
            settings.value("ai/context_aware", True, type=bool)
        )
        self.preserve_formatting_cb.setChecked(
            settings.value("ai/preserve_formatting", True, type=bool)
        )

    def accept(self):
        """Save settings and accept"""
        # Save API key
        provider = self.ai_processor.current_provider
        api_key = self.api_key_input.text().strip()
        self.ai_processor.set_api_key(provider, api_key)

        # Save other settings
        settings = QSettings()
        settings.setValue("ai/current_provider", provider)
        settings.setValue("ai/auto_enhance", self.auto_enhance_cb.isChecked())
        settings.setValue("ai/context_aware", self.context_aware_cb.isChecked())
        settings.setValue("ai/preserve_formatting", self.preserve_formatting_cb.isChecked())

        super().accept()

class ChangeTracker:
    """نظام محسن لتتبع التغييرات"""
    def __init__(self):
        self.changes = []
        self.original_blocks = {}  # حفظ الكتل الأصلية
        self.text_modifications = {}  # تتبع تعديلات النصوص
        self.deleted_blocks = {}  # تتبع المحذوفات
    def save_original_state(self, page_num, block_id, original_data):
        """حفظ الحالة الأصلية مع معلومات شاملة"""
        key = f"{page_num}_{block_id}"
        if key not in self.original_blocks:
            self.original_blocks[key] = {
                'text': original_data.get('text', ''),
                'bbox': original_data.get('bbox', []).copy() if original_data.get('bbox') else [0, 0, 0, 0],
                'font_size': original_data.get('font_size', 12),
                'font_family': original_data.get('font_family', 'Arial'),
                'color': original_data.get('color', '#000000'),
                'original_bbox': original_data.get('bbox', []).copy() if original_data.get('bbox') else [0, 0, 0, 0],
                'page_num': page_num,
                'block_id': block_id,
                'timestamp': time.time()
            }
            print(f"💾 تم حفظ الحالة الأصلية للكتلة: {block_id}")
    def add_change(self, page_index, block_id, change_type, data):
        """إضافة تغيير مع معلومات شاملة"""
        # التحقق من التغييرات المتكررة وتحديثها
        existing_change = None
        for i, change in enumerate(self.changes):
            if (change['page'] == page_index and
                change['block_id'] == block_id and
                change['type'] == change_type):
                existing_change = i
                break

        # إنشاء بيانات التغيير الشاملة
        change_data = {
            'page': page_index,
            'block_id': block_id,
            'type': change_type,
            'data': data.copy(),  # نسخة كاملة من البيانات
            'timestamp': time.time()
        }

        if existing_change is not None:
            # تحديث التغيير الموجود
            self.changes[existing_change] = change_data
            print(f"🔄 تم تحديث التغيير: {change_type} للكتلة {block_id}")
        else:
            # إضافة تغيير جديد
            self.changes.append(change_data)
            print(f"➕ تم إضافة تغيير جديد: {change_type} للكتلة {block_id}")

        # تحديث قاموس التعديلات النصية
        if change_type in ['edit', 'move', 'add']:
            key = f"{page_index}_{block_id}"
            self.text_modifications[key] = {
                'original_text': data.get('original_text', ''),
                'new_text': data.get('new_text', data.get('text', '')),
                'bbox': data.get('bbox', [0, 0, 0, 0]).copy(),
                'font_size': data.get('font_size', 12),
                'font_family': data.get('font_family', 'Arial'),
                'color': data.get('color', '#000000'),
                'change_type': change_type,
                'position': data.get('position', data.get('new_pos', [0, 0]))
            }

        # تتبع الحذف
        elif change_type == 'delete':
            key = f"{page_index}_{block_id}"
            self.deleted_blocks[key] = {
                'original_text': data.get('original_text', ''),
                'bbox': data.get('bbox', [0, 0, 0, 0]),
                'timestamp': time.time()
            }
    def get_comprehensive_changes(self, page_index):
        """الحصول على تغييرات شاملة للصفحة"""
        page_changes = [change for change in self.changes if change['page'] == page_index]
        page_modifications = {k: v for k, v in self.text_modifications.items()
                            if k.startswith(f"{page_index}_")}
        page_deletions = {k: v for k, v in self.deleted_blocks.items()
                         if k.startswith(f"{page_index}_")}

        return {
            'changes': page_changes,
            'modifications': page_modifications,
            'deletions': page_deletions,
            'total_count': len(page_changes)
        }
    def get_changes_for_page(self, page_index):
        """الحصول على التغييرات لصفحة معينة"""
        page_changes = [change for change in self.changes if change['page'] == page_index]
        print(f"✓ تم العثور على {len(page_changes)} تغيير للصفحة {page_index}")
        return page_changes

    def get_text_modifications_for_page(self, page_index):
        """الحصول على تعديلات النصوص لصفحة معينة"""
        modifications = {}
        for key, mod in self.text_modifications.items():
            if key.startswith(f"{page_index}_"):
                modifications[key] = mod
        return modifications

    def clear_page_changes(self, page_index):
        """مسح تغييرات صفحة مع الاحتفاظ بالنسخ الاحتياطية"""
        # نسخ احتياطية قبل المسح
        backup_changes = [change for change in self.changes if change['page'] == page_index]

        # مسح التغييرات
        self.changes = [change for change in self.changes if change['page'] != page_index]

        # مسح التعديلات النصية
        keys_to_remove = [key for key in self.text_modifications.keys()
                         if key.startswith(f"{page_index}_")]
        for key in keys_to_remove:
            del self.text_modifications[key]

        # مسح المحذوفات
        keys_to_remove = [key for key in self.deleted_blocks.keys()
                         if key.startswith(f"{page_index}_")]
        for key in keys_to_remove:
            del self.deleted_blocks[key]

        print(f"🗑️ تم مسح {len(backup_changes)} تغيير للصفحة {page_index}")

    def clear(self):
        """مسح جميع التغييرات"""
        self.changes = []
        self.original_blocks = {}
        self.text_modifications = {}



class PDFSaveDiagnostic:
    """أداة تشخيص مشاكل حفظ PDF"""

    @staticmethod
    def diagnose_save_issues(original_doc, output_path, changes):
        """تشخيص مشاكل الحفظ"""
        report = {
            'original_doc_valid': original_doc is not None,
            'output_path_writable': os.access(os.path.dirname(output_path), os.W_OK),
            'changes_count': len(changes),
            'file_size_before': 0,
            'file_size_after': 0,
            'encoding_issues': [],
            'font_issues': [],
            'recommendations': []
        }

        try:
            # فحص الملف الأصلي
            if original_doc:
                report['original_pages'] = len(original_doc)
                report['file_size_before'] = os.path.getsize(original_doc.name) if hasattr(original_doc, 'name') else 0

            # فحص التغييرات
            for change in changes:
                data = change.get('data', {})
                text = data.get('new_text', data.get('text', ''))

                # فحص مشاكل الترميز
                try:
                    text.encode('utf-8')
                except UnicodeEncodeError as e:
                    report['encoding_issues'].append(f"Block {change['block_id']}: {str(e)}")

                # فحص الخطوط
                font_family = data.get('font_family', '')
                if font_family and font_family not in ['Arial', 'Times', 'Courier', 'Helvetica']:
                    report['font_issues'].append(f"Unknown font: {font_family} in block {change['block_id']}")

            # فحص الملف بعد الحفظ
            if os.path.exists(output_path):
                report['file_size_after'] = os.path.getsize(output_path)
                report['save_successful'] = report['file_size_after'] > 1000

            # توصيات
            if report['encoding_issues']:
                report['recommendations'].append("استخدم EncodingFixer لحل مشاكل الترميز")

            if report['font_issues']:
                report['recommendations'].append("تأكد من استخدام خطوط مدعومة في PyMuPDF")

            if report['file_size_after'] < 1000:
                report['recommendations'].append("الملف المحفوظ صغير جداً - قد تكون هناك مشكلة في الحفظ")

            return report

        except Exception as e:
            report['diagnostic_error'] = str(e)
            return report
# ==============================================================
# MAIN APPLICATION
# ==============================================================
class AdvancedPDFEditor(QMainWindow):
    """Main application window with Adobe-like interface"""
    def __init__(self):
        super().__init__()
        self.setup_app()
        self.scene.selectionChanged.connect(self.selectionChanged)
        self.change_tracker = ChangeTracker()

    def setup_app(self):
        """Initialize application settings and UI"""
        self.setWindowTitle("محرر PDF المتقدم - نسخة Adobe مطورة")
        self.setMinimumSize(1400, 900)
        self.init_state()
        self.init_ui()
        self.setup_shortcuts()
        self.setup_theme()
        self.load_settings()

    def init_state(self):
        """Initialize application state variables"""
        self.pdf_path = None
        self.doc = None
        self.current_page = 0
        self.total_pages = 0
        self.zoom_level = 1.0
        self.text_blocks = []
        self.history_stack = []  # For undo/redo
        self.history_index = -1
        self.scene = QGraphicsScene()
        self.graphics_view = QGraphicsView(self.scene)
        self.ai_processor = AITextProcessor()
        self.edit_mode = False

    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.create_menu_bar()
        self.create_toolbar()
        self.create_status_bar()

        left_sidebar = self.create_left_sidebar()
        center_widget = self.create_center_widget()
        right_sidebar = self.create_right_sidebar()

        main_layout.addWidget(left_sidebar)
        main_layout.addWidget(center_widget)
        main_layout.addWidget(right_sidebar)

        main_layout.setStretch(0, 1)  # Left sidebar
        main_layout.setStretch(1, 4)  # Center (PDF viewer)
        main_layout.setStretch(2, 2)  # Right sidebar
    def selectionChanged(self):
      """عند تغيير التحديد، نجعل العناصر المحددة مرئية"""
      for item in self.scene.items():
          if isinstance(item, AdvancedTextItem):
              if item.isSelected():
                  item.setOpacity(1.0)
              elif not self.edit_mode:
                  item.setOpacity(0.0)
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('ملف')

        open_action = QAction('فتح PDF', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_pdf)
        file_menu.addAction(open_action)

        save_action = QAction('حفظ', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_pdf)
        file_menu.addAction(save_action)

        save_as_action = QAction('حفظ باسم', self)
        save_as_action.setShortcut('Ctrl+Shift+S')
        save_as_action.triggered.connect(self.save_pdf_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        exit_action = QAction('خروج', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu('تحرير')

        undo_action = QAction('تراجع', self)
        undo_action.setShortcut('Ctrl+Z')
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction('إعادة', self)
        redo_action.setShortcut('Ctrl+Y')
        redo_action.triggered.connect(self.redo)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        select_all_action = QAction('تحديد الكل', self)
        select_all_action.setShortcut('Ctrl+A')
        select_all_action.triggered.connect(self.select_all_text)
        edit_menu.addAction(select_all_action)

        # View menu
        view_menu = menubar.addMenu('عرض')

        zoom_in_action = QAction('تكبير', self)
        zoom_in_action.setShortcut('Ctrl++')
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction('تصغير', self)
        zoom_out_action.setShortcut('Ctrl+-')
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)

        fit_width_action = QAction('ملائمة العرض', self)
        fit_width_action.triggered.connect(self.fit_width)
        view_menu.addAction(fit_width_action)

        fit_page_action = QAction('ملائمة الصفحة', self)
        fit_page_action.triggered.connect(self.fit_page)
        view_menu.addAction(fit_page_action)

        # AI menu
        ai_menu = menubar.addMenu('AI')

        ai_config_action = QAction('إعدادات AI', self)
        ai_config_action.triggered.connect(self.show_ai_config)
        ai_menu.addAction(ai_config_action)

        enhance_all_action = QAction('تحسين جميع النصوص', self)
        enhance_all_action.triggered.connect(self.enhance_all_text)
        ai_menu.addAction(enhance_all_action)

        translate_page_action = QAction('ترجمة الصفحة', self)
        translate_page_action.triggered.connect(self.translate_current_page)
        ai_menu.addAction(translate_page_action)

    def create_toolbar(self):
        """Create application toolbar"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        # File operations
        open_btn = QPushButton("📂 فتح")
        open_btn.clicked.connect(self.open_pdf)
        toolbar.addWidget(open_btn)

        save_btn = QPushButton("💾 حفظ")
        save_btn.clicked.connect(self.save_pdf)
        toolbar.addWidget(save_btn)

        toolbar.addSeparator()

        # Navigation
        prev_btn = QPushButton("⬅️")
        prev_btn.clicked.connect(self.prev_page)
        toolbar.addWidget(prev_btn)

        self.page_input = QSpinBox()
        self.page_input.setMinimum(1)
        self.page_input.valueChanged.connect(self.goto_page)
        toolbar.addWidget(self.page_input)

        self.page_label = QLabel("/ 0")
        toolbar.addWidget(self.page_label)

        next_btn = QPushButton("➡️")
        next_btn.clicked.connect(self.next_page)
        toolbar.addWidget(next_btn)

        toolbar.addSeparator()

        # Zoom controls
        zoom_out_btn = QPushButton("🔍➖")
        zoom_out_btn.clicked.connect(self.zoom_out)
        toolbar.addWidget(zoom_out_btn)

        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(["25%", "50%", "75%", "100%", "125%", "150%", "200%", "400%"])
        self.zoom_combo.setCurrentText("100%")
        self.zoom_combo.currentTextChanged.connect(self.set_zoom_level)
        toolbar.addWidget(self.zoom_combo)

        zoom_in_btn = QPushButton("🔍➕")
        zoom_in_btn.clicked.connect(self.zoom_in)
        toolbar.addWidget(zoom_in_btn)

        toolbar.addSeparator()

        # Text tools
        add_text_btn = QPushButton("➕ إضافة نص")
        add_text_btn.clicked.connect(self.add_text_block)
        toolbar.addWidget(add_text_btn)

        # AI enhancement
        ai_enhance_btn = QPushButton("🤖 تحسين AI")
        ai_enhance_btn.clicked.connect(self.enhance_selected_text)
        toolbar.addWidget(ai_enhance_btn)

        # Edit mode toggle
        self.edit_mode_btn = QPushButton("✏️ وضع التحرير")
        self.edit_mode_btn.setCheckable(True)
        self.edit_mode_btn.setChecked(False)
        self.edit_mode_btn.toggled.connect(self.toggle_edit_mode)
        toolbar.addWidget(self.edit_mode_btn)

    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Progress bar for operations
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("جاهز")
        self.status_bar.addWidget(self.status_label)

    def create_left_sidebar(self) -> QWidget:
        """Create left sidebar with tools"""
        sidebar = QFrame()
        sidebar.setMaximumWidth(250)
        sidebar.setFrameStyle(QFrame.StyledPanel)

        layout = QVBoxLayout()
        sidebar.setLayout(layout)

        # Pages panel
        pages_group = QGroupBox("الصفحات")
        pages_layout = QVBoxLayout()

        self.pages_list = QListWidget()
        self.pages_list.currentItemChanged.connect(self.on_page_selected)
        pages_layout.addWidget(self.pages_list)

        pages_group.setLayout(pages_layout)
        layout.addWidget(pages_group)

        # Properties panel
        props_group = QGroupBox("خصائص النص")
        props_layout = QGridLayout()

        # Font controls
        props_layout.addWidget(QLabel("الخط:"), 0, 0)
        self.font_combo = QComboBox()
        self.font_combo.addItems(["Arial", "Times New Roman", "Calibri", "Tahoma"])
        props_layout.addWidget(self.font_combo, 0, 1)

        props_layout.addWidget(QLabel("الحجم:"), 1, 0)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 72)
        self.font_size_spin.setValue(12)
        props_layout.addWidget(self.font_size_spin, 1, 1)

        # Color picker
        props_layout.addWidget(QLabel("اللون:"), 2, 0)
        self.color_btn = QPushButton()
        self.color_btn.setStyleSheet("background-color: black")
        self.color_btn.clicked.connect(self.choose_text_color)
        props_layout.addWidget(self.color_btn, 2, 1)

        # Style checkboxes
        self.bold_cb = QCheckBox("عريض")
        self.italic_cb = QCheckBox("مائل")
        props_layout.addWidget(self.bold_cb, 3, 0)
        props_layout.addWidget(self.italic_cb, 3, 1)

        props_group.setLayout(props_layout)
        layout.addWidget(props_group)

        # Search panel
        search_group = QGroupBox("البحث والاستبدال")
        search_layout = QVBoxLayout()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("البحث...")
        search_layout.addWidget(self.search_input)

        self.replace_input = QLineEdit()
        self.replace_input.setPlaceholderText("استبدال...")
        search_layout.addWidget(self.replace_input)

        search_buttons_layout = QGridLayout()

        find_btn = QPushButton("بحث")
        find_btn.clicked.connect(self.find_text)
        search_buttons_layout.addWidget(find_btn, 0, 0)

        replace_one_btn = QPushButton("استبدال أول نتيجة")
        replace_one_btn.clicked.connect(lambda: self.replace_text(replace_all=False))
        search_buttons_layout.addWidget(replace_one_btn, 0, 1)

        replace_all_btn = QPushButton("استبدال الكل")
        replace_all_btn.clicked.connect(lambda: self.replace_text(replace_all=True))
        search_buttons_layout.addWidget(replace_all_btn, 1, 0, 1, 2)

        search_layout.addLayout(search_buttons_layout)

        search_group.setLayout(search_layout)
        layout.addWidget(search_group)

        layout.addStretch()

        return sidebar

    def create_center_widget(self) -> QWidget:
        """Create center widget with PDF viewer"""
        center_widget = QFrame()
        center_widget.setFrameStyle(QFrame.StyledPanel)

        layout = QVBoxLayout()
        center_widget.setLayout(layout)

        # PDF viewer
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.graphics_view.setDragMode(QGraphicsView.RubberBandDrag)
        self.graphics_view.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.graphics_view)

        return center_widget

    def create_right_sidebar(self) -> QWidget:
        """Create right sidebar with AI tools and text editor"""
        sidebar = QFrame()
        sidebar.setMaximumWidth(350)
        sidebar.setFrameStyle(QFrame.StyledPanel)

        layout = QVBoxLayout()
        sidebar.setLayout(layout)

        # AI Tools
        ai_group = QGroupBox("أدوات AI المتقدمة")
        ai_layout = QVBoxLayout()

        # Quick AI actions
        enhance_btn = QPushButton("🚀 تحسين النص المحدد")
        enhance_btn.clicked.connect(self.enhance_selected_text)
        ai_layout.addWidget(enhance_btn)

        translate_btn = QPushButton("🌍 ترجمة المحدد")
        translate_btn.clicked.connect(self.translate_selected_text)
        ai_layout.addWidget(translate_btn)

        summarize_btn = QPushButton("📝 تلخيص الصفحة")
        summarize_btn.clicked.connect(self.summarize_page)
        ai_layout.addWidget(summarize_btn)

        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)

        # Text editor
        editor_group = QGroupBox("محرر النص")
        editor_layout = QVBoxLayout()

        self.text_editor = QTextEdit()
        self.text_editor.setPlaceholderText("النص المستخرج سيظهر هنا...")
        editor_layout.addWidget(self.text_editor)

        # Text editor buttons
        editor_buttons_layout = QHBoxLayout()

        apply_btn = QPushButton("تطبيق التغييرات")
        apply_btn.clicked.connect(self.apply_text_changes)
        editor_buttons_layout.addWidget(apply_btn)

        reset_btn = QPushButton("إعادة تعيين")
        reset_btn.clicked.connect(self.reset_text_editor)
        editor_buttons_layout.addWidget(reset_btn)

        editor_layout.addLayout(editor_buttons_layout)
        editor_group.setLayout(editor_layout)
        layout.addWidget(editor_group)

        # OCR Settings
        ocr_group = QGroupBox("إعدادات OCR")
        ocr_layout = QVBoxLayout()

        self.language_combo = QComboBox()
        self.language_combo.addItems(["العربية + الإنجليزية", "العربية فقط", "الإنجليزية فقط"])
        ocr_layout.addWidget(QLabel("اللغة:"))
        ocr_layout.addWidget(self.language_combo)

        self.auto_enhance_cb = QCheckBox("تحسين تلقائي بـ AI")
        self.auto_enhance_cb.setChecked(True)
        ocr_layout.addWidget(self.auto_enhance_cb)

        reprocess_btn = QPushButton("إعادة معالجة الصفحة")
        reprocess_btn.clicked.connect(self.reprocess_current_page)
        ocr_layout.addWidget(reprocess_btn)

        ocr_group.setLayout(ocr_layout)
        layout.addWidget(ocr_group)

        return sidebar

    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        from PyQt5.QtWidgets import QShortcut

        # Navigation shortcuts
        QShortcut(QKeySequence("Ctrl+Right"), self, self.next_page)
        QShortcut(QKeySequence("Ctrl+Left"), self, self.prev_page)

        # Edit shortcuts
        QShortcut(QKeySequence("Delete"), self, self.delete_selected)
        QShortcut(QKeySequence("Ctrl+D"), self, self.duplicate_selected)

        # AI shortcuts
        QShortcut(QKeySequence("Ctrl+E"), self, self.enhance_selected_text)
        QShortcut(QKeySequence("Ctrl+T"), self, self.translate_selected_text)

        # Edit mode shortcut
        QShortcut(QKeySequence("Ctrl+E"), self, self.toggle_edit_mode)

        # Search shortcuts
        QShortcut(QKeySequence("Ctrl+F"), self, self.search_input.setFocus)
        QShortcut(QKeySequence("F3"), self, self.next_search_result)
        QShortcut(QKeySequence("Shift+F3"), self, self.prev_search_result)

    def setup_theme(self):
        """Setup dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QMenuBar {
                background-color: #404040;
                color: #ffffff;
                border-bottom: 1px solid #555555;
            }
            QMenuBar::item:selected {
                background-color: #505050;
            }
            QToolBar {
                background-color: #404040;
                border: none;
                spacing: 3px;
            }
            QPushButton {
                background-color: #505050;
                border: 1px solid #666666;
                border-radius: 4px;
                padding: 6px 12px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #606060;
            }
            QPushButton:pressed {
                background-color: #404040;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTextEdit, QLineEdit, QSpinBox, QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px;
                color: #ffffff;
            }
            QListWidget {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                alternate-background-color: #404040;
            }
            QGraphicsView {
                background-color: #2b2b2b;
                border: 1px solid #555555;
            }
            QStatusBar {
                background-color: #404040;
                border-top: 1px solid #555555;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)

    def load_settings(self):
        """Load application settings"""
        settings = QSettings()

        # Window geometry
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        # Last opened file
        last_file = settings.value("last_file")
        if last_file and os.path.exists(last_file):
            QTimer.singleShot(100, lambda: self.open_pdf_file(last_file))

    def save_settings(self):
        """Save application settings"""
        settings = QSettings()
        settings.setValue("geometry", self.saveGeometry())
        if self.pdf_path:
            settings.setValue("last_file", self.pdf_path)

    def closeEvent(self, event):
        """Handle application close"""
        self.save_settings()
        super().closeEvent(event)

    # ==============================================================
    # PDF OPERATIONS
    # ==============================================================
    def open_pdf(self):
        """Open PDF file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "فتح ملف PDF", "", "PDF Files (*.pdf)"
        )

        if file_path:
            self.open_pdf_file(file_path)

    def open_pdf_file(self, file_path: str):
        """Open specific PDF file"""
        try:
            if self.doc:
                self.doc.close()

            self.doc = fitz.open(file_path)
            self.pdf_path = file_path
            self.total_pages = len(self.doc)
            self.current_page = 0

            # Update UI
            self.page_input.setMaximum(self.total_pages)
            self.page_input.setValue(1)
            self.page_label.setText(f"/ {self.total_pages}")

            # Update pages list
            self.update_pages_list()

            # Load first page
            self.load_current_page()

            self.status_label.setText(f"تم فتح: {os.path.basename(file_path)}")

        except Exception as e:
            QMessageBox.critical(self, "خطأ", f"فشل في فتح الملف: {str(e)}")

    def update_pages_list(self):
        """Update pages list widget"""
        self.pages_list.clear()

        for i in range(self.total_pages):
            item = QListWidgetItem(f"الصفحة {i + 1}")

            # Add thumbnail (simplified)
            try:
                page = self.doc[i]
                mat = fitz.Matrix(0.2, 0.2)  # Small thumbnail
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("ppm")

                pil_img = Image.open(io.BytesIO(img_data))
                qimage = self.pil_to_qimage(pil_img)
                pixmap = QPixmap.fromImage(qimage)

                icon = QIcon(pixmap)
                item.setIcon(icon)

            except Exception as e:
                print(f"Thumbnail error: {e}")

            self.pages_list.addItem(item)

    def load_current_page(self):
        """Load and process current page"""
        if not self.doc:
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("جاري تحميل الصفحة...")

        # Clear scene
        self.scene.clear()
        self.text_blocks = []

        # Start OCR thread
        use_ai = self.auto_enhance_cb.isChecked()
        self.ocr_thread = EnhancedOCRThread(self.pdf_path, self.current_page, use_ai)
        self.ocr_thread.finished.connect(self.on_page_loaded)
        self.ocr_thread.error.connect(self.on_load_error)
        self.ocr_thread.progress.connect(self.progress_bar.setValue)
        self.ocr_thread.start()

    def on_page_loaded(self, blocks: List[TextBlock], pixmap: QPixmap, total_pages: int):
        """Handle page loading completion with OCR only"""
        # Add background image
        self.scene.addPixmap(pixmap)

        # Add text blocks (OCR only)
        self.text_blocks = blocks

        for block in blocks:
            text_item = AdvancedTextItem(block)
            text_item.setOpacity(0.0)  # إخفاء النص افتراضيًا
            self.scene.addItem(text_item)

        # Update text editor with all text
        all_text = "\n".join([block.text for block in blocks])
        self.text_editor.setText(all_text)

        # Fit view
        self.graphics_view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

        self.progress_bar.setVisible(False)
        self.status_label.setText(f"تم تحميل الصفحة {self.current_page + 1}")

        # Save state for undo
        self.save_state()

    def on_load_error(self, error_msg: str):
        """Handle page loading error"""
        self.progress_bar.setVisible(False)
        self.status_label.setText("خطأ في التحميل")
        QMessageBox.critical(self, "خطأ", error_msg)

    def save_pdf(self):
        """Save current PDF"""
        if not self.pdf_path:
            self.save_pdf_as()
            return

        self.save_pdf_to_file(self.pdf_path)

    def save_pdf_as(self):
        """Save PDF with new name"""
        if not self.doc:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "حفظ PDF", "", "PDF Files (*.pdf)"
        )

        if file_path:
            self.save_pdf_to_file(file_path)

    def save_pdf_to_file(self, file_path: str):
        try:
            if not self.doc:
                QMessageBox.warning(self, "خطأ", "لا يوجد مستند للحفظ")
                return False

            print(f"🚀 بدء عملية الحفظ المحسنة إلى: {file_path}")

            # إنشاء كائن الحفظ المحسن
            pdf_saver = ImprovedPDFSaver()

            # تجميع النصوص الحالية من المشهد
            current_text_blocks = []
            for item in self.scene.items():
                if isinstance(item, AdvancedTextItem):
                    # تحديث بيانات الكتلة بالقيم الحالية
                    block = item.text_block
                    block.text = item.toPlainText()
                    block.bbox[0] = item.pos().x()
                    block.bbox[1] = item.pos().y()
                    current_text_blocks.append(block)

            # 🔍 طباعة تشخيص النصوص قبل الحفظ
            print("📄 النصوص التي سيتم حفظها:")
            for block in current_text_blocks:
                print(f" - [{block.block_id}] النص: '{block.text[:30]}...' | الموقع: {block.bbox} | الخط: {block.font_family} | الحجم: {block.font_size}")

            # 🔍 طباعة التغييرات المسجلة في change_tracker
            print("🧾 تغييرات مسجلة:")
            for change in self.change_tracker.get_changes_for_page(self.current_page):
                print(f" - {change['type']} | block_id: {change['block_id']} | نص جديد: '{change['data'].get('new_text', '')[:30]}...'")

            # تنفيذ الحفظ الفعلي
            success = pdf_saver.save_pdf_with_modifications(
                self.doc, file_path, self.change_tracker,
                self.current_page, current_text_blocks
            )

            if success:
                if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:
                    QMessageBox.information(self, "تم الحفظ", "تم حفظ التعديلات بنجاح!")
                    self.change_tracker.clear_page_changes(self.current_page)
                    return True
                else:
                    QMessageBox.warning(self, "تحذير", "تم إنشاء الملف لكن حجمه غير كافٍ")
                    return False
            else:
                QMessageBox.critical(self, "خطأ", "فشل في حفظ الملف")
                return False

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ خطأ عام في الحفظ: {error_details}")
            QMessageBox.critical(self, "خطأ في الحفظ", f"حدث خطأ أثناء الحفظ:\n{str(e)}")
            return False


    def apply_changes_to_page(self, original_page, new_page, page_num):
        """تطبيق التغييرات على صفحة معينة"""
        try:
            print(f"تطبيق التغييرات على الصفحة {page_num}")

            # نسخ الصفحة الأصلية أولاً
            new_page.show_pdf_page(original_page.rect, self.doc, page_num)

            # الحصول على التغييرات لهذه الصفحة
            page_changes = self.change_tracker.get_changes_for_page(page_num)

            if not page_changes:
                print("لا توجد تغييرات لتطبيقها")
                return True

            # تجميع مناطق الحذف والنصوص الجديدة
            redaction_areas = []
            text_insertions = []

            for change in page_changes:
                change_type = change['type']
                change_data = change['data']
                block_id = change['block_id']

                print(f"معالجة تغيير: {change_type} للكتلة {block_id}")

                if change_type in ['edit', 'move']:
                    # البحث عن الكتلة الأصلية
                    original_block = self.find_original_block(block_id)
                    if original_block:
                        # إضافة منطقة للحذف
                        bbox = original_block.original_bbox
                        redact_rect = fitz.Rect(
                            bbox[0] - 2, bbox[1] - 2,
                            bbox[0] + bbox[2] + 2, bbox[1] + bbox[3] + 2
                        )
                        redaction_areas.append(redact_rect)

                        # إضافة النص الجديد للإدراج
                        if change_type == 'edit':
                            new_text = change_data.get('new_text', original_block.text)
                            insert_pos = [bbox[0], bbox[1] + bbox[3] * 0.8]
                        else:  # move
                            new_text = original_block.text
                            new_pos = change_data.get('new_pos', [bbox[0], bbox[1]])
                            insert_pos = [new_pos[0], new_pos[1] + bbox[3] * 0.8]

                        text_insertions.append({
                            'text': new_text,
                            'position': insert_pos,
                            'font_size': original_block.font_size,
                            'color': original_block.color,
                            'font_family': original_block.font_family
                        })

                elif change_type == 'delete':
                    # حذف فقط - إضافة منطقة للحذف
                    bbox = change_data.get('bbox', [0, 0, 0, 0])
                    redact_rect = fitz.Rect(
                        bbox[0] - 2, bbox[1] - 2,
                        bbox[0] + bbox[2] + 2, bbox[1] + bbox[3] + 2
                    )
                    redaction_areas.append(redact_rect)

            # تطبيق عمليات الحذف
            if redaction_areas:
                print(f"تطبيق {len(redaction_areas)} عملية حذف")
                for rect in redaction_areas:
                    new_page.add_redact_annot(rect, fill=(1, 1, 1))  # تعبئة بيضاء
                new_page.apply_redactions()

            # إدراج النصوص الجديدة
            if text_insertions:
                print(f"إدراج {len(text_insertions)} نص جديد")
                for insertion in text_insertions:
                    try:
                        # تحديد الخط المناسب
                        font_name = self.get_safe_font_name(insertion['font_family'])

                        # تحويل اللون
                        color = self.hex_to_rgb(insertion['color'])

                        # إدراج النص
                        point = fitz.Point(insertion['position'][0], insertion['position'][1])
                        result = new_page.insert_text(
                            point,
                            insertion['text'],
                            fontsize=max(8, min(insertion['font_size'], 72)),
                            fontname=font_name,
                            color=color
                        )

                        if True:
                            print(f"تم إدراج النص بنجاح: {insertion['text'][:30]}...")
                        else:
                            print(f"فشل في إدراج النص: {insertion['text'][:30]}...")

                    except Exception as e:
                        print(f"خطأ في إدراج النص: {e}")
                        # محاولة إدراج مع إعدادات افتراضية
                        try:
                            new_page.insert_text(
                                point,
                                insertion['text'],
                                fontsize=12,
                                color=(0, 0, 0)
                            )
                        except:
                            print(f"فشل نهائياً في إدراج النص: {insertion['text'][:30]}...")

            print("تم الانتهاء من تطبيق التغييرات")
            return True

        except Exception as e:
            print(f"خطأ في تطبيق التغييرات: {e}")
            import traceback
            traceback.print_exc()
            return False

    def find_original_block(self, block_id):
        """الحصول على النص الأصلي للكتلة مع التعامل مع الحالات الخاصة"""
        for block in self.text_blocks:
            if block.block_id == block_id:
                # استبدال المسافات غير القابلة للكسر بمسافات عادية
                return block
        return None

    def get_safe_font_name(self, font_family):
        """الحصول على اسم خط آمن"""
        font_mapping = {
            'Arial': 'helv',
            'Times': 'tiro',
            'Courier': 'cour',
            'Helvetica': 'helv'
        }
        return font_mapping.get(font_family, 'helv')

    def safe_save_document(self, doc, file_path):
        """حفظ آمن للمستند"""
        try:
            # طريقة 1: الحفظ المباشر
            doc.save(
                file_path,
                garbage=4,
                deflate=True,
                clean=True,
                ascii=False
            )
            print("تم الحفظ بنجاح")
            return True

        except Exception as e:
            print(f"فشل الحفظ المباشر: {e}")

            # طريقة 2: الحفظ عبر ملف مؤقت
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_path = temp_file.name

                doc.save(temp_path)

                # نقل الملف المؤقت إلى الهدف
                if os.path.exists(file_path):
                    os.remove(file_path)
                shutil.move(temp_path, file_path)

                print("تم الحفظ عبر الملف المؤقت")
                return True

            except Exception as e2:
                print(f"فشل الحفظ عبر الملف المؤقت: {e2}")
                return False
    def apply_text_modifications(self, page):
        """Apply text modifications to PDF page"""
        # Get all text items from scene
        for item in self.scene.items():
            if isinstance(item, AdvancedTextItem):
                block = item.text_block
                current_text = item.toPlainText()

                # Create redaction rectangle
                bbox = block.bbox
                rect = fitz.Rect(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])

                # Redact original text
                page.add_redact_annot(rect)
                page.apply_redactions()

                # Insert new text with formatting
                try:
                    # Calculate text position
                    insert_point = fitz.Point(
                        bbox[0],
                        bbox[1] + bbox[3] - 5  # Slightly above bottom
                    )

                    # Insert text with original formatting
                    page.insert_text(
                        insert_point,
                        current_text,
                        fontsize=block.font_size,
                        fontname=block.font_family,
                        color=self.hex_to_rgb(block.color),

                    )

                except Exception as e:
                    print(f"Text insertion error: {e}")
                    # Fallback - simple text insertion
                    page.insert_text(
                        insert_point,
                        current_text,
                        fontsize=block.font_size,

                    )

    def hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert hex color to RGB tuple"""
        try:
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
        except:
            return (0, 0, 0)  # Default black

    # ==============================================================
    # NAVIGATION
    # ==============================================================
    def next_page(self):
        """Go to next page"""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.page_input.setValue(self.current_page + 1)
            self.load_current_page()

    def prev_page(self):
        """Go to previous page"""
        if self.current_page > 0:
            self.current_page -= 1
            self.page_input.setValue(self.current_page + 1)
            self.load_current_page()

    def goto_page(self, page_num: int):
        """Go to specific page"""
        if 1 <= page_num <= self.total_pages:
            self.current_page = page_num - 1
            self.load_current_page()

    def on_page_selected(self, current, previous):
        """Handle page selection from list"""
        if current:
            page_num = self.pages_list.row(current) + 1
            if page_num != self.current_page + 1:
                self.page_input.setValue(page_num)

    # ==============================================================
    # ZOOM CONTROLS
    # ==============================================================
    def zoom_in(self):
        """Zoom in"""
        self.graphics_view.scale(1.25, 1.25)
        self.zoom_level *= 1.25
        self.update_zoom_combo()

    def zoom_out(self):
        """Zoom out"""
        self.graphics_view.scale(0.8, 0.8)
        self.zoom_level *= 0.8
        self.update_zoom_combo()

    def set_zoom_level(self, zoom_text: str):
        """Set specific zoom level"""
        try:
            zoom_percent = int(zoom_text.rstrip('%'))
            target_zoom = zoom_percent / 100.0

            # Reset transform and apply new zoom
            self.graphics_view.resetTransform()
            self.graphics_view.scale(target_zoom, target_zoom)
            self.zoom_level = target_zoom

        except ValueError:
            pass

    def update_zoom_combo(self):
        """Update zoom combo box"""
        zoom_percent = int(self.zoom_level * 100)
        zoom_text = f"{zoom_percent}%"

        index = self.zoom_combo.findText(zoom_text)
        if index >= 0:
            self.zoom_combo.setCurrentIndex(index)
        else:
            self.zoom_combo.setEditText(zoom_text)

    def fit_width(self):
        """Fit page width"""
        if self.scene.items():
            rect = self.scene.itemsBoundingRect()
            self.graphics_view.fitInView(rect, Qt.KeepAspectRatioByExpanding)

    def fit_page(self):
        """Fit entire page"""
        if self.scene.items():
            rect = self.scene.itemsBoundingRect()
            self.graphics_view.fitInView(rect, Qt.KeepAspectRatio)

    # ==============================================================
    # TEXT OPERATIONS
    # ==============================================================
    def add_text_block(self):
        """إضافة كتلة نص جديدة مع تسجيل التغيير"""
        if not self.scene.items():
            QMessageBox.warning(self, "تحذير", "يرجى فتح ملف PDF أولاً")
            return

        # الحصول على مركز العرض
        center = self.graphics_view.mapToScene(self.graphics_view.rect().center())

        # إنشاء كتلة نص جديدة
        new_block_id = f"new_{len(self.text_blocks)}_{int(time.time())}"
        new_block = TextBlock(
            text="نص جديد",
            bbox=[center.x() - 50, center.y() - 10, 100, 20],
            font_family="Arial",
            font_size=12,
            font_weight="normal",
            font_style="normal",
            color="#000000",
            alignment="left",
            line_height=1.2,
            char_spacing=0.0,
            word_spacing=0.0,
            original_bbox=[center.x() - 50, center.y() - 10, 100, 20],
            page_number=self.current_page,
            block_id=new_block_id,
            rotation=0.0,
            opacity=1.0
        )

        # إضافة إلى المشهد
        text_item = AdvancedTextItem(new_block)
        self.scene.addItem(text_item)
        self.text_blocks.append(new_block)

        # تسجيل التغيير
        self.change_tracker.add_change(
            self.current_page,
            new_block_id,
            'add',
            {
                'text': new_block.text,
                'bbox': new_block.bbox.copy(),
                'font_size': new_block.font_size,
                'color': new_block.color,
                'font_family': new_block.font_family
            }
        )

        # تحديد وتحرير
        text_item.setSelected(True)
        text_item.setTextInteractionFlags(Qt.TextEditorInteraction)
        text_item.setFocus()

        print(f"تم إضافة كتلة نص جديدة: {new_block_id}")

    def delete_selected(self):
        """Delete selected items"""
        selected_items = self.scene.selectedItems()

        if not selected_items:
            return

        for item in selected_items:
            if isinstance(item, AdvancedTextItem):
                # Remove from text_blocks list
                self.text_blocks = [b for b in self.text_blocks
                                 if b.block_id != item.text_block.block_id]

            self.scene.removeItem(item)

        self.save_state()

    def duplicate_selected(self):
        """Duplicate selected items"""
        selected_items = self.scene.selectedItems()

        for item in selected_items:
            if isinstance(item, AdvancedTextItem):
                item.duplicate_item()

        self.save_state()

    def select_all_text(self):
        """Select all text items"""
        for item in self.scene.items():
            if isinstance(item, AdvancedTextItem):
                item.setSelected(True)

    # ==============================================================
    # EDIT MODE TOGGLE
    # ==============================================================
    def toggle_edit_mode(self, enabled):
        """Toggle edit mode for text items"""
        self.edit_mode = enabled

        if enabled:
            self.edit_mode_btn.setStyleSheet("background-color: #4CAF50; color: white;")
            self.status_label.setText("وضع التحرير: مفعل - انقر على النص للتعديل")

            # جعل جميع النصوص مرئية عند تفعيل وضع التحرير
            for item in self.scene.items():
                if isinstance(item, AdvancedTextItem):
                    item.setOpacity(1.0)
        else:
            self.edit_mode_btn.setStyleSheet("")
            self.status_label.setText("جاهز")

            # إرجاع الشفافية عند إيقاف وضع التحرير
            for item in self.scene.items():
                if isinstance(item, AdvancedTextItem) and not item.isSelected():
                    item.setOpacity(0.0)

        # Set cursor
        if enabled:
            self.graphics_view.setCursor(Qt.IBeamCursor)
        else:
            self.graphics_view.setCursor(Qt.ArrowCursor)

        # If disabling, end any ongoing edit by setting focus to the view
        if not enabled:
            self.graphics_view.setFocus()

    # ==============================================================
    # SEARCH AND REPLACE
    # ==============================================================
    def find_text(self):
        """بحث مع تمييز النتائج"""
        search_term = self.search_input.text().strip()

        if not search_term:
            QMessageBox.warning(self, "تحذير", "يرجى إدخال نص للبحث")
            return

        # مسح التحديدات السابقة
        for item in self.scene.items():
            if isinstance(item, AdvancedTextItem):
                item.setSelected(False)

        # البحث في جميع العناصر
        self.found_items = []
        for item in self.scene.items():
            if isinstance(item, AdvancedTextItem):
                if search_term.lower() in item.toPlainText().lower():
                    self.found_items.append(item)

        if not self.found_items:
            QMessageBox.information(self, "البحث", "لم يتم العثور على النص")
            return

        # تمييز أول نتيجة
        self.current_found_index = 0
        self.highlight_found_item()

    def highlight_found_item(self):
        """تمييز العنصر الحالي في نتائج البحث"""
        if not self.found_items:
            return

        item = self.found_items[self.current_found_index]
        item.setSelected(True)
        self.graphics_view.centerOn(item)
        self.status_label.setText(f"النتيجة {self.current_found_index + 1} من {len(self.found_items)}")

    def next_search_result(self):
        """الانتقال إلى نتيجة البحث التالية"""
        if not hasattr(self, 'found_items') or not self.found_items:
            self.find_text()
            return

        self.current_found_index = (self.current_found_index + 1) % len(self.found_items)
        self.highlight_found_item()

    def prev_search_result(self):
        """الانتقال إلى نتيجة البحث السابقة"""
        if not hasattr(self, 'found_items') or not self.found_items:
            self.find_text()
            return

        self.current_found_index = (self.current_found_index - 1) % len(self.found_items)
        self.highlight_found_item()

    def replace_text(self, replace_all=False):
        """استبدال النص مع خيار الاستبدال الجزئي"""
        search_term = self.search_input.text().strip()
        replace_term = self.replace_input.text()

        if not search_term:
            QMessageBox.warning(self, "تحذير", "يرجى إدخال نص للبحث")
            return

        replaced_count = 0

        if replace_all:
            # استبدال الكل
            for item in self.scene.items():
                if isinstance(item, AdvancedTextItem):
                    if search_term in item.toPlainText():
                        old_text = item.toPlainText()
                        new_text = old_text.replace(search_term, replace_term)
                        item.setPlainText(new_text)
                        item.text_block.text = new_text

                        # تسجيل التغيير
                        self.change_tracker.add_change(
                            item.text_block.page_number,
                            item.text_block.block_id,
                            'edit',
                            {
                                'original_text': old_text,
                                'new_text': new_text,
                                'bbox': item.text_block.original_bbox.copy() if item.text_block.original_bbox else item.text_block.bbox.copy(),
                                'font_size': item.text_block.font_size,
                                'font_family': item.text_block.font_family,
                                'color': item.text_block.color
                            }
                        )
                        replaced_count += 1
        else:
            # استبدال أول ظهور فقط
            if hasattr(self, 'found_items') and self.found_items:
                item = self.found_items[self.current_found_index]
                old_text = item.toPlainText()
                new_text = old_text.replace(search_term, replace_term, 1)
                item.setPlainText(new_text)
                item.text_block.text = new_text

                # تسجيل التغيير
                self.change_tracker.add_change(
                    item.text_block.page_number,
                    item.text_block.block_id,
                    'edit',
                    {
                        'original_text': old_text,
                        'new_text': new_text,
                        'bbox': item.text_block.original_bbox.copy() if item.text_block.original_bbox else item.text_block.bbox.copy(),
                        'font_size': item.text_block.font_size,
                        'font_family': item.text_block.font_family,
                        'color': item.text_block.color
                    }
                )
                replaced_count = 1

                # الانتقال إلى النتيجة التالية
                self.current_found_index = (self.current_found_index + 1) % len(self.found_items)
                self.highlight_found_item()

        if replaced_count > 0:
            self.status_label.setText(f"تم استبدال {replaced_count} حالة")
            self.save_state()
        else:
            QMessageBox.information(self, "الاستبدال", "لم يتم العثور على نص للاستبدال")
    # ==============================================================
    # AI OPERATIONS
    # ==============================================================
    def show_ai_config(self):
        """Show AI configuration dialog"""
        dialog = AIConfigDialog(self)
        dialog.exec_()

    def enhance_selected_text(self):
        """Enhance selected text with AI"""
        selected_items = [item for item in self.scene.selectedItems()
                         if isinstance(item, AdvancedTextItem)]

        if not selected_items:
            QMessageBox.warning(self, "تحذير", "يرجى تحديد نص أولاً")
            return

        for item in selected_items:
            item.enhance_with_ai()

    def translate_selected_text(self):
        """Translate selected text with AI"""
        selected_items = [item for item in self.scene.selectedItems()
                         if isinstance(item, AdvancedTextItem)]

        if not selected_items:
            QMessageBox.warning(self, "تحذير", "يرجى تحديد نص أولاً")
            return

        for item in selected_items:
            item.translate_with_ai()

    def enhance_all_text(self):
        """Enhance all text on current page"""
        text_items = [item for item in self.scene.items()
                     if isinstance(item, AdvancedTextItem)]

        if not text_items:
            QMessageBox.warning(self, "تحذير", "لا يوجد نص للتحسين")
            return

        progress = QProgressDialog("جاري تحسين جميع النصوص...", "إلغاء", 0, len(text_items), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        def enhance_thread():
            for i, item in enumerate(text_items):
                if progress.wasCanceled():
                    break

                QTimer.singleShot(0, lambda: progress.setValue(i))

                try:
                    enhanced_text = self.ai_processor.enhance_ocr_text(item.toPlainText())
                    if enhanced_text and enhanced_text.strip():
                        QTimer.singleShot(0, lambda txt=enhanced_text, itm=item: self.update_item_text(itm, txt))
                except Exception as e:
                    print(f"Enhancement failed for item: {e}")

            QTimer.singleShot(0, progress.close)

        threading.Thread(target=enhance_thread, daemon=True).start()

    def update_item_text(self, item: AdvancedTextItem, new_text: str):
        """Update text item safely in main thread"""
        item.setPlainText(new_text)
        item.text_block.text = new_text

    def translate_current_page(self):
        """Translate entire current page"""
        languages = ["الإنجليزية", "الفرنسية", "الألمانية", "الإسبانية"]
        language, ok = QInputDialog.getItem(
            self, "ترجمة الصفحة", "ترجمة إلى:", languages, 0, False
        )

        if not ok:
            return

        text_items = [item for item in self.scene.items()
                     if isinstance(item, AdvancedTextItem)]

        if not text_items:
            QMessageBox.warning(self, "تحذير", "لا يوجد نص للترجمة")
            return

        progress = QProgressDialog("جاري ترجمة الصفحة...", "إلغاء", 0, len(text_items), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        def translate_thread():
            for i, item in enumerate(text_items):
                if progress.wasCanceled():
                    break

                QTimer.singleShot(0, lambda: progress.setValue(i))

                try:
                    translated_text = self.ai_processor.translate_text(item.toPlainText(), language)
                    if translated_text and translated_text.strip():
                        QTimer.singleShot(0, lambda txt=translated_text, itm=item: self.update_item_text(itm, txt))
                except Exception as e:
                    print(f"Translation failed for item: {e}")

            QTimer.singleShot(0, progress.close)

        threading.Thread(target=translate_thread, daemon=True).start()

    def summarize_page(self):
        """Summarize current page content"""
        all_text = " ".join([block.text for block in self.text_blocks])

        if not all_text.strip():
            QMessageBox.warning(self, "تحذير", "لا يوجد نص للتلخيص")
            return

        progress = QProgressDialog("جاري تلخيص الصفحة...", "إلغاء", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        def summarize_thread():
            try:
                prompt = f"لخص النص التالي في فقرة واحدة:\n\n{all_text}"
                summary = self.ai_processor._call_ai_api(prompt)

                QTimer.singleShot(0, lambda: self.show_summary(summary, progress))
            except Exception as e:
                QTimer.singleShot(0, lambda: self.show_summary_error(str(e), progress))

        threading.Thread(target=summarize_thread, daemon=True).start()

    def show_summary(self, summary: str, progress_dialog):
        """Show page summary"""
        progress_dialog.close()

        dialog = QDialog(self)
        dialog.setWindowTitle("ملخص الصفحة")
        dialog.resize(500, 300)

        layout = QVBoxLayout()

        summary_text = QTextEdit()
        summary_text.setPlainText(summary)
        summary_text.setReadOnly(True)

        layout.addWidget(QLabel("ملخص محتوى الصفحة:"))
        layout.addWidget(summary_text)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.setLayout(layout)
        dialog.exec_()

    def show_summary_error(self, error_msg: str, progress_dialog):
        """Show summary error"""
        progress_dialog.close()
        QMessageBox.critical(self, "خطأ في التلخيص", f"فشل في تلخيص الصفحة:\n{error_msg}")

    # ==============================================================
    # TEXT EDITOR OPERATIONS
    # ==============================================================
    def apply_text_changes(self):
        """Apply changes from text editor to scene"""
        editor_text = self.text_editor.toPlainText()
        lines = editor_text.split('\n')

        # Match lines to text blocks (simplified approach)
        text_items = [item for item in self.scene.items()
                     if isinstance(item, AdvancedTextItem)]

        for i, item in enumerate(text_items):
            if i < len(lines):
                new_text = lines[i].strip()
                if new_text != item.toPlainText():
                    old_text = item.toPlainText()
                    item.setPlainText(new_text)
                    item.text_block.text = new_text

                    # تسجيل التغيير
                    self.change_tracker.add_change(
                        item.text_block.page_number,
                        item.text_block.block_id,
                        'edit',
                        {
                            'original_text': old_text,
                            'new_text': new_text,
                            'bbox': item.text_block.original_bbox.copy() if item.text_block.original_bbox else item.text_block.bbox.copy(),
                            'font_size': item.text_block.font_size,
                            'font_family': item.text_block.font_family,
                            'color': item.text_block.color
                        }
                    )

        self.save_state()
        self.status_label.setText("تم تطبيق التغييرات")

    def reset_text_editor(self):
        """Reset text editor to original content"""
        all_text = "\n".join([block.text for block in self.text_blocks])
        self.text_editor.setText(all_text)

    def reprocess_current_page(self):
        """Reprocess current page with new settings"""
        if self.doc:
            self.load_current_page()

    # ==============================================================
    # FORMAT CONTROLS
    # ==============================================================
    def choose_text_color(self):
        """Choose text color for selected items"""
        color = QColorDialog.getColor(Qt.black, self)

        if color.isValid():
            # Update button color
            self.color_btn.setStyleSheet(f"background-color: {color.name()}")

            # Apply to selected items
            for item in self.scene.selectedItems():
                if isinstance(item, AdvancedTextItem):
                    old_color = item.text_block.color
                    item.setDefaultTextColor(color)
                    item.text_block.color = color.name()

                    # تسجيل التغيير
                    self.change_tracker.add_change(
                        item.text_block.page_number,
                        item.text_block.block_id,
                        'format',
                        {
                            'old_color': old_color,
                            'new_color': color.name(),
                            'bbox': item.text_block.original_bbox.copy() if item.text_block.original_bbox else item.text_block.bbox.copy()
                        }
                    )

    # ==============================================================
    # HISTORY (UNDO/REDO)
    # ==============================================================
    def save_state(self):
        """Save current state for undo"""
        # Simplified state saving - just save block positions and text
        state = []

        for item in self.scene.items():
            if isinstance(item, AdvancedTextItem):
                state.append({
                    'block_id': item.text_block.block_id,
                    'text': item.toPlainText(),
                    'pos': (item.pos().x(), item.pos().y()),
                    'color': item.text_block.color,
                    'font_family': item.text_block.font_family,
                    'font_size': item.text_block.font_size
                })

        # Add to history
        if self.history_index < len(self.history_stack) - 1:
            # Remove future states
            self.history_stack = self.history_stack[:self.history_index + 1]

        self.history_stack.append(state)
        self.history_index += 1

        # Limit history size
        if len(self.history_stack) > 50:
            self.history_stack.pop(0)
            self.history_index -= 1

    def undo(self):
        """Undo last action"""
        if self.history_index > 0:
            self.history_index -= 1
            self.restore_state(self.history_stack[self.history_index])

    def redo(self):
        """Redo last undone action"""
        if self.history_index < len(self.history_stack) - 1:
            self.history_index += 1
            self.restore_state(self.history_stack[self.history_index])

    def restore_state(self, state):
        """Restore application state"""
        # Clear current items
        for item in list(self.scene.items()):
            if isinstance(item, AdvancedTextItem):
                self.scene.removeItem(item)

        # Restore items from state
        for item_state in state:
            # Find corresponding text block
            block = None
            for b in self.text_blocks:
                if b.block_id == item_state['block_id']:
                    block = b
                    break

            if block:
                # Update block with saved state
                block.text = item_state['text']
                block.bbox[0] = item_state['pos'][0]
                block.bbox[1] = item_state['pos'][1]
                block.color = item_state['color']
                block.font_family = item_state['font_family']
                block.font_size = item_state['font_size']

                # Create and add item
                text_item = AdvancedTextItem(block)
                self.scene.addItem(text_item)

    def pil_to_qimage(self, pil_image: Image.Image) -> QImage:
        """Convert PIL image to QImage"""
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        np_array = np.array(pil_image)
        height, width, channel = np_array.shape
        bytes_per_line = 3 * width

        return QImage(np_array.data, width, height, bytes_per_line, QImage.Format_RGB888)

# ==============================================================
# ENTRY POINT
# ==============================================================
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PDF Editor Pro")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Advanced PDF Solutions")
    QApplication.setQuitOnLastWindowClosed(True)

    window = AdvancedPDFEditor()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

