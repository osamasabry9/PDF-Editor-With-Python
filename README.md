# Advanced PDF Editor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

An advanced open-source PDF editor leveraging AI and OCR technologies for efficient PDF editing. Fully supports Arabic language and provides an Adobe-like user interface.

## Key Features

- **Direct text editing** in PDF files
- **Full Arabic language support** with mixed text processing (Arabic/English)
- **Advanced OCR technology** for text extraction from scanned PDFs
- **AI integration** for text enhancement, translation, and summarization
- **Intuitive Adobe-like user interface**
- **Save modifications** while preserving original file quality
- **Dark theme support** for comfortable user experience

## Requirements

- Python 3.8 or higher
- Windows, Linux, or macOS
- Sufficient storage space for temporary files

## Installation

1. Clone the repository:
```bash
git clone https://github.com/osamasabry9/PDF-Editor-With-Python.git
cd advanced-pdf-editor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate    # Windows
```

3. Install required libraries:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python newt.py
```

## How to Use

1. **Open PDF file**: Use the "Open" button to load a PDF
2. **Process page**: Texts will be automatically extracted from current page
3. **Edit texts**:
   - Double-click on any text to edit it directly
   - Use context menu to change font, color, or alignment
4. **Use AI tools**:
   - Select text and click "Enhance Text" to improve it using AI
   - Use "Translate" to translate selected text
   - Use "Summarize Page" to get content summary
5. **Save modifications**: Use "Save" or "Save As" to save modified PDF

## Setting Up AI API Keys

To use AI features:

1. Go to menu: `AI > AI Settings`
2. Choose a provider (OpenAI, Claude, or Gemini)
3. Enter your API key
4. Click "Test Connection" to verify the key


## File Structure

```
advanced-pdf-editor/
├── newt.py              # Main application file
├── requirements.txt     # Installation requirements
├── README.md            # This file
└── LICENSE              # Project license
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes and add them (`git add .`)
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support and Contact

For any inquiries or issues, please open an Issue in the GitHub repository.
