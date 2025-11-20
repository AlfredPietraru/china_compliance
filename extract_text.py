from pypdf import PdfReader
from china_compliance_utils import extract_raw_text_from_pdf
path = "documents/regulations_chinese.pdf"
data = extract_raw_text_from_pdf(path)
with open("out.txt", "w") as f:
    f.write(data)
