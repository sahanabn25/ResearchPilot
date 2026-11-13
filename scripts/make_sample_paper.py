"""Write a FICTIONAL sample research paper (PDF) for trying Option 2 without a real paper.

Usage: python scripts/make_sample_paper.py        -> data/sample_paper_FICTIONAL.pdf
The content is invented for testing (a hybrid CNN-LSTM weather-forecasting 'study'); none of it is a real result.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from pdf_fixtures import make_paper_pdf  # noqa: E402

if __name__ == "__main__":
    out = os.path.join(ROOT, "data", "sample_paper_FICTIONAL.pdf")
    with open(out, "wb") as fh:
        fh.write(make_paper_pdf())
    print("Wrote", out, "(fictional content, for testing only)")
