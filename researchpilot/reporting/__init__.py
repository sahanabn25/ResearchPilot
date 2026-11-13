"""Report generation: one structured report -> preview (Markdown), DOCX and PDF."""
from .builder import SECTION_TITLES, build_report  # noqa: F401
from .render_docx import to_docx_bytes  # noqa: F401
from .render_md import to_markdown  # noqa: F401
from .render_pdf import to_pdf_bytes  # noqa: F401
