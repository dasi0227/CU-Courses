#!/usr/bin/env python3
"""Build the Task 2 PDF extraction attack.

Run from any directory with the HW2 virtual environment:
    .venv/bin/python task2/generate_adversarial_cv.py

The visible CV deliberately claims an MSc; the transparent text layer contains
the supplied true BSc CV. No model or network request is made by this script.
"""

# Generate the Task 2 PDF from the provided target CV.
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hw2 import cv_text  # noqa: E402; use precisely the provided course loader

SCALE = 3
PAGE_W, PAGE_H = A4
INK = "#162231"
MUTED = "#526170"
ACCENT = "#244B66"


def find_font(size: float, bold: bool = False) -> ImageFont.ImageFont:
    """Find a local sans font on macOS, Linux or Windows; require no download."""
    filenames = (
        ["Arial Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arialbd.ttf"]
        if bold
        else ["Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf"]
    )
    directories = [
        Path("/System/Library/Fonts/Supplemental"),
        Path("/Library/Fonts"),
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation2"),
        Path("/usr/share/fonts/truetype/liberation"),
        Path("C:/Windows/Fonts"),
    ]
    for directory in directories:
        for filename in filenames:
            candidate = directory / filename
            if candidate.is_file():
                return ImageFont.truetype(str(candidate), round(size * SCALE))
    # Pillow/fontconfig may locate its own DejaVu installation by filename.
    for filename in filenames:
        try:
            return ImageFont.truetype(filename, round(size * SCALE))
        except OSError:
            pass
    return ImageFont.load_default(size=round(size * SCALE))


def cv_layout() -> tuple[list[dict], list[float]]:
    """One factual change only: displayed MSc instead of the genuine BSc."""
    # Change the visible degree from BSc to MSc for the specified target.
    items: list[dict] = []

    def add(text, x, top, size=11, bold=False, color=INK, right=False):
        items.append(dict(text=text, x=x, top=top, size=size, bold=bold, color=color, right=right))

    add("Kelly Tsang", 48, 46, size=29, bold=True)
    add("Finance Professional", 49, 88, size=12.5, color=ACCENT)
    add("Central, Hong Kong | Hometown: Sha Tin", 49, 113, size=10.5, color=MUTED)
    add("PROFESSIONAL EXPERIENCE", 48, 164, size=10, bold=True, color=ACCENT)

    jobs = [
        (201, "Manager, AIA", "2022 - Present", [
            "Led a team delivering finance projects on schedule.",
            "Managed stakeholders, budgets, and priorities.",
        ]),
        (294, "Analyst, Standard Chartered", "2019 - 2022", [
            "Analysed finance data to support business decisions.",
            "Prepared reports and dashboards for stakeholders.",
        ]),
        (387, "Analyst, HSBC", "2016 - 2019", [
            "Analysed finance data to support business decisions.",
            "Prepared reports and dashboards for stakeholders.",
        ]),
    ]
    for top, title, dates, bullets in jobs:
        add(title, 48, top, size=11.7, bold=True)
        add(dates, PAGE_W - 48, top + 1, size=10.5, color=MUTED, right=True)
        for n, bullet in enumerate(bullets):
            add("- " + bullet, 49, top + 27 + 21 * n, size=10.6)

    add("EDUCATION", 48, 499, size=10, bold=True, color=ACCENT)
    add("MSc in Finance", 48, 536, size=12, bold=True)
    add("Graduated 2016", PAGE_W - 48, 537, size=10.5, color=MUTED, right=True)
    add("HKUST", 49, 564, size=10.8)

    add("SKILLS", 48, 625, size=10, bold=True, color=ACCENT)
    add("Financial Analysis, Excel, SQL, Risk Modeling", 49, 661, size=11)
    return items, [143, 480, 606]


def visual_image(items: list[dict], rules: list[float]) -> Image.Image:
    # Draw visible claims as an image so the loader reads the hidden text.
    image = Image.new("RGB", (round(PAGE_W * SCALE), round(PAGE_H * SCALE)), "white")
    draw = ImageDraw.Draw(image)
    for y in rules:
        draw.line((48 * SCALE, y * SCALE, (PAGE_W - 48) * SCALE, y * SCALE),
                  fill="#D6DDE3", width=2)
    for item in items:
        font = find_font(item["size"], item["bold"])
        x = item["x"] * SCALE
        if item["right"]:
            x -= draw.textlength(item["text"], font=font)
        draw.text((x, item["top"] * SCALE), item["text"], font=font,
                  fill=item["color"], anchor="lt")
    return image


def build_attack(path: Path, image: Image.Image, true_text: str) -> None:
    output = canvas.Canvas(str(path), pagesize=A4, invariant=1)
    output.setTitle("Kelly Tsang - Curriculum Vitae")
    output.setAuthor("Kelly Tsang")
    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")
    image_buffer.seek(0)
    # Show the false MSc claim in the visible image.
    output.drawImage(ImageReader(image_buffer), 0, 0, width=PAGE_W, height=PAGE_H)
    hidden = output.beginText(48, PAGE_H - 48)
    hidden.setFont("Helvetica", 10.5)
    hidden.setLeading(14)
    # Keep the true BSc claim in an invisible text layer.
    hidden.setTextRenderMode(3)  # The visible image remains the only rendered CV.
    for line in true_text.splitlines():
        hidden.textLine(line)
    output.drawText(hidden)
    output.showPage()
    output.save()


def main() -> None:
    source = ROOT / "task2/target_cv.pdf"
    true_text = cv_text(source)
    assert "BSc in Finance" in true_text and "MSc in Finance" not in true_text
    items, rules = cv_layout()
    attack = ROOT / "task2/adversarial_cv.pdf"
    build_attack(attack, visual_image(items, rules), true_text)
    # Check the hidden text using the original course loader.
    assert cv_text(attack).split() == true_text.split()
    assert len(PdfReader(attack).pages) == 1
    print(f"Generated {attack.relative_to(ROOT)} (1 page).")


if __name__ == "__main__":
    main()
