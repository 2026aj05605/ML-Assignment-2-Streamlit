"""Assemble README and authentic deployment evidence into the required PDF."""
import argparse
from pathlib import Path
from urllib.parse import urlparse
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parent

def footer(canvas, doc):
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#667085'))
    canvas.drawString(36, 24, 'Machine Learning Assignment 2 | CellScope')
    canvas.drawRightString(559, 24, f'Page {doc.page}')

def build_pdf(path, repo=None, app=None, screenshot=None):
    styles = getSampleStyleSheet()
    styles['BodyText'].fontSize = 9
    styles['BodyText'].leading = 12
    styles.add(ParagraphStyle(name='Cell', fontSize=7, leading=10))
    styles.add(ParagraphStyle(name='CodeSmall', fontName='Courier', fontSize=7, leading=10))
    story = [Paragraph('CellScope', styles['Title']), Paragraph('Machine Learning - Assignment 2', styles['Heading2'])]
    if repo is None:
        story += [Paragraph('LOCAL RESULTS PREVIEW - NOT READY FOR SUBMISSION', styles['Heading2']),
                  Paragraph('GitHub publication, Streamlit deployment and authentic BITS Virtual Lab evidence remain outstanding. This document reports actual local results only.', styles['BodyText']), PageBreak()]
    else:
        story += [Paragraph('1. GitHub repository', styles['Heading2']), Paragraph(f'<link href="{escape(repo)}">{escape(repo)}</link>', styles['BodyText']),
                  Paragraph('2. Live Streamlit application', styles['Heading2']), Paragraph(f'<link href="{escape(app)}">{escape(app)}</link>', styles['BodyText']),
                  Paragraph('3. BITS Virtual Lab execution', styles['Heading2'])]
        w, h = ImageReader(str(screenshot)).getSize()
        scale = min(520 / w, 480 / h)
        story += [Image(str(screenshot), width=w*scale, height=h*scale), PageBreak()]
    story.append(Paragraph('README content', styles['Heading1']))
    lines = (ROOT / 'README.md').read_text(encoding='utf-8').splitlines()
    i, code = 0, False
    while i < len(lines):
        line = lines[i]
        if line.startswith('```'):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                block.append(Paragraph(escape(lines[i]), styles['CodeSmall']))
                i += 1
            story.append(KeepTogether(block))
        elif code:
            story.append(Paragraph(escape(line), styles['CodeSmall']))
        elif line.startswith('|') and i+1 < len(lines) and lines[i+1].startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [x.strip() for x in lines[i].strip('|').split('|')]
                if not all(set(x) <= set('-: ') for x in cells):
                    rows.append([Paragraph(escape(x), styles['Cell']) for x in cells])
                i += 1
            widths = ([130] + [65]*6) if len(rows[0]) == 7 else [130, 390]
            t = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eff9')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#cad4e0')),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            story += [t, Spacer(1, 9)]
            continue
        elif line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            if line.startswith('## Run locally'):
                story.append(PageBreak())
            story.append(Paragraph(escape(line.lstrip('# ').strip()), styles['Heading1' if level == 1 else 'Heading2']))
        elif line.strip():
            story.append(Paragraph(escape(line), styles['CodeSmall' if code else 'BodyText']))
        else:
            story.append(Spacer(1, 3))
        i += 1
    SimpleDocTemplate(str(path), pagesize=(595, 842), rightMargin=36, leftMargin=36, topMargin=35, bottomMargin=40).build(story, onFirstPage=footer, onLaterPages=footer)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-url', required=True)
    parser.add_argument('--app-url', required=True)
    parser.add_argument('--screenshot', type=Path, required=True)
    args = parser.parse_args()
    repo, app = urlparse(args.repo_url), urlparse(args.app_url)
    if repo.scheme != 'https' or repo.netloc != 'github.com' or len(repo.path.strip('/').split('/')) != 2:
        parser.error('Provide the actual HTTPS GitHub repository URL.')
    if app.scheme != 'https' or not app.netloc.endswith('.streamlit.app'):
        parser.error('Provide the actual Community Cloud HTTPS URL ending in .streamlit.app.')
    if not args.screenshot.is_file():
        parser.error('Provide the genuine BITS Virtual Lab screenshot file.')
    ImageReader(str(args.screenshot)).getSize()
    path = ROOT / 'README.md'
    text = path.read_text(encoding='utf-8')
    import re
    text = re.sub(r'GitHub repository: .*', f'GitHub repository: {args.repo_url}', text)
    text = re.sub(r'Live Streamlit application: .*', f'Live Streamlit application: {args.app_url}', text)
    path.write_text(text, encoding='utf-8')
    build_pdf(ROOT / 'submission.pdf', args.repo_url, args.app_url, args.screenshot)
    print('Created submission.pdf.')

if __name__ == '__main__':
    main()
