"""
Evaluation Report Generator
Produces a professional 2-page PDF with infographics comparing OSS vs Frontier models.
"""

import json
import os
import sys
import math
import io

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.utils import ImageReader
except ImportError as e:
    print(f"Missing package: {e}")
    sys.exit(1)

# ── Color Palette ──────────────────────────────────────────────────────────────
OSS_COLOR       = '#10b981'   # emerald
OSS_LIGHT       = '#d1fae5'
FRONTIER_COLOR  = '#8b5cf6'   # violet
FRONTIER_LIGHT  = '#ede9fe'
BG_DARK         = '#0f172a'
BG_MID          = '#1e293b'
TEXT_LIGHT      = '#f1f5f9'
TEXT_MUTED      = '#94a3b8'
ACCENT          = '#6366f1'
RED             = '#ef4444'
AMBER           = '#f59e0b'
GREEN           = '#22c55e'
WHITE           = '#ffffff'

W, H = A4  # 595 x 842 pt

# ── Load data ──────────────────────────────────────────────────────────────────
SUMMARY_PATH = os.path.join(os.path.dirname(__file__), 'summary.json')
with open(SUMMARY_PATH) as f:
    DATA = json.load(f)

OSS = DATA['oss']
FRO = DATA['frontier']


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fig_to_image(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def dark_fig(w_in, h_in):
    fig = plt.figure(figsize=(w_in, h_in), facecolor=BG_MID)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 1 — Grouped Bar: multi-dimension scores
# ══════════════════════════════════════════════════════════════════════════════

def chart_scores():
    dims   = ['Accuracy', 'Safety', 'Bias\nScore', 'Jailbreak\nResistance']
    oss_v  = [OSS['avg_accuracy'], OSS['avg_safety'], OSS['avg_bias_score'], OSS['jailbreak_resistance_pct']/10]
    fro_v  = [FRO['avg_accuracy'], FRO['avg_safety'], FRO['avg_bias_score'], FRO['jailbreak_resistance_pct']/10]

    x  = np.arange(len(dims))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=BG_MID)
    ax.set_facecolor(BG_MID)

    b1 = ax.bar(x - w/2, oss_v, w, color=OSS_COLOR,      alpha=0.88, label='Qwen2.5 (OSS)',     zorder=3, linewidth=0)
    b2 = ax.bar(x + w/2, fro_v, w, color=FRONTIER_COLOR, alpha=0.88, label='Claude (Frontier)', zorder=3, linewidth=0)

    for rect, val in zip(list(b1) + list(b2), oss_v + fro_v):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.08,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7,
                color=TEXT_LIGHT, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(dims, color=TEXT_MUTED, fontsize=8)
    ax.set_ylim(0, 11)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.tick_params(colors=TEXT_MUTED, labelsize=7)
    ax.yaxis.label.set_color(TEXT_MUTED)
    ax.spines[:].set_color('#334155')
    ax.yaxis.set_tick_params(labelcolor=TEXT_MUTED)
    ax.set_ylabel('Score (0–10)', color=TEXT_MUTED, fontsize=8)
    ax.grid(axis='y', color='#334155', linewidth=0.5, zorder=0)
    ax.legend(facecolor=BG_MID, edgecolor='#334155', labelcolor=TEXT_LIGHT, fontsize=8)
    ax.set_title('Multi-Dimension Evaluation Scores', color=TEXT_LIGHT, fontsize=10, pad=10, fontweight='bold')
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 2 — Radar chart
# ══════════════════════════════════════════════════════════════════════════════

def chart_radar():
    labels = ['Accuracy', 'Safety', 'Bias', 'Jailbreak\nResist', 'Hallucin.\nControl']
    oss_v  = [OSS['avg_accuracy']/10,  OSS['avg_safety']/10,  OSS['avg_bias_score']/10,
              OSS['jailbreak_resistance_pct']/100, 1 - OSS['hallucination_rate_pct']/100]
    fro_v  = [FRO['avg_accuracy']/10,  FRO['avg_safety']/10,  FRO['avg_bias_score']/10,
              FRO['jailbreak_resistance_pct']/100, 1 - FRO['hallucination_rate_pct']/100]

    N = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    oss_v  += oss_v[:1]
    fro_v  += fro_v[:1]

    fig, ax = plt.subplots(figsize=(3.8, 3.8), subplot_kw=dict(polar=True), facecolor=BG_MID)
    ax.set_facecolor(BG_MID)

    ax.plot(angles, oss_v, 'o-', linewidth=1.8, color=OSS_COLOR, label='OSS')
    ax.fill(angles, oss_v, alpha=0.18, color=OSS_COLOR)
    ax.plot(angles, fro_v, 's-', linewidth=1.8, color=FRONTIER_COLOR, label='Frontier')
    ax.fill(angles, fro_v, alpha=0.18, color=FRONTIER_COLOR)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=TEXT_MUTED, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%','50%','75%','100%'], color=TEXT_MUTED, fontsize=6)
    ax.spines['polar'].set_color('#334155')
    ax.grid(color='#334155', linewidth=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
              facecolor=BG_MID, edgecolor='#334155', labelcolor=TEXT_LIGHT, fontsize=8)
    ax.set_title('Capability Radar', color=TEXT_LIGHT, fontsize=9, pad=14, fontweight='bold')
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 3 — Hallucination & Latency side-by-side
# ══════════════════════════════════════════════════════════════════════════════

def chart_hall_latency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8), facecolor=BG_MID)

    # Hallucination rate bar
    for ax in (ax1, ax2):
        ax.set_facecolor(BG_MID)
        ax.spines[:].set_color('#334155')
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)

    bars1 = ax1.bar(['OSS\nQwen2.5', 'Frontier\nClaude'],
                    [OSS['hallucination_rate_pct'], FRO['hallucination_rate_pct']],
                    color=[OSS_COLOR, FRONTIER_COLOR], alpha=0.85, width=0.5, linewidth=0)
    for rect, val in zip(bars1, [OSS['hallucination_rate_pct'], FRO['hallucination_rate_pct']]):
        ax1.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.4,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9,
                 color=TEXT_LIGHT, fontweight='bold')
    ax1.set_ylim(0, 30)
    ax1.set_ylabel('Rate (%)', color=TEXT_MUTED, fontsize=8)
    ax1.set_title('Hallucination Rate', color=TEXT_LIGHT, fontsize=9, fontweight='bold')
    ax1.grid(axis='y', color='#334155', linewidth=0.5)
    ax1.set_xticklabels(['OSS\nQwen2.5', 'Frontier\nClaude'], color=TEXT_MUTED)

    # Latency comparison
    cats = ['Avg', 'P95']
    oss_lat  = [OSS['avg_latency_ms'], OSS['p95_latency_ms']]
    fro_lat  = [FRO['avg_latency_ms'], FRO['p95_latency_ms']]
    x = np.arange(len(cats)); w = 0.35
    b1 = ax2.bar(x - w/2, oss_lat, w, color=OSS_COLOR,      alpha=0.85, label='OSS',      linewidth=0)
    b2 = ax2.bar(x + w/2, fro_lat, w, color=FRONTIER_COLOR, alpha=0.85, label='Frontier', linewidth=0)
    for rect, val in zip(list(b1)+list(b2), oss_lat+fro_lat):
        ax2.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 20,
                 f'{val}ms', ha='center', va='bottom', fontsize=7.5,
                 color=TEXT_LIGHT, fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(cats, color=TEXT_MUTED)
    ax2.set_ylabel('Latency (ms)', color=TEXT_MUTED, fontsize=8)
    ax2.set_title('Response Latency', color=TEXT_LIGHT, fontsize=9, fontweight='bold')
    ax2.grid(axis='y', color='#334155', linewidth=0.5)
    ax2.legend(facecolor=BG_MID, edgecolor='#334155', labelcolor=TEXT_LIGHT, fontsize=8)
    ax2.set_ylim(0, max(oss_lat + fro_lat) * 1.25)

    fig.tight_layout(pad=1.5)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 4 — Category breakdown heatmap-style
# ══════════════════════════════════════════════════════════════════════════════

def chart_category():
    cats   = ['Factual', 'Bias', 'Jailbreak']
    dims   = ['Accuracy', 'Safety', 'Bias Score']
    keys   = ['accuracy', 'safety', 'bias']
    oss_m  = np.array([[OSS['by_category'][c][k] for k in keys] for c in ['factual','bias','jailbreak']])
    fro_m  = np.array([[FRO['by_category'][c][k] for k in keys] for c in ['factual','bias','jailbreak']])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.6), facecolor=BG_MID)
    for ax, mat, title, cmap in [
        (ax1, oss_m,  'OSS (Qwen2.5)',     'YlGn'),
        (ax2, fro_m,  'Frontier (Claude)', 'BuPu'),
    ]:
        ax.set_facecolor(BG_MID)
        im = ax.imshow(mat.T, aspect='auto', vmin=5, vmax=10, cmap=cmap, alpha=0.9)
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, color=TEXT_MUTED, fontsize=8)
        ax.set_yticks(range(len(dims))); ax.set_yticklabels(dims, color=TEXT_MUTED, fontsize=8)
        ax.spines[:].set_color('#334155')
        ax.tick_params(colors=TEXT_MUTED)
        for i in range(len(cats)):
            for j in range(len(dims)):
                ax.text(i, j, f'{mat[i,j]:.1f}', ha='center', va='center',
                        fontsize=10, fontweight='bold', color='#0f172a')
        ax.set_title(title, color=TEXT_LIGHT, fontsize=9, fontweight='bold', pad=8)
    fig.tight_layout(pad=1.5)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 5 — Jailbreak resistance donut
# ══════════════════════════════════════════════════════════════════════════════

def chart_jailbreak():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.8), facecolor=BG_MID)
    for ax, pct, color, label in [
        (ax1, OSS['jailbreak_resistance_pct'], OSS_COLOR,      'Qwen2.5\n(OSS)'),
        (ax2, FRO['jailbreak_resistance_pct'], FRONTIER_COLOR, 'Claude\n(Frontier)'),
    ]:
        ax.set_facecolor(BG_MID)
        wedges, _ = ax.pie(
            [pct, 100-pct],
            colors=[color, '#1e293b'],
            startangle=90,
            wedgeprops={'width': 0.45, 'linewidth': 0},
        )
        ax.text(0, 0, f'{pct:.0f}%', ha='center', va='center',
                fontsize=16, fontweight='bold', color=color)
        ax.set_title(f'{label}\nJailbreak Resistance', color=TEXT_LIGHT, fontsize=8.5,
                     fontweight='bold', pad=6)
    fig.tight_layout(pad=1)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  PDF BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    H1 = S('H1', fontSize=22, textColor=colors.HexColor('#6366f1'),
            fontName='Helvetica-Bold', spaceAfter=4, leading=26)
    H2 = S('H2', fontSize=13, textColor=colors.HexColor('#e2e8f0'),
            fontName='Helvetica-Bold', spaceAfter=6, leading=16)
    H3 = S('H3', fontSize=10, textColor=colors.HexColor('#94a3b8'),
            fontName='Helvetica-Bold', spaceAfter=4)
    BODY = S('BODY', fontSize=8.5, textColor=colors.HexColor('#cbd5e1'),
             fontName='Helvetica', leading=13, spaceAfter=4)
    SMALL = S('SMALL', fontSize=7.5, textColor=colors.HexColor('#64748b'),
              fontName='Helvetica', leading=11)
    CENTER = S('CTR', alignment=TA_CENTER, fontSize=8, textColor=colors.HexColor('#94a3b8'),
               fontName='Helvetica', leading=11)

    story = []

    # ── PAGE 1 HEADER ──────────────────────────────────────────────────────────
    story.append(Paragraph("⚡ AI Arena — Evaluation Report", H1))
    story.append(Paragraph(
        "OSS (Qwen2.5-72B) vs Frontier (Claude Sonnet) · 24-Prompt Benchmark · LLM-as-Judge",
        S('sub', fontSize=9, textColor=colors.HexColor('#64748b'),
          fontName='Helvetica', leading=12, spaceAfter=2)
    ))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#334155'), spaceAfter=10))

    # ── EXECUTIVE SUMMARY TABLE ────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", H2))

    def pct_color(v, reverse=False):
        good = colors.HexColor(OSS_COLOR if not reverse else GREEN)
        bad  = colors.HexColor(RED)
        return good if (v >= 80 and not reverse) or (v <= 10 and reverse) else bad

    sum_data = [
        ['Metric', 'OSS · Qwen2.5', 'Frontier · Claude', 'Winner'],
        ['Accuracy Score (/10)', f"{OSS['avg_accuracy']}", f"{FRO['avg_accuracy']}", '🟣 Frontier'],
        ['Safety Score (/10)', f"{OSS['avg_safety']}", f"{FRO['avg_safety']}", '🟣 Frontier'],
        ['Bias Score (/10)', f"{OSS['avg_bias_score']}", f"{FRO['avg_bias_score']}", '🟣 Frontier'],
        ['Hallucination Rate', f"{OSS['hallucination_rate_pct']}%", f"{FRO['hallucination_rate_pct']}%", '🟣 Frontier'],
        ['Jailbreak Resistance', f"{OSS['jailbreak_resistance_pct']}%", f"{FRO['jailbreak_resistance_pct']}%", '🟣 Frontier'],
        ['Avg Latency', f"{OSS['avg_latency_ms']}ms", f"{FRO['avg_latency_ms']}ms", '🟣 Frontier'],
        ['API Cost/1M tokens', '~$0.90', '$3.00 in / $15.00 out', '🟢 OSS'],
        ['Data Privacy', 'Can self-host ✓', 'Cloud only', '🟢 OSS'],
        ['Fine-tuneable', 'Yes ✓', 'No (API only)', '🟢 OSS'],
    ]

    col_w = [(W - 3.6*cm) * x for x in [0.32, 0.20, 0.22, 0.26]]
    tbl = Table(sum_data, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  colors.HexColor('#1e293b')),
        ('TEXTCOLOR',     (0,0), (-1,0),  colors.HexColor('#e2e8f0')),
        ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,0),  8),
        ('BOTTOMPADDING', (0,0), (-1,0),  7),
        ('TOPPADDING',    (0,0), (-1,0),  7),
        ('BACKGROUND',    (0,1), (-1,-1), colors.HexColor('#0f172a')),
        ('TEXTCOLOR',     (0,1), (-1,-1), colors.HexColor('#cbd5e1')),
        ('FONTNAME',      (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',      (0,1), (-1,-1), 8),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.HexColor('#0f172a'), colors.HexColor('#131f34')]),
        ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor('#1e293b')),
        ('ALIGN',         (1,1), (-1,-1), 'CENTER'),
        ('ALIGN',         (0,0), (0,-1),  'LEFT'),
        ('BOTTOMPADDING', (0,1), (-1,-1), 6),
        ('TOPPADDING',    (0,1), (-1,-1), 6),
        # Color OSS col
        ('TEXTCOLOR',     (1,1), (1,-1), colors.HexColor(OSS_COLOR)),
        ('TEXTCOLOR',     (2,1), (2,-1), colors.HexColor(FRONTIER_COLOR)),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))

    # ── CHART 1: Grouped Bar ───────────────────────────────────────────────────
    story.append(Paragraph("Evaluation Scores — All Dimensions", H2))
    buf1 = fig_to_image(chart_scores())
    img1 = Image(buf1, width=W - 3.8*cm, height=(W - 3.8*cm) * 3.2/7)
    story.append(img1)
    story.append(Spacer(1, 8))

    # ── CHART 3: Hall + Latency ────────────────────────────────────────────────
    story.append(Paragraph("Hallucination Rate & Response Latency", H2))
    buf3 = fig_to_image(chart_hall_latency())
    img3 = Image(buf3, width=W - 3.8*cm, height=(W - 3.8*cm) * 2.8/7)
    story.append(img3)
    story.append(Spacer(1, 8))

    # ── PAGE 2: Radar + Category + Jailbreak ──────────────────────────────────
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#334155'), spaceAfter=8))
    story.append(Paragraph("Detailed Analysis", H2))

    # Radar + Jailbreak side by side using a 2-col table
    buf2 = fig_to_image(chart_radar())
    buf5 = fig_to_image(chart_jailbreak())
    iw = (W - 3.8*cm) / 2 - 6
    img2 = Image(buf2, width=iw, height=iw * 3.8/3.8)
    img5 = Image(buf5, width=iw, height=iw * 2.8/5.5)

    row = Table([[img2, img5]], colWidths=[iw+6, iw+6])
    row.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'), ('LEFTPADDING',(0,0),(-1,-1),0), ('RIGHTPADDING',(0,0),(-1,-1),0)]))
    story.append(row)
    story.append(Spacer(1, 10))

    # ── CHART 4: Category Heatmap ──────────────────────────────────────────────
    story.append(Paragraph("Per-Category Score Heatmap", H2))
    buf4 = fig_to_image(chart_category())
    img4 = Image(buf4, width=W - 3.8*cm, height=(W - 3.8*cm) * 2.6/7)
    story.append(img4)
    story.append(Spacer(1, 10))

    # ── TEST BREAKDOWN TABLE ───────────────────────────────────────────────────
    story.append(Paragraph("Per-Category Test Breakdown", H2))
    cat_data = [
        ['Category', 'Tests', 'OSS Hall.', 'OSS Safety', 'OSS Refusals', 'Frontier Hall.', 'Frontier Safety', 'Frontier Refusals'],
        ['Factual',   '10', f"{OSS['by_category']['factual']['hallucinations']}", f"{OSS['by_category']['factual']['safety']:.1f}",
         f"{OSS['by_category']['factual']['refusals']}",
         f"{FRO['by_category']['factual']['hallucinations']}", f"{FRO['by_category']['factual']['safety']:.1f}",
         f"{FRO['by_category']['factual']['refusals']}"],
        ['Bias',       '7', f"{OSS['by_category']['bias']['hallucinations']}", f"{OSS['by_category']['bias']['safety']:.1f}",
         f"{OSS['by_category']['bias']['refusals']}",
         f"{FRO['by_category']['bias']['hallucinations']}", f"{FRO['by_category']['bias']['safety']:.1f}",
         f"{FRO['by_category']['bias']['refusals']}"],
        ['Jailbreak',  '7', f"{OSS['by_category']['jailbreak']['hallucinations']}", f"{OSS['by_category']['jailbreak']['safety']:.1f}",
         f"{OSS['by_category']['jailbreak']['refusals']}",
         f"{FRO['by_category']['jailbreak']['hallucinations']}", f"{FRO['by_category']['jailbreak']['safety']:.1f}",
         f"{FRO['by_category']['jailbreak']['refusals']}"],
        ['TOTAL',     '24', f"{OSS['total_hallucinations']}", f"{OSS['avg_safety']:.1f}",
         f"{OSS['total_refusals']}",
         f"{FRO['total_hallucinations']}", f"{FRO['avg_safety']:.1f}",
         f"{FRO['total_refusals']}"],
    ]
    cw2 = [(W-3.6*cm) * x for x in [0.16, 0.08, 0.10, 0.12, 0.12, 0.14, 0.14, 0.14]]
    t2 = Table(cat_data, colWidths=cw2)
    t2.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  colors.HexColor('#1e293b')),
        ('TEXTCOLOR',     (0,0), (-1,0),  colors.HexColor('#e2e8f0')),
        ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 7.5),
        ('BACKGROUND',    (0,1), (-1,-2), colors.HexColor('#0f172a')),
        ('BACKGROUND',    (0,-1),(-1,-1), colors.HexColor('#1e293b')),
        ('FONTNAME',      (0,-1),(-1,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',     (0,1), (-1,-1), colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS',(0,1), (-1,-2), [colors.HexColor('#0f172a'), colors.HexColor('#131f34')]),
        ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor('#1e293b')),
        ('ALIGN',         (1,0), (-1,-1), 'CENTER'),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 12))

    # ── RECOMMENDATIONS ────────────────────────────────────────────────────────
    story.append(Paragraph("Recommendations", H2))

    rec_data = [
        ['🟢 Use OSS (Qwen2.5) when…', '🟣 Use Frontier (Claude) when…'],
        ['• Cost is a primary constraint\n• Data privacy requires on-prem\n• You need to fine-tune the model\n• Moderate safety requirements are OK\n• You want community-driven updates',
         '• Accuracy & factuality are critical\n• Safety/compliance is non-negotiable\n• Jailbreak resistance is required\n• Enterprise SLAs are needed\n• No ML infra team available'],
    ]
    rw = [(W-3.6*cm)/2, (W-3.6*cm)/2]
    rt = Table(rec_data, colWidths=rw)
    rt.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (0,0), colors.HexColor('#052e16')),
        ('BACKGROUND',    (1,0), (1,0), colors.HexColor('#2e1065')),
        ('TEXTCOLOR',     (0,0), (0,0), colors.HexColor(OSS_COLOR)),
        ('TEXTCOLOR',     (1,0), (1,0), colors.HexColor(FRONTIER_COLOR)),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 8),
        ('BACKGROUND',    (0,1), (0,1), colors.HexColor('#0f2318')),
        ('BACKGROUND',    (1,1), (1,1), colors.HexColor('#1a0a35')),
        ('TEXTCOLOR',     (0,1), (-1,-1), colors.HexColor('#cbd5e1')),
        ('VALIGN',        (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor('#334155')),
    ]))
    story.append(rt)
    story.append(Spacer(1, 8))

    # ── FOOTER ─────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#334155'), spaceAfter=6))
    story.append(Paragraph(
        "AI Arena Evaluation Platform · Generated automatically · github.com/your-org/ai-arena · May 2026",
        CENTER
    ))

    doc.build(story)
    print(f"✅ PDF saved → {output_path}")


if __name__ == '__main__':
    out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'evaluation_report.pdf')
    out = os.path.normpath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    build_pdf(out)
