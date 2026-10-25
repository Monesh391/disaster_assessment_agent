import io
import pandas as pd
from typing import Dict, Any, List, Optional
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def _process_image_for_reportlab(img_input: Any, max_width: float = 250, max_height: float = 175) -> Optional[RLImage]:
    """
    Safely converts a PIL Image, file path, or bytes into a ReportLab Image Flowable
    with preserved aspect ratio.
    """
    try:
        if img_input is None:
            return None
        if isinstance(img_input, str):
            pil_img = PILImage.open(img_input).convert("RGB")
        elif isinstance(img_input, bytes):
            pil_img = PILImage.open(io.BytesIO(img_input)).convert("RGB")
        elif hasattr(img_input, "convert"):
            pil_img = img_input.convert("RGB")
        else:
            return None
            
        orig_w, orig_h = pil_img.size
        if orig_w <= 0 or orig_h <= 0:
            return None
            
        ratio = min(max_width / orig_w, max_height / orig_h)
        target_w = orig_w * ratio
        target_h = orig_h * ratio
        
        # Fast downsample thumbnail so ReportLab builds in milliseconds
        thumb_w = int(max(80, min(orig_w, target_w * 2)))
        thumb_h = int(max(80, min(orig_h, target_h * 2)))
        pil_thumb = pil_img.resize((thumb_w, thumb_h), PILImage.Resampling.BILINEAR)
        
        img_buf = io.BytesIO()
        pil_thumb.save(img_buf, format="JPEG", quality=80)
        img_buf.seek(0)
        
        return RLImage(img_buf, width=target_w, height=target_h)
    except Exception as e:
        return None

def generate_recovery_plan_pdf(
    grid_ranked: pd.DataFrame, 
    overall_text: str = "",
    pre_image: Optional[Any] = None,
    post_image: Optional[Any] = None
) -> bytes:
    """
    Generates a clean, professional PDF document of the Master Recovery Plan
    using ReportLab.
    - Side-by-side pre- and post-disaster satellite imagery.
    - Primary Recovery Priority directive identifying which sector must be recovered first.
    - Regional priority matrix with guaranteed non-overflowing text cells.
    - Comprehensive multi-sector operational profiles with equipment needs and execution plans.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Title'],
        fontSize=17,
        leading=21,
        textColor=colors.HexColor('#0f172a'),
        alignment=0,
        fontName='Helvetica-Bold'
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#475569'),
        fontName='Helvetica'
    )
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#0f172a'),
        fontName='Helvetica-Bold',
        spaceBefore=10,
        spaceAfter=5
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontSize=10.5,
        leading=13.5,
        textColor=colors.HexColor('#1e293b'),
        fontName='Helvetica-Bold',
        spaceBefore=7,
        spaceAfter=3
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#334155'),
        fontName='Helvetica'
    )
    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#991b1b'),
        fontName='Helvetica-Bold'
    )
    
    # Table cell paragraph styles
    tbl_hdr_style = ParagraphStyle(
        'TblHdr',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1
    )
    tbl_cell_center = ParagraphStyle(
        'TblCellC',
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#1e293b'),
        alignment=1
    )
    tbl_cell_left = ParagraphStyle(
        'TblCellL',
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#1e293b'),
        alignment=0
    )
    tbl_cell_bold = ParagraphStyle(
        'TblCellB',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#0f172a'),
        alignment=1
    )
    img_lbl_style = ParagraphStyle(
        'ImgLbl',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#0f172a'),
        alignment=1
    )

    story = []
    
    # Header Banner
    story.append(Paragraph("MASTER DISASTER RECOVERY DIRECTIVE & OPERATIONAL PLAN", title_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph("Automated Satellite Damage Assessment & Spatial Prioritization System", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#cbd5e1"), spaceBefore=5, spaceAfter=8))
    
    # 0. Satellite Imagery Comparison (Pre & Post Disaster)
    rl_pre = _process_image_for_reportlab(pre_image, max_width=255, max_height=175)
    rl_post = _process_image_for_reportlab(post_image, max_width=255, max_height=175)
    
    if rl_pre is not None and rl_post is not None:
        img_table_data = [
            [
                Paragraph("<b>PRE-DISASTER SATELLITE IMAGERY</b>", img_lbl_style),
                Paragraph("<b>POST-DISASTER SATELLITE IMAGERY</b>", img_lbl_style)
            ],
            [rl_pre, rl_post]
        ]
        img_table = Table(img_table_data, colWidths=[270, 270])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 8))
    
    # 1. Primary Recovery Priority: Which Region Must Be Recovered First
    story.append(Paragraph("1. PRIMARY RECOVERY FOCUS — WHICH REGION MUST BE RECOVERED FIRST", h1_style))
    
    # Determine the top damaged region
    damaged_regions = grid_ranked[
        (grid_ranked["count_destroyed"] > 0) | 
        (grid_ranked["count_major_damage"] > 0) | 
        (grid_ranked["count_minor_damage"] > 0)
    ]
    
    if len(damaged_regions) > 0:
        top_row = damaged_regions.iloc[0]
        top_name = top_row["grid_id"]
        top_des = int(top_row["count_destroyed"])
        top_maj = int(top_row["count_major_damage"])
        top_min = int(top_row["count_minor_damage"])
        top_tot = int(top_row["total_buildings"])
        
        focus_text = (
            f"<b>CRITICAL DIRECTIVE:</b> <b>{top_name}</b> is designated as the <b>Primary Emergency Recovery Sector (RECOVER FIRST)</b>. "
            f"Ground and aerial analytics confirm {top_name} sustained the highest structural damage in the surveyed area "
            f"({top_des} destroyed, {top_maj} major damage, {top_min} minor damage out of {top_tot} assessed structures). "
            f"Emergency dispatch protocols mandate immediate deployment of heavy rescue assets, hazard cordoning, "
            f"and structural stabilization to this sector before secondary zones."
        )
    else:
        focus_text = (
            "<b>BASELINE STATUS:</b> No severe structural destruction was detected across surveyed sectors. "
            "All sectors are assigned routine municipal monitoring and baseline infrastructure safety verification."
        )
    
    story.append(Paragraph(focus_text, callout_style))
    story.append(Spacer(1, 8))
    
    # 2. Ranked Regional Assessment Table (Guaranteed Wrapping within column boundaries)
    story.append(Paragraph("2. REGIONAL DAMAGE & RECOVERY PRIORITY MATRIX", h1_style))
    
    # 9 columns totaling exactly 540 pt: 26 + 52 + 72 + 30 + 44 + 40 + 40 + 48 + 188 = 540 pt
    col_widths = [26, 52, 72, 30, 44, 40, 40, 48, 188]
    
    table_data = [
        [
            Paragraph("Rank", tbl_hdr_style),
            Paragraph("Sector", tbl_hdr_style),
            Paragraph("Priority Tier", tbl_hdr_style),
            Paragraph("Total", tbl_hdr_style),
            Paragraph("Destroyed", tbl_hdr_style),
            Paragraph("Major", tbl_hdr_style),
            Paragraph("Minor", tbl_hdr_style),
            Paragraph("Undamaged", tbl_hdr_style),
            Paragraph("Recovery Directive", tbl_hdr_style)
        ]
    ]
    
    for idx, row in grid_ranked.iterrows():
        rank_str = f"#{idx + 1}"
        sector_name = str(row["grid_id"])
        tier_str = str(row.get("priority_label", "Recover Later"))
        tot_b = str(int(row["total_buildings"]))
        des_b = str(int(row["count_destroyed"]))
        maj_b = str(int(row["count_major_damage"]))
        min_b = str(int(row["count_minor_damage"]))
        und_b = str(int(row["count_no_damage"]))
        action_str = str(row.get("action", "Routine monitoring"))
        
        table_data.append([
            Paragraph(rank_str, tbl_cell_center),
            Paragraph(sector_name, tbl_cell_bold),
            Paragraph(tier_str, tbl_cell_center),
            Paragraph(tot_b, tbl_cell_center),
            Paragraph(f"<font color='#dc2626'><b>{des_b}</b></font>", tbl_cell_center),
            Paragraph(f"<font color='#ea580c'><b>{maj_b}</b></font>", tbl_cell_center),
            Paragraph(f"<font color='#ca8a04'><b>{min_b}</b></font>", tbl_cell_center),
            Paragraph(f"<font color='#16a34a'><b>{und_b}</b></font>", tbl_cell_center),
            Paragraph(action_str, tbl_cell_left)
        ])
        
    t = Table(table_data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,0), 4),
        ('TOPPADDING', (0,0), (-1,0), 4),
        ('BOTTOMPADDING', (0,1), (-1,-1), 3),
        ('TOPPADDING', (0,1), (-1,-1), 3),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8fafc'), colors.white]),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))
    
    # 3. Overall Sector Breakdown: What It Needs & How It Can Be Recovered First
    story.append(Paragraph("3. COMPREHENSIVE SECTOR PROFILES: RESOURCE NEEDS & ACTION PROCEDURES", h1_style))
    story.append(Paragraph(
        "Detailed operational requirements for every sector detailing structural impact, specific tactical equipment needs, "
        "and sequential recovery procedures.",
        body_style
    ))
    story.append(Spacer(1, 5))
    
    for idx, row in grid_ranked.iterrows():
        r_name = row["grid_id"]
        tier_label = row.get("priority_label", "Recover Later")
        b_tot = int(row["total_buildings"])
        b_des = int(row["count_destroyed"])
        b_maj = int(row["count_major_damage"])
        b_min = int(row["count_minor_damage"])
        b_und = int(row["count_no_damage"])
        
        needs = row.get("needs", "")
        how_to = row.get("how_to_recover", "")
        
        sector_block = []
        sector_block.append(Paragraph(f"<b>Sector: {r_name}</b> — <i>Priority: {tier_label} (Rank #{idx+1})</i>", h2_style))
        sector_block.append(Paragraph(
            f"<b>Structural Damage:</b> {b_tot} assessed structures | "
            f"<font color='#dc2626'><b>{b_des} destroyed</b></font> | "
            f"<font color='#ea580c'><b>{b_maj} major damage</b></font> | "
            f"<font color='#ca8a04'><b>{b_min} minor damage</b></font> | "
            f"<font color='#16a34a'><b>{b_und} undamaged</b></font>",
            body_style
        ))
        sector_block.append(Spacer(1, 2))
        sector_block.append(Paragraph(f"<b>What It Needs (Required Resources):</b> {needs}", body_style))
        sector_block.append(Spacer(1, 2))
        sector_block.append(Paragraph(f"<b>How It Can Be Recovered (Execution Plan):</b> {how_to}", body_style))
        sector_block.append(Spacer(1, 6))
        
        story.append(KeepTogether(sector_block))
        
    # Note: Recommended Recovery Timeline & Execution Phases is removed per user request.
    
    doc.build(story)
    return buf.getvalue()
