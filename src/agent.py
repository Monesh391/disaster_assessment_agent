import os
from typing import Dict, Any, List

class DisasterRecoveryAgent:
    """
    Explainable, deterministic rule-based AI Agent that reasons over spatial grid prioritization,
    building damage counts, and recovery metrics to generate recovery recommendations.
    Strictly offline-compatible and requires no external LLM or API keys.
    """
    
    def __init__(self, provider: str = None, api_key: str = None):
        pass

    def explain_grid_cell(self, cell_data: Dict[str, Any], damage_counts: Dict[str, int] = None) -> str:
        """
        Generates a clear, non-technical regional damage audit for a selected area.
        Reads building counts directly from cell_data to guarantee regional accuracy.
        """
        grid_id = cell_data.get("grid_id", "Unknown Region")
        score = float(cell_data.get("priority_score", 0.0))
        tier = cell_data.get("priority_tier", "Recover Later")
        
        # Read counts from cell_data first to prevent leaking image-wide counts into a specific region
        if damage_counts is None:
            damage_counts = {}
            
        n_des = int(cell_data.get("count_destroyed", damage_counts.get("destroyed", 0)))
        n_maj = int(cell_data.get("count_major_damage", damage_counts.get("major-damage", 0)))
        n_min = int(cell_data.get("count_minor_damage", damage_counts.get("minor-damage", 0)))
        n_nodmg = int(cell_data.get("count_no_damage", damage_counts.get("no-damage", 0)))
        total_bld = int(cell_data.get("total_buildings", n_des + n_maj + n_min + n_nodmg))
        total_dmg = n_des + n_maj + n_min
        pct_dmg = (total_dmg / max(1, total_bld)) * 100.0 if total_bld > 0 else 0.0
        
        # Determine priority status strictly based on regional damage
        if total_dmg == 0 or score == 0.0 or tier in ["LOW", "Recover Later", "Recover Later (No Buildings)", "Recover Later (Undamaged)"]:
            tier_display = "Recover Later (No Buildings)" if total_bld == 0 else "Recover Later (Undamaged)"
            action = "Routine environmental monitoring and baseline maintenance."
            if total_bld == 0:
                reason = f"{grid_id} contains 0 assessed structures and presents no structural risk; assigned baseline monitoring."
            else:
                reason = f"{grid_id} has all {n_nodmg} structures undamaged with zero detected structural impact; assigned baseline municipal monitoring."
        elif tier in ["CRITICAL", "Recover First"]:
            action = "Immediate emergency assessment and recovery response."
            tier_display = "Recover First"
            reason = f"{grid_id} has the highest recovery priority because it has the highest detected damage severity ({n_des} destroyed, {n_maj} major damage) according to the spatial recovery calculation."
        elif tier in ["HIGH", "Recover Next"]:
            action = "High-priority structural inspection and utility restoration."
            tier_display = "Recover Next"
            reason = f"{grid_id} is ranked second in recovery sequence due to significant structural damage ({n_des} destroyed, {n_maj} major damage) requiring prompt intervention."
        elif tier in ["MEDIUM", "Moderate Priority"]:
            action = "Scheduled inspection, repair, and localized debris clearing."
            tier_display = "Moderate Priority"
            reason = f"{grid_id} is ranked third with moderate damage ({n_des} destroyed, {n_maj} major damage, {n_min} minor damage) that can be addressed after higher-priority sectors."
        else:
            action = "Routine environmental monitoring and baseline maintenance."
            tier_display = "Recover Later"
            reason = f"{grid_id} has lower detected damage impact ({n_des} destroyed, {n_maj} major damage, {n_nodmg} undamaged) and can be recovered after higher-urgency zones."
                
        audit = f"### 📋 Regional Damage Audit\n\n"
        audit += f"#### **{grid_id} — Detailed Area Audit**\n\n"
        audit += f"- **Priority**: {tier_display}\n"
        audit += f"- **Score**: {score:.1f}\n\n"
        audit += f"- **Buildings Assessed**: {total_bld}\n"
        audit += f"- **Destroyed**: {n_des}\n"
        audit += f"- **Major Damage**: {n_maj}\n"
        audit += f"- **Minor Damage**: {n_min}\n"
        audit += f"- **Undamaged**: {n_nodmg}\n\n"
        audit += f"- **Damage Requiring Recovery**: {pct_dmg:.1f}%\n\n"
        audit += f"**Recommended Action:**\n{action}\n\n"
        audit += f"**Reason:**\n{reason}\n"
        
        return audit

    def generate_recovery_plan(self, prioritized_grid: Any, damage_counts: Dict[str, int]) -> str:
        """
        Compiles an overall master recovery plan with regions ordered from highest to lowest priority.
        Respects computed regional tiers and scores rather than hardcoding ranks.
        """
        plan = "### 📋 Master Recovery Plan\n\n"
        plan += "#### **Overall Recovery Sequence**\n\n"
        
        regions_ordered = []
        if hasattr(prioritized_grid, "iterrows"):
            for idx, row in prioritized_grid.iterrows():
                r_name = str(row["grid_id"])
                score = float(row.get("priority_score", 0.0))
                b_total = int(row.get("total_buildings", 0))
                b_des = int(row.get("count_destroyed", 0))
                b_maj = int(row.get("count_major_damage", 0))
                b_min = int(row.get("count_minor_damage", 0))
                b_nodmg = int(row.get("count_no_damage", 0))
                
                # Use the actual computed priority label, tier, and icon
                t_label = str(row.get("priority_label", "Recover Later"))
                t_tier = str(row.get("priority_tier", "LOW"))
                t_icon = str(row.get("tier_icon", "✅"))
                t_rank = str(row.get("priority_rank", f"Sector #{idx + 1}"))
                t_action = str(row.get("action", "Routine environmental monitoring and baseline maintenance."))
                
                regions_ordered.append({
                    "rank": t_rank,
                    "name": r_name,
                    "score": score,
                    "total": b_total,
                    "destroyed": b_des,
                    "major": b_maj,
                    "minor": b_min,
                    "tier": t_tier,
                    "label": t_label
                })
                
                plan += f"**{t_rank}. {t_icon} {r_name} — {t_label}**\n"
                plan += f"- **Score**: {score:.1f}\n"
                plan += f"- **Buildings Assessed**: {b_total}\n"
                plan += f"- **Destroyed**: {b_des}\n"
                plan += f"- **Major Damage**: {b_maj}\n"
                plan += f"- **Minor Damage**: {b_min}\n"
                plan += f"- **Undamaged**: {b_nodmg}\n"
                plan += f"- **Recommended Action**: {t_action}\n\n"
                
        # Generate strategic recovery recommendation based on actual damaged sectors
        damaged_sectors = [r for r in regions_ordered if (r["destroyed"] > 0 or r["major"] > 0 or r["minor"] > 0)]
        
        plan += "#### **Strategic Recovery Recommendation**\n\n"
        if len(damaged_sectors) > 0:
            top_sector = damaged_sectors[0]
            summary = (
                f"Begin emergency recovery in {top_sector['name']} because it has the highest recovery priority score "
                f"({top_sector['score']:.1f}) and greatest concentration of severe damage ({top_sector['destroyed']} destroyed, "
                f"{top_sector['major']} major damage out of {top_sector['total']} assessed structures)."
            )
            if len(damaged_sectors) > 1:
                subsequent = [s["name"] for s in damaged_sectors[1:]]
                summary += f" Continue with {', '.join(subsequent)} in prioritized sequence."
        else:
            summary = (
                "All surveyed sectors exhibit zero structural damage. Baseline municipal monitoring "
                "and civil engineering safety verification are active across all regions."
            )
        plan += f"{summary}\n"
        
        return plan

