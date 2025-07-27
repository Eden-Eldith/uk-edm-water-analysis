#!/usr/bin/env python3
"""
Correlation Analysis for Water Quality - Final Production Version
Analyzes correlated events to identify the most suspicious overflow sites
"""

import pandas as pd
import numpy as np

# === CONFIGURATION ===
INPUT_FILE = "correlated_events_final.csv"
OUTPUT_FILE = "suspicious_overflow_summary.csv"
TOP_N_RESULTS = 50

# Risk thresholds
LOW_DO_THRESHOLD = 2.0      # mg/L - below this is critically low
CRITICAL_DO_THRESHOLD = 1.0  # mg/L - below this is extremely critical
MIN_DISTANCE_M = 10         # Minimum distance for calculations


# === ANALYSIS FUNCTIONS ===

def categorize_do_level(value):
    """Categorize dissolved oxygen levels."""
    if pd.isna(value):
        return "Unknown"
    elif value < CRITICAL_DO_THRESHOLD:
        return "Critical"
    elif value < LOW_DO_THRESHOLD:
        return "Low"
    elif value < 4.0:
        return "Moderate"
    else:
        return "Normal"


def calculate_event_score(row):
    """
    Calculate severity score for a single pollution event.
    Returns (score, severity_category)
    """
    distance = max(row['distance_m'], MIN_DISTANCE_M)
    concentration = row['pollutant_numeric']
    
    if pd.isna(concentration):
        return 0, "Unknown"
    
    pollutant_label = str(row['pollutant']).lower()
    base_score = 0
    severity = "Unknown"
    
    if row['is_dissolved_oxygen']:
        # Low DO is bad - inverse scoring
        if concentration < CRITICAL_DO_THRESHOLD:
            base_score = 2000 / max(concentration, 0.01)
            severity = "Critical DO"
        elif concentration < LOW_DO_THRESHOLD:
            base_score = 1000 / max(concentration, 0.1)
            severity = "Low DO"
        else:
            base_score = 100
            severity = "Normal DO"
    else:
        # Regular pollutants - higher concentration is worse
        
        # Check for BOD (Biochemical Oxygen Demand)
        if 'bod' in pollutant_label:
            base_score = concentration * 1800
            severity = "BOD"
        # Check for suspended solids
        elif any(term in pollutant_label for term in ['sld sus', 'suspended solid']):
            base_score = concentration * 1600
            severity = "Solids"
        # Sewage indicators
        elif any(term in pollutant_label for term in ['sewage', 'pyridine', 'faecal', 'e.coli', 'enterococci']):
            base_score = concentration * 2000
            severity = "Sewage/Bacterial"
        # Oil and grease
        elif any(term in pollutant_label for term in ['oil', 'grease', 'pah']):
            base_score = concentration * 1500
            severity = "Oil/Grease"
        # Ammonia
        elif 'ammonia' in pollutant_label:
            base_score = concentration * 1200
            severity = "Ammonia"
        # Other pollutants
        else:
            base_score = concentration * 1000
            severity = "Other"
    
    # Distance-adjusted score
    final_score = base_score / distance
    return final_score, severity


def analyze_correlations(filepath):
    """
    Main analysis function using multi-factor risk assessment.
    """
    print("\n" + "="*80)
    print("WATER QUALITY CORRELATION ANALYSIS - RISK ASSESSMENT")
    print("="*80)
    
    # Load data
    try:
        df = pd.read_csv(filepath, parse_dates=['sample_time'], low_memory=False)
        print(f"\n✅ Loaded {len(df):,} correlated events")
    except FileNotFoundError:
        print(f"\n❌ ERROR: Input file '{filepath}' not found")
        print("   Please run batch_filter_final.py first")
        return
    except Exception as e:
        print(f"\n❌ ERROR loading file: {e}")
        return
    
    # Data preparation
    df['pollutant_numeric'] = pd.to_numeric(df['pollutant_result'], errors='coerce')
    initial_count = len(df)
    df = df.dropna(subset=['pollutant_numeric'])
    dropped_count = initial_count - len(df)
    
    if dropped_count > 0:
        print(f"ℹ️  Dropped {dropped_count:,} events with non-numeric results")
    
    # Ensure magnitude data is numeric
    df['overflow_spill_count'] = pd.to_numeric(df['overflow_spill_count'], errors='coerce').fillna(0)
    df['overflow_spill_duration_hrs'] = pd.to_numeric(df['overflow_spill_duration_hrs'], errors='coerce').fillna(0)
    
    # Calculate event scores
    print("\n⏳ Calculating event severity scores...")
    df[['event_score', 'severity_category']] = df.apply(calculate_event_score, axis=1, result_type='expand')
    
    # Add DO category
    df['do_category'] = df.apply(
        lambda r: categorize_do_level(r['pollutant_numeric']) if r['is_dissolved_oxygen'] else None,
        axis=1
    )
    
    # Show event breakdown
    print("\n📊 EVENTS BY SEVERITY:")
    severity_counts = df['severity_category'].value_counts()
    for category, count in severity_counts.items():
        print(f"  {category}: {count:,}")
    
    # Aggregate by site
    print("\n⏳ Aggregating data by overflow site...")
    
    # First get unique magnitude data per site per year
    annual_magnitude = df.groupby(['overflow_site', 'year']).agg(
        annual_spill_count=('overflow_spill_count', 'first'),
        annual_spill_duration=('overflow_spill_duration_hrs', 'first')
    ).reset_index()
    
    # Sum across years for total magnitude
    site_magnitude = annual_magnitude.groupby('overflow_site').agg(
        total_spill_count=('annual_spill_count', 'sum'),
        total_spill_duration_hrs=('annual_spill_duration', 'sum')
    ).reset_index()
    
    # Aggregate pollution events
    site_pollution = df.groupby('overflow_site').agg(
        # Score metrics
        pollution_score_p95=('event_score', lambda x: x.quantile(0.95)),
        pollution_score_max=('event_score', 'max'),
        total_pollution_events=('event_score', 'count'),
        
        # Distance metrics
        median_distance_m=('distance_m', 'median'),
        min_distance_m=('distance_m', 'min'),
        
        # DO events
        critical_do_count=('do_category', lambda x: (x == 'Critical').sum()),
        low_do_count=('do_category', lambda x: (x == 'Low').sum()),
        
        # Pollutant type events
        sewage_events=('severity_category', lambda x: x.isin(['Sewage/Bacterial']).sum()),
        ammonia_events=('severity_category', lambda x: (x == 'Ammonia').sum()),
        bod_events=('severity_category', lambda x: (x == 'BOD').sum()),
        solids_events=('severity_category', lambda x: (x == 'Solids').sum()),
        oil_grease_events=('severity_category', lambda x: (x == 'Oil/Grease').sum()),
        
        # Temporal metrics
        years_active=('year', 'nunique'),
        first_year=('year', 'min'),
        last_year=('year', 'max')
    ).reset_index()
    
    # Merge magnitude and pollution data
    summary = pd.merge(site_pollution, site_magnitude, on='overflow_site', how='left')
    summary[['total_spill_count', 'total_spill_duration_hrs']] = summary[['total_spill_count', 'total_spill_duration_hrs']].fillna(0)
    
    # Calculate risk components
    print("\n⏳ Calculating composite risk scores...")
    
    # 1. Raw risk components
    summary['raw_magnitude'] = np.log1p(
        summary['total_spill_count'] + 
        (summary['total_spill_duration_hrs'] / 8)  # Convert hours to ~shifts
    )
    
    summary['raw_severity'] = np.log1p(
        (summary['critical_do_count'] * 5) +
        (summary['low_do_count'] * 2) +
        (summary['sewage_events'] * 3) +
        (summary['ammonia_events'] * 2) +
        (summary['bod_events'] * 2) +
        (summary['solids_events'] * 1.5) +
        (summary['oil_grease_events'] * 1)
    )
    
    summary['raw_proximity'] = 1 / np.log1p(summary['median_distance_m'])
    summary['raw_pollution'] = np.log1p(summary['pollution_score_p95'].fillna(0))
    
    # 2. Normalize components (0-1 scale)
    for component in ['magnitude', 'severity', 'proximity', 'pollution']:
        raw_col = f'raw_{component}'
        norm_col = f'norm_{component}'
        max_val = summary[raw_col].max()
        if max_val > 0:
            summary[norm_col] = summary[raw_col] / max_val
        else:
            summary[norm_col] = 0
    
    # 3. Weighted composite score
    weights = {
        'magnitude': 0.35,   # How much they spill
        'severity': 0.30,    # What type of pollution
        'pollution': 0.20,   # Measured concentrations
        'proximity': 0.15,   # How close to water
    }
    
    summary['composite_risk_score'] = (
        summary['norm_magnitude'] * weights['magnitude'] +
        summary['norm_severity'] * weights['severity'] +
        summary['norm_pollution'] * weights['pollution'] +
        summary['norm_proximity'] * weights['proximity']
    ) * 100  # Scale to 0-100
    
    # 4. Add persistence bonus (up to 10% for multi-year offenders)
    max_years = summary['years_active'].max()
    if max_years > 0:
        summary['persistence_bonus'] = (summary['years_active'] / max_years) * 0.1
        summary['composite_risk_score'] *= (1 + summary['persistence_bonus'])
    
    # 5. Categorize risk levels
    summary['risk_category'] = pd.cut(
        summary['composite_risk_score'],
        bins=[-np.inf, 10, 25, 50, 75, np.inf],
        labels=['Low', 'Moderate', 'High', 'Very High', 'Critical']
    )
    
    # Sort and save
    summary = summary.sort_values('composite_risk_score', ascending=False).reset_index(drop=True)
    summary.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Analysis complete! Results saved to '{OUTPUT_FILE}'")
    
    # Display top results
    print("\n" + "="*80)
    print(f"TOP {TOP_N_RESULTS} SUSPICIOUS OVERFLOW SITES")
    print("="*80)
    
    # Select display columns
    display_cols = [
        'overflow_site', 'composite_risk_score', 'risk_category',
        'total_spill_count', 'total_spill_duration_hrs',
        'total_pollution_events', 'critical_do_count',
        'sewage_events', 'ammonia_events',
        'median_distance_m', 'years_active'
    ]
    
    top_sites = summary.head(TOP_N_RESULTS)[display_cols].copy()
    
    # Format for display
    top_sites['composite_risk_score'] = top_sites['composite_risk_score'].round(1)
    top_sites['total_spill_duration_hrs'] = top_sites['total_spill_duration_hrs'].round(0)
    top_sites['median_distance_m'] = top_sites['median_distance_m'].round(0)
    
    # Set display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', TOP_N_RESULTS)
    
    print(top_sites.to_string(index=False))
    
    # Critical alerts
    print("\n" + "="*80)
    print("⚠️  CRITICAL ALERTS")
    print("="*80)
    
    # Sites with critical risk
    critical_sites = summary[summary['risk_category'] == 'Critical']
    if len(critical_sites) > 0:
        print(f"\n🚨 {len(critical_sites)} sites classified as CRITICAL risk:")
        for _, site in critical_sites.head(10).iterrows():
            print(f"   - {site['overflow_site']}: Score {site['composite_risk_score']:.1f}")
    
    # Sites with extreme spill duration
    extreme_duration = summary[summary['total_spill_duration_hrs'] > 1000].head(5)
    if len(extreme_duration) > 0:
        print(f"\n⏱️  Sites with extreme spill duration (>1000 hours):")
        for _, site in extreme_duration.iterrows():
            print(f"   - {site['overflow_site']}: {site['total_spill_duration_hrs']:.0f} hours")
    
    # Sites with critical DO events
    critical_do_sites = summary[summary['critical_do_count'] > 0].head(5)
    if len(critical_do_sites) > 0:
        print(f"\n💀 Sites with critical dissolved oxygen events:")
        for _, site in critical_do_sites.iterrows():
            print(f"   - {site['overflow_site']}: {site['critical_do_count']} critical DO events")
    
    # Summary statistics
    print("\n" + "="*80)
    print("📊 OVERALL STATISTICS")
    print("="*80)
    
    print(f"\nTotal sites analyzed: {len(summary)}")
    print(f"\nRisk distribution:")
    risk_dist = summary['risk_category'].value_counts().sort_index()
    for category, count in risk_dist.items():
        percentage = (count / len(summary)) * 100
        print(f"  {category}: {count} sites ({percentage:.1f}%)")
    
    print(f"\nPollution impact summary:")
    print(f"  Sites with critical DO events: {(summary['critical_do_count'] > 0).sum()}")
    print(f"  Sites with sewage/bacterial events: {(summary['sewage_events'] > 0).sum()}")
    print(f"  Sites with ammonia events: {(summary['ammonia_events'] > 0).sum()}")
    print(f"  Average events per site: {summary['total_pollution_events'].mean():.1f}")
    
    print(f"\nRecommendations:")
    print(f"  IMMEDIATE investigation required: {len(critical_sites)} sites")
    very_high_sites = summary[summary['risk_category'] == 'Very High']
    print(f"  URGENT investigation recommended: {len(very_high_sites)} sites")
    print(f"  Total high-priority sites: {len(critical_sites) + len(very_high_sites)}")
    
    # Top site breakdown
    if len(summary) > 0:
        top_site = summary.iloc[0]
        print(f"\n🔍 Detailed breakdown for #1 site: {top_site['overflow_site']}")
        print(f"   Composite Risk Score: {top_site['composite_risk_score']:.1f}")
        print(f"   Risk Components:")
        print(f"     - Magnitude (normalized): {top_site['norm_magnitude']:.2f} × {weights['magnitude']} = {top_site['norm_magnitude'] * weights['magnitude']:.3f}")
        print(f"     - Severity (normalized):  {top_site['norm_severity']:.2f} × {weights['severity']} = {top_site['norm_severity'] * weights['severity']:.3f}")
        print(f"     - Pollution (normalized): {top_site['norm_pollution']:.2f} × {weights['pollution']} = {top_site['norm_pollution'] * weights['pollution']:.3f}")
        print(f"     - Proximity (normalized): {top_site['norm_proximity']:.2f} × {weights['proximity']} = {top_site['norm_proximity'] * weights['proximity']:.3f}")
        if 'persistence_bonus' in top_site:
            print(f"     - Persistence bonus: +{top_site['persistence_bonus']*100:.1f}%")


# === MAIN EXECUTION ===

def main():
    analyze_correlations(INPUT_FILE)


if __name__ == "__main__":
    main()
