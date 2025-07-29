#!/usr/bin/env python3
"""
Batch Filter for Water Quality Analysis - Final Production Version
Correlates pollution samples with EDM (Event Duration Monitoring) overflow data
"""

import pandas as pd
import os
import re
import numpy as np
from datetime import datetime

# Check for required dependencies
try:
    from scipy.spatial import KDTree
except ImportError:
    print("ERROR: scipy is required. Install with: pip install scipy")
    exit(1)

# === CONFIGURATION ===
EDM_DIR = "xlsx"
SAMPLE_DIR = "csvs"
OUTPUT_FILE = "correlated_events_final.csv"

# Analysis Parameters
RADIUS_METERS = 5000  # 5km radius for correlation
MIN_DISTANCE_M = 10   # Minimum distance to prevent division by zero

# UK coordinate bounds for validation (approximate)
UK_EASTING_MIN = 0
UK_EASTING_MAX = 700000
UK_NORTHING_MIN = 0
UK_NORTHING_MAX = 1300000

# Comprehensive pollutant regex including all variations found across iterations
TARGET_POLLUTANTS_REGEX = r"Oil|Grease|Ammonia|PAH|pyridine|sewage|DO\b|Dissolved\s*Oxygen|Oxygen\s*Diss|BOD|Sld Sus|Suspended\s*Solids"
DO_SPECIFIC_REGEX = r"\bDO\b|Dissolved\s*Oxygen|Oxygen\s*Diss|Oxygen,\s*Dissolved"


# === UTILITY FUNCTIONS ===

def parse_duration_to_hours(duration_val):
    """Convert various duration formats to hours."""
    if pd.isna(duration_val):
        return 0.0
    
    # Already numeric
    if isinstance(duration_val, (int, float)):
        return float(duration_val)
    
    # String format HH:MM:SS
    if isinstance(duration_val, str):
        try:
            parts = duration_val.split(':')
            if len(parts) == 3:
                h, m, s = map(int, parts)
                return h + m/60 + s/3600
        except:
            pass
    
    # datetime.time object
    if hasattr(duration_val, 'hour'):
        return duration_val.hour + duration_val.minute/60 + duration_val.second/3600
    
    return 0.0


def validate_uk_coordinates(easting, northing):
    """Validate that coordinates are within reasonable UK bounds."""
    if pd.isna(easting) or pd.isna(northing):
        return False
    
    return (UK_EASTING_MIN <= easting <= UK_EASTING_MAX and 
            UK_NORTHING_MIN <= northing <= UK_NORTHING_MAX)

def parse_osgr(osgr_str):
    """Convert OS Grid Reference to (easting, northing) coordinates with validation."""
    try:
        # Clean and normalize
        osgr_str = re.sub(r'\s+', '', str(osgr_str)).upper()
        if len(osgr_str) < 4:
            return None, None
        
        # Extract grid letters and numbers
        match = re.match(r'([A-Z]{2})(\d+)', osgr_str)
        if not match:
            return None, None
        
        grid_letters, grid_numbers = match.groups()
        
        # Must have even number of digits
        if len(grid_numbers) % 2 != 0:
            return None, None
        
        # Split easting and northing
        num_digits = len(grid_numbers) // 2
        east_val = int(grid_numbers[:num_digits].ljust(5, '0'))
        north_val = int(grid_numbers[num_digits:].ljust(5, '0'))
        
        # Comprehensive grid map
        grid_map = {
            'SV': (0, 0), 'SW': (100000, 0), 'SX': (200000, 0), 'SY': (300000, 0), 'SZ': (400000, 0),
            'TV': (500000, 0), 'TW': (600000, 0),
            'SQ': (0, 100000), 'SR': (100000, 100000), 'SS': (200000, 100000), 'ST': (300000, 100000), 
            'SU': (400000, 100000), 'TQ': (500000, 100000), 'TR': (600000, 100000),
            'SL': (0, 200000), 'SM': (100000, 200000), 'SN': (200000, 200000), 'SO': (300000, 200000), 
            'SP': (400000, 200000), 'TL': (500000, 200000), 'TM': (600000, 200000),
            'SH': (200000, 300000), 'SC': (200000, 400000), 'TA': (500000, 400000),
            'NA': (100000, 600000), 'NB': (200000, 600000), 'NC': (300000, 600000), 'ND': (400000, 600000),
            'NF': (100000, 700000), 'NG': (200000, 700000), 'NH': (300000, 700000), 'NJ': (400000, 700000), 
            'NK': (500000, 700000), 'NL': (100000, 800000), 'NM': (200000, 800000), 'NN': (300000, 800000), 
            'NO': (300000, 700000), 'NR': (100000, 900000), 'NS': (200000, 1000000), 'NT': (300000, 900000), 
            'NU': (400000, 900000), 'HT': (400000, 500000), 'HU': (500000, 500000), 'HW': (100000, 1000000), 
            'HX': (200000, 1000000), 'HY': (300000, 1000000), 'HZ': (400000, 1000000), 'HP': (400000, 1100000), 
            'OV': (0, 1000000)
        }
        
        if grid_letters not in grid_map:
            return None, None
        
        base_e, base_n = grid_map[grid_letters]
        final_easting = base_e + east_val
        final_northing = base_n + north_val
        
        # Validate coordinates are within UK bounds
        if not validate_uk_coordinates(final_easting, final_northing):
            return None, None
        
        return final_easting, final_northing
        
    except (ValueError, TypeError, AttributeError):
        return None, None


def extract_year(filename):
    """Extract year from filename."""
    match = re.search(r'(20\d{2})', filename)
    return match.group(1) if match else None


def normalize_do_value(value, unit):
    """Normalize dissolved oxygen values to mg/L."""
    try:
        numeric_value = float(value)
        # Convert percentage to mg/L (assuming ~8mg/L = 100% at sea level)
        if unit and '%' in str(unit):
            return numeric_value * 0.08
        return numeric_value
    except (ValueError, TypeError):
        return None


def find_column(df_columns, keywords):
    """Find column name containing any of the keywords (case-insensitive)."""
    for col in df_columns:
        col_lower = str(col).lower().replace('\n', ' ')
        for keyword in keywords:
            if keyword.lower() in col_lower:
                return col
    return None


def find_header_row(filepath, keywords):
    """Find the row containing all specified keywords."""
    try:
        df_preview = pd.read_excel(filepath, header=None, nrows=20, engine="openpyxl")
        for i, row in df_preview.iterrows():
            row_str = ' '.join(str(s).lower().replace('\n', ' ') for s in row.dropna())
            if all(keyword.lower() in row_str for keyword in keywords):
                return i
    except Exception as e:
        print(f"  Error previewing {filepath}: {e}")
    return None


# === DATA LOADERS ===

def load_all_edm_data(directory):
    """Load and parse all EDM Excel files."""
    all_edm_data = []
    header_keywords = ["site name", "ngr"]
    
    print("\n=== Loading EDM Overflow Data ===")
    
    for filename in os.listdir(directory):
        if not filename.endswith(".xlsx") or "summary" in filename.lower():
            continue
        
        filepath = os.path.join(directory, filename)
        year = extract_year(filename)
        if not year:
            print(f"⚠️  Skipping {filename}: Cannot determine year")
            continue
        
        print(f"\nProcessing {filename} (Year: {year})")
        
        try:
            # Special handling for known problematic 2020 files
            if "2020" in filename and any(water_company in filename.lower() 
                                         for water_company in ['thames', 'yorkshire', 'anglian', 'welsh', 
                                                              'northumbrian', 'severn', 'southern', 
                                                              'south_west', 'united', 'wessex']):
                print(f"  ℹ️  Skipping known 2020 file without coordinates: {filename}")
                continue
            
            # Find header row
            header_row = find_header_row(filepath, header_keywords)
            if header_row is None:
                print(f"  ⚠️  Cannot find header row in {filename}")
                continue
            
            # Load data
            df = pd.read_excel(filepath, header=header_row, engine="openpyxl")
            df.columns = df.columns.map(lambda x: str(x).replace('\n', ' ').strip())
            
            # Find required columns
            site_col = find_column(df.columns, ['site name'])
            ngr_col = find_column(df.columns, ['outlet discharge ngr', 'ngr'])
            duration_col = find_column(df.columns, ['total duration'])
            count_col = find_column(df.columns, ['counted spills', 'spills using 12-24hr'])
            
            if not all([site_col, ngr_col]):
                print(f"  ⚠️  Missing required columns in {filename}")
                continue
            
            # Create clean dataframe
            clean_df = pd.DataFrame()
            clean_df['overflow_site'] = df[site_col]
            clean_df['overflow_osgr'] = df[ngr_col]
            clean_df['overflow_spill_duration_hrs'] = df[duration_col].apply(parse_duration_to_hours) if duration_col else 0
            clean_df['overflow_spill_count'] = pd.to_numeric(df[count_col], errors='coerce').fillna(0) if count_col else 0
            clean_df['year'] = year
            
            # Parse coordinates
            clean_df[['easting', 'northing']] = clean_df['overflow_osgr'].apply(lambda x: pd.Series(parse_osgr(x)))
            
            # Drop invalid coordinates
            initial_count = len(clean_df)
            clean_df = clean_df.dropna(subset=['easting', 'northing', 'overflow_site'])
            valid_count = len(clean_df)
            
            if valid_count > 0:
                all_edm_data.append(clean_df)
                print(f"  ✅ Parsed {valid_count}/{initial_count} valid records")
            else:
                print(f"  ⚠️  No valid coordinates found")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    if not all_edm_data:
        raise SystemExit("\n💥 FAILURE: No EDM files could be parsed successfully")
    
    combined = pd.concat(all_edm_data, ignore_index=True)
    print(f"\n📊 Total EDM records loaded: {len(combined)}")
    print(f"📍 Years covered: {sorted(combined['year'].unique())}")
    return combined


def load_sample_data(filepath):
    """Load and standardize a sample CSV file with enhanced validation."""
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        raise ValueError(f"Failed to load CSV file {filepath}: {e}")
    
    if df.empty:
        raise ValueError(f"CSV file {filepath} is empty")
    
    # Map columns (case-insensitive)
    column_mapping = {}
    
    # Find and map each required column
    datetime_col = find_column(df.columns, ['sample.sampledatetime', 'sampledatetime'])
    if datetime_col:
        column_mapping[datetime_col] = 'sample_datetime'
    
    pollutant_col = find_column(df.columns, ['determinand.label', 'determinand', 'pollutant'])
    if pollutant_col:
        column_mapping[pollutant_col] = 'pollutant'
    
    result_col = find_column(df.columns, ['result', 'value'])
    if result_col:
        column_mapping[result_col] = 'result'
    
    qualifier_col = find_column(df.columns, ['resultqualifier.notation', 'qualifier'])
    if qualifier_col:
        column_mapping[qualifier_col] = 'qualifier'
    
    unit_col = find_column(df.columns, ['determinand.unit.label', 'unit'])
    if unit_col:
        column_mapping[unit_col] = 'unit'
    
    site_col = find_column(df.columns, ['sample.samplingpoint.label', 'site', 'location'])
    if site_col:
        column_mapping[site_col] = 'sample_site'
    
    easting_col = find_column(df.columns, ['sample.samplingpoint.easting', 'easting'])
    if easting_col:
        column_mapping[easting_col] = 'sample_easting'
    
    northing_col = find_column(df.columns, ['sample.samplingpoint.northing', 'northing'])
    if northing_col:
        column_mapping[northing_col] = 'sample_northing'
    
    # Check if essential columns exist
    essential_mappings = {'sample_easting': easting_col, 'sample_northing': northing_col, 
                         'pollutant': pollutant_col, 'sample_datetime': datetime_col}
    missing_essentials = [key for key, val in essential_mappings.items() if val is None]
    
    if missing_essentials:
        raise ValueError(f"Missing essential columns in {filepath}: {missing_essentials}")
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Handle < and > qualifiers in results
    if 'result' in df.columns:
        df['result'] = df['result'].astype(str).str.replace('<', '').str.replace('>', '')
    
    # Parse datetime with better error handling
    if 'sample_datetime' in df.columns:
        df['sample_datetime'] = pd.to_datetime(df['sample_datetime'], errors='coerce', utc=True)
        # Convert to local time (assuming UK data)
        df['sample_datetime'] = df['sample_datetime'].dt.tz_convert('Europe/London').dt.tz_localize(None)
    
    # Validate and convert coordinates
    if 'sample_easting' in df.columns:
        df['sample_easting'] = pd.to_numeric(df['sample_easting'], errors='coerce')
    if 'sample_northing' in df.columns:
        df['sample_northing'] = pd.to_numeric(df['sample_northing'], errors='coerce')
    
    # Validate coordinate ranges
    initial_count = len(df)
    if 'sample_easting' in df.columns and 'sample_northing' in df.columns:
        df = df[df.apply(lambda row: validate_uk_coordinates(row['sample_easting'], row['sample_northing']), axis=1)]
        invalid_coords = initial_count - len(df)
        if invalid_coords > 0:
            print(f"  ⚠️  Filtered out {invalid_coords} samples with invalid coordinates")
    
    # Drop rows with missing essential data
    required_cols = ['pollutant', 'sample_easting', 'sample_northing', 'sample_datetime', 'result']
    existing_required = [col for col in required_cols if col in df.columns]
    df = df.dropna(subset=existing_required)
    
    return df


def process_year(year, edm_data, samples_data):
    """Correlate samples with overflows for a specific year."""
    print(f"\n--- Processing Year {year} ---")
    
    # Filter samples for target pollutants
    samples_filtered = samples_data[
        samples_data['pollutant'].str.contains(TARGET_POLLUTANTS_REGEX, case=False, na=False, regex=True)
    ]
    
    if samples_filtered.empty:
        print(f"  No target pollutant samples found")
        return pd.DataFrame()
    
    print(f"  EDM sites: {len(edm_data)}")
    print(f"  Relevant samples: {len(samples_filtered)}")
    
    if edm_data.empty:
        return pd.DataFrame()
    
    # Build spatial index
    overflow_coords = edm_data[['easting', 'northing']].to_numpy()
    sample_coords = samples_filtered[['sample_easting', 'sample_northing']].to_numpy()
    
    tree = KDTree(overflow_coords)
    
    # Find nearby overflows for each sample
    print(f"  Finding correlations within {RADIUS_METERS}m...")
    nearby_indices = tree.query_ball_point(sample_coords, r=RADIUS_METERS)
    
    # Build correlation records
    matches = []
    for sample_idx, overflow_indices in enumerate(nearby_indices):
        if overflow_indices:
            sample_row = samples_filtered.iloc[sample_idx]
            
            for overflow_idx in overflow_indices:
                overflow_row = edm_data.iloc[overflow_idx]
                
                # Calculate distance
                dist = np.linalg.norm(sample_coords[sample_idx] - overflow_coords[overflow_idx])
                
                # Check if dissolved oxygen
                is_do = bool(re.search(DO_SPECIFIC_REGEX, sample_row['pollutant'], re.IGNORECASE))
                
                # Normalize DO values if needed
                result_value = sample_row['result']
                if is_do and 'unit' in sample_row and pd.notna(sample_row.get('unit')):
                    normalized = normalize_do_value(result_value, sample_row['unit'])
                    if normalized is not None:
                        result_value = normalized
                
                matches.append({
                    'year': year,
                    'overflow_site': overflow_row['overflow_site'],
                    'sample_point': sample_row['sample_site'],
                    'sample_time': sample_row['sample_datetime'],
                    'pollutant': sample_row['pollutant'],
                    'pollutant_result': result_value,
                    'distance_m': max(int(dist), MIN_DISTANCE_M),
                    'is_dissolved_oxygen': is_do,
                    'overflow_spill_count': overflow_row['overflow_spill_count'],
                    'overflow_spill_duration_hrs': overflow_row['overflow_spill_duration_hrs']
                })
    
    print(f"  ✅ Found {len(matches)} correlations")
    return pd.DataFrame(matches)


# === MAIN EXECUTION ===

def main():
    print("="*60)
    print("WATER QUALITY CORRELATION ANALYSIS")
    print("="*60)
    
    # Load all EDM data
    edm_data = load_all_edm_data(EDM_DIR)
    
    # Process each year
    all_results = []
    
    print("\n=== Processing Sample Data ===")
    
    for filename in sorted(os.listdir(SAMPLE_DIR)):
        if not filename.endswith('.csv'):
            continue
        
        year = filename.replace('.csv', '')
        filepath = os.path.join(SAMPLE_DIR, filename)
        
        # Check if we have EDM data for this year
        edm_year_data = edm_data[edm_data['year'] == year]
        if edm_year_data.empty:
            print(f"\nYear {year}: No EDM data available, skipping")
            continue
        
        try:
            # Load samples
            samples = load_sample_data(filepath)
            
            # Process correlations
            year_results = process_year(year, edm_year_data, samples)
            
            if not year_results.empty:
                all_results.append(year_results)
                
        except Exception as e:
            print(f"\nYear {year}: ERROR - {e}")
    
    # Save results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df = final_df.sort_values(by=['year', 'overflow_site', 'sample_time'])
        final_df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS! Saved {len(final_df)} correlated events to '{OUTPUT_FILE}'")
        print(f"{'='*60}")
        
        # Summary statistics
        print("\n📊 SUMMARY STATISTICS:")
        print(f"  Total events: {len(final_df):,}")
        print(f"  Dissolved oxygen events: {final_df['is_dissolved_oxygen'].sum():,}")
        print(f"  Average distance: {final_df['distance_m'].mean():.0f}m")
        print(f"  Unique overflow sites: {final_df['overflow_site'].nunique()}")
        print(f"  Years analyzed: {sorted(final_df['year'].unique())}")
        
        # Breakdown by pollutant type
        print("\n📊 EVENTS BY POLLUTANT TYPE:")
        pollutant_counts = final_df['pollutant'].value_counts().head(10)
        for pollutant, count in pollutant_counts.items():
            print(f"  {pollutant}: {count:,}")
            
    else:
        print("\n⚠️  No correlated events found")


if __name__ == "__main__":
    main()
