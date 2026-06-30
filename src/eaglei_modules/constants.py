# ------------------------- Pre-defined Constants ------------------------- #
# This file contains pre-defined constants used across the EAGLEi Outage Analysis module.


# Path to the data directories
import os
EAGLEI_DATA_DIR = os.path.join('eagle-idatasets')
PROCESSED_DATA_DIR = os.path.join('processed_data')
OUTAGE_DATA_DIR = os.path.join('outage_data')    # to store cleaned eagle-i outage data
RESULTS_DIR = os.path.join('results')
MISC_DIR = os.path.join('misc')

# column names for the EAGLEi outage data
CUSTOMERS_COL = 'customers_out'
TIMESTAMP_COL = 'run_start_time'
COUNTY_COL = 'county'
STATE_COL = 'state'
YEAR_COL = 'year'

# Define colors for the curves
COLOR_OUTAGE_CURVE = 'blue'
COLOR_RESTORE_CURVE = 'red'
COLOR_PERFORMANCE_CURVE = 'orange'

# create a dictionary to store state names and their FIPS codes
STATE_FIPS_DICT = {
    'alabama': '01',
    'alaska': '02',
    'arizona': '04',
    'arkansas': '05',
    'california': '06',
    'colorado': '08',
    'connecticut': '09',
    'delaware': '10',
    'florida': '12',
    'georgia': '13',
    'hawaii': '15',
    'idaho': '16',
    'illinois': '17',
    'indiana': '18',
    'iowa': '19',
    'kansas': '20',
    'kentucky': '21',
    'louisiana': '22',
    'maine': '23',
    'maryland': '24',
    'massachusetts': '25',
    'michigan': '26',
    'minnesota': '27',
    'mississippi': '28',
    'missouri': '29',
    'montana': '30',
    'nebraska': '31',
    'nevada': '32',
    'new hampshire': '33',
    'new jersey': '34',
    'new mexico': '35',
    'new york': '36',
    'north carolina': '37',
    'north dakota': '38',
    'ohio': '39',
    'oklahoma': '40',
    'oregon': '41',
    'pennsylvania': '42',
    'rhode island': '44',
    'south carolina': '45',
    'south dakota': '46',
    'tennessee': '47',
    'texas': '48',
    'utah': '49',
    'vermont': '50',
    'virginia': '51',
    'washington': '53',
    'west virginia': '54',
    'wisconsin': '55',
    'wyoming': '56'
}