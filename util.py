SAMPLING_RATE = 100  # Hz
# The leads used for the data
LEAD_LABELS = [
    'I', 'II', 'III',
    #'aVR', 'aVL', 'aVF',
    'V1', 'V2', 'V3',
    'V4', 'V5', 'V6',
]
TOTAL_NUM_LEADS = len(LEAD_LABELS)
# The number of ECGs in the dataset
TOTAL_NUM_ECGS = 7072#4368
# The database file to save ECG data to, will be prepended by `org_` or `rec_` depending on the data
DB_FILE = 'ecg_data.db'
# The name of the table in the db
TABLE_NAME = 'ecg_data'
# Make an inner directory in the output/ dir, useful if you're working
# on multiple projects at the same time
INNER_OUTPUT_DIR = 'other_project'


def df_column_name(column: str) -> str:
    """
    Pretty print the column name. I.e. 'r_peak_mean' is written as 'R-peak mean' since it's better in plot titles
    and terminal printing.
    :param column: The column
    :return: Pretty printed column name
    """
    match column:
        case 'r_peak_mean':
            return 'R-peak mean'
        case 't_mean':
            return 'T-peak mean'
        case 'p_mean':
            return 'P-peak mean'
        case 'q_mean':
            return 'Q-peak mean'
        case 's_mean':
            return 'S-peak mean'
        case 'rr_interval_mean':
            return 'RR-interval mean'
        case 'qt_interval_mean':
            return 'QT-interval mean'
        case 'pr_interval_mean':
            return 'PR-interval mean'
        case _:
            print(f'Error: {column} is not recognized, returning {column}')
            return column


def feature_columns(column: str=None) -> list[str] | str:
    """
    The columns that are considered features and used for plotting and statistics.
    :param column: Use to only get a specific column.
    :return: The list of columns or a specific column.
    """
    columns = [
        'r_peak_mean',
        't_mean',
        'p_mean',
        'q_mean',
        's_mean',
        'rr_interval_mean',
        'qt_interval_mean',
        'pr_interval_mean'
    ]
    if column is None:
        return columns
    if column not in columns:
        raise ValueError(f'Column {column} not in columns list')
    index = columns.index(column)
    return columns[index]