def df_column_name(column: str) -> str:
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
            return column

def feature_columns(column: str=None) -> list[str] | str:
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