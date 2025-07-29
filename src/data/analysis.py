import pandas as pd

class Top3Analyser:
    def __init__(self, data):
        # Initialize analyser with master data
        self.data = data
        self.results = {}

    def analyse(self, group_col, name_col=None, year_from=None, sort_by=None, label=None):
        df = self.data.copy()
        if year_from:
            df = df[df['year'] >= year_from]

        result = (
            df.groupby(group_col)
            .agg(
                top3_count=('top3', 'sum'),
                total_races=('raceId', 'count')
            )
            .assign(top3_percent=lambda d: 100 * d['top3_count'] / d['total_races'])
            .sort_values(by='top3_percent' if sort_by == 'percent' else 'top3_count', ascending=False)
            .reset_index()
        )

        if name_col:
            if name_col == "full_name":
                df["full_name"] = df["forename"] + " " + df["surname"]
            if name_col in df.columns:
                names = df[[group_col, name_col]].drop_duplicates(subset=group_col)
                result = result.merge(names, on=group_col, how='left')

        if label:
            self.results[label] = result

        return result

    def get_results(self):
        # Return stored results
        return self.results