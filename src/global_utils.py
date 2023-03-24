"""
Home for global functions used across the codebase 
"""

def get_concept_set_members(csm_path):
    """
    Returns a pandas dataframe with the concept_set_members table at the given path
    """
    pass


def get_index_range(index_range_path):
    pass


def rename_cols(df, prefix="", suffix=""):
    """
    Helper function to rename columns by adding a prefix or a suffix
    """
    index_cols = ["person_id", "before_or_after_index"]
    select_list = [col(col_name).alias(prefix + col_name + suffix) if col_name not in index_cols else col(col_name) for col_name in df.columns]
    df = df.select(select_list).drop(col("before_or_after_index"))
    return df
