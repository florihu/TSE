
from D_sp_data_clean import get_data, var_exp_path


def mine_ids_per_commodity(site, target_commodities):
    """
    This function identifies the IDs of mines that contain one or more of the target commodities.

    Parameters:
        site (pd.DataFrame): DataFrame containing columns 'PROP_ID' and 'Commodities_list'.
        target_commodities (set): Set of target commodities to filter for.

    Returns:
        dict: Dictionary of sets containing the IDs of mines that contain each target commodity.
        list: List of unique IDs of mines that contain at least one target commodity.

    """
    # Ensure target_commodities is a set for efficient lookups
    target_commodities = set(target_commodities)

    def com_check(x):
        return any(com in target_commodities for com in x.split(', '))

    # Filter rows where at least one target commodity is present
    site['Matching_Commodities'] = site['Commodities_list'].apply(
        lambda x: com_check(x) if isinstance(x, str) else False
    )

    # Keep only rows with matching commodities and explode the list to individual rows
    filtered = site[site['Matching_Commodities'].str.len() > 0].explode('Matching_Commodities')

    # Create dictionary of sets for each commodity
    by_com = filtered.groupby('Matching_Commodities')['PROP_ID'].apply(set).to_dict()

    unique_ids = list(set().union(*by_com.values()))

    # get all coms that are mined together with the target commodities
    all_coms = site[site.PROP_ID.isin(unique_ids)]['Commodities_list'].explode().unique()

    return by_com, unique_ids, all_coms




# Define target commodities as a set
target_commodities = {'Copper', 'Zinc', 'Nickel'}

if __name__ == '__main__':
    site = get_data('site')  # Assuming get_data is a function that loads your DataFrame

    # Pass target_commodities as an argument to the function
    com_id_dict = mine_ids_per_commodity(site, target_commodities)
    print(com_id_dict)
