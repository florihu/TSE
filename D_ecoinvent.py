from ecoinvent_interface import Settings, EcoinventRelease, EcoinventProcess, ReleaseType, ProcessFileType, CachedStorage
from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm
import glob
import re


def download_data(username, password, version, system_model, output_path):
    # Set up your Ecoinvent credentials

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    my_settings = Settings(username=username, password=password, output_path=output_path)

    # Initialize the EcoinventRelease and choose the version and system model
    release = EcoinventRelease(my_settings)

    # Get the release
    release.get_release(version=version, system_model=system_model, release_type=ReleaseType.lci)


    return None


def relevant_activity_check(activity, reference_prod, kew, not_contain = ['market', 'treatment'], p_kews = ['production', 'primary', 'operation']):
    '''
    Check if the activity or reference product contains any of the keywords
    '''
    word_list =   re.split(r',\s*|\s+', activity) + re.split(r',\s*|\s+', reference_prod)

    if any(w in not_contain for w in word_list):
        return False, ""
    
    if any(p in word_list for p in p_kews) == False:
        return False, ""
    
    # Find the first matched keyword
    first_match = next((k for k in kew if (k in word_list)), "")

    return (bool(first_match), first_match)  # Return a tuple of (found, matched_keyword)


def get_mining_related_activity_id(lookup_path, kew, look_out_path):
    '''
    Loop through the lookup table and extract ids that include keywords
    '''
    lookup = pd.read_csv(lookup_path, delimiter=';')

    lookup[['Relevant', 'First_Matched_Keyword']] = lookup.apply(
        lambda x: relevant_activity_check(x['ActivityName'], x['ReferenceProduct'], kew), axis=1, result_type='expand'
    )

    lookup[lookup['Relevant']].to_csv(look_out_path, index=False)

    return None



if __name__ == '__main__':
    username = "LUCML"
    password = "ecoV3JG62,0"
    version = '3.10'
    system_model = 'cutoff'
    output_path = r'data\ecoinvent'

    look_out_path = r'data\ecoinvent\ecoinvent 3.10_cutoff_lci_ecoSpold02\mining_related_activity.csv'
    

    # act_key = [
    # 'mining', 'extraction', 'tailings', 'mineral', 'quarrying', 'beneficiation', 
    # 'smelting', 'refining', 'open-pit', 'underground', 'surface', 'leaching', 
    # 'flotation', 'blasting', 'crushing', 'concentration', 'drilling', 'overburden', 
    # 'waste', 'strip', 'dredging', 'sluicing', 'cyanidation', 'hydrometallurgy', 
    # 'pyrometallurgy', 'ore', 'closure', 'reclamation', 'run-of-mine', 'placer', 
    # 'recovery', 'stockpile', 'gangue', 'froth'
    # ]

    ref_key = [
    'aluminium', 'antimony', 'barium', 'beryllium', 'bismuth', 'cadmium', 
    'chromium', 'cobalt', 'copper', 'gallium', 'gold', 'indium', 'iron', 'lead', 
    'lithium', 'magnesium', 'manganese', 'mercury', 'molybdenum', 'nickel', 
    'niobium', 'palladium', 'platinum', 'rhenium', 'rhodium', 'silver', 
    'strontium', 'tantalum', 'tellurium', 'tin', 'titanium', 'tungsten', 
    'uranium', 'vanadium', 'zinc', 'zirconium', 'rare', 'earth', 'metal',
    'yttrium', 'scandium', 'lithium', 'cobalt', 'neodymium', 'praseodymium',
    'terbium', 'dysprosium', 'samarium', 'europium', 'gadolinium', 'holmium',
    'lutetium', 'thulium', 'ytterbium', 'erbium', 'cerium', 'lanthanum', 'ore'
    ]
    
    keyw = ref_key

    lpath = r'data\ecoinvent\ecoinvent 3.10_cutoff_lci_ecoSpold02\FilenameToActivityLookup.csv'

    get_mining_related_activity_id(lpath, keyw, look_out_path)
