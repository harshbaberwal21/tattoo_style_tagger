import numpy as np

def model_data_splitting(tattoos_meta_data_processed_augmented, val_perc=0.1, test_perc=0.1):
    """Split model data into training, validation & test sets.
    The function aims to maintain the split percentages for all data as well as
    for a particular style.
    Args:
        tattoos_meta_data_processed_augmented (pd.core.frame.DataFrame): Processed tattoos metadata
        with augmented data as well.
        val_perc (float, optional): Validation set percentage. Defaults to 0.1.
        test_perc (float, optional): Test set percentage. Defaults to 0.1.

    Returns:
        pd.core.frame.DataFrame: Tattoos metadata with a column example_type to indicate
        if the image is to be considered for train, validation or test
    """
    tattoos_meta_data_processed_augmented_filtered = tattoos_meta_data_processed_augmented[
        tattoos_meta_data_processed_augmented['to_process']
        ].copy()

    syle_tattoo_counts = tattoos_meta_data_processed_augmented.groupby('styles', as_index=False)['tattoo_id'].nunique()

    val_perc = test_perc = 0.1
    np_repr = np.random.RandomState(21)

    test_set_ids = set()
    val_set_ids = set()
    train_set_ids = set()

    for row in syle_tattoo_counts.sort_values('tattoo_id', ascending=True).iterrows():
        row_val = row[1]
        style_name = row_val['styles']
        tattoo_count_for_style = row_val['tattoo_id']

        val_set_size = int(np.ceil(tattoo_count_for_style * val_perc))
        test_set_size = int(np.ceil(tattoo_count_for_style * test_perc))
        
        tattoo_image_data_for_style = tattoos_meta_data_processed_augmented_filtered[tattoos_meta_data_processed_augmented_filtered['styles']==style_name].copy()
        tattoo_ids_for_style = tattoo_image_data_for_style[['tattoo_id']].drop_duplicates().reset_index(drop=True)
        
        val_set_size_left = val_set_size - tattoo_ids_for_style[tattoo_ids_for_style['tattoo_id'].isin(val_set_ids)].shape[0]
        test_set_size_left = test_set_size - tattoo_ids_for_style[tattoo_ids_for_style['tattoo_id'].isin(test_set_ids)].shape[0]
        
        tattoo_ids_for_style_left = tattoo_ids_for_style[
        (~tattoo_ids_for_style['tattoo_id'].isin(val_set_ids)) &
        (~tattoo_ids_for_style['tattoo_id'].isin(train_set_ids)) &
        (~tattoo_ids_for_style['tattoo_id'].isin(test_set_ids))
        ].copy()
        
        if test_set_size_left < 0:
            test_set_size_left = 0 

        if val_set_size_left < 0:
            val_set_size_left = 0
            
        style_val_ids = np_repr.choice(tattoo_ids_for_style_left['tattoo_id'].values, size = val_set_size_left, replace=False)
        tattoo_ids_for_style_left_after_val = tattoo_ids_for_style_left[~tattoo_ids_for_style_left['tattoo_id'].isin(style_val_ids)]
        style_test_ids = np_repr.choice(tattoo_ids_for_style_left_after_val['tattoo_id'].values, size = test_set_size_left, replace=False)


        val_set_ids = val_set_ids.union(set(style_val_ids))
        test_set_ids = test_set_ids.union(set(style_test_ids))

        train_set_ids = train_set_ids.union(
            set(tattoo_ids_for_style_left['tattoo_id'].values).difference(set(style_val_ids)).difference(set(style_test_ids))
            )


    tattoos_meta_data_processed_augmented.loc[:,'example_type'] = 'train'
    tattoos_meta_data_processed_augmented.loc[tattoos_meta_data_processed_augmented['tattoo_id'].isin(test_set_ids),'example_type'] = 'test'

    tattoos_meta_data_processed_augmented.loc[tattoos_meta_data_processed_augmented['tattoo_id'].isin(val_set_ids),'example_type'] = 'val'

    return tattoos_meta_data_processed_augmented

if __name__ == "__main__":
    pass
