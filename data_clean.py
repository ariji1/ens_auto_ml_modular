from datacleaner import autoclean
import ppscore as pps

def clean_data(df):
    my_clean_data = autoclean(df.copy())
    return my_clean_data
def pred_power(clean_df,pred_column):
    pps_df = pps.predictors(clean_df, y=pred_column,cross_validation=4,random_seed=123)
    # pps_df[pps_df['ppscore']>0]
    return pps_df.sort_values(by="model_score",axis=0,ascending=True),pps_df.sort_values(by="model_score",axis=0,ascending=True)['x'][:5].values

