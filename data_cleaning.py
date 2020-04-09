import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

class DataPreparation:


    def __init__(self, 
                     attribute_values,
                     imputer,
                     scaler,
                     final_cols):
        """
        Data Preparation, including:
        * Deal with "Unknown" values
        * Deal with missing values
        * Remove "fine" categorical variables
        * Decide which columns will need one-hot-encoding or extra engineering
        * Make sure that all columns are numerical
        * Fill missing values and scale the data
        
        params: 
        attribute_values: the table with the description of the attributes ("attribute_values_cleaned_checkpoint.csv")
        imputer: the fitted imputer
        scaler: the fitted scaler
        final_cols: final columns
        """
        self.attribute_values = attribute_values
        self.imputer = imputer
        self.scaler = scaler
        self.final_cols = final_cols

    @staticmethod
    def code_unknown_to_nan(data, attribute_values):
        """
        Code numbers that represent "unknown" values to NaN
        """
        attribute_values_unknown = attribute_values[attribute_values['Meaning'] == "unknown"]
        for i in range(len(attribute_values_unknown)):
            colname = attribute_values_unknown.iloc[i]['Attribute']
            unknown_values = eval('[' + str(attribute_values_unknown.iloc[i]['Value']) + ']')
            try:
                data[colname] = data[colname].replace(unknown_values, float('nan'))
            except:
                pass
        return data


    @staticmethod
    def remove_cols_high_missing_rates(data, min_missing_rate=0.4):
        """
        Remove columns with missing rates more than min_missing_rate
        """
        cols_keep = list(data.isna().mean()[data.isna().mean() < min_missing_rate].index)
        return data[cols_keep], cols_keep


    @staticmethod
    def remove_fine_cols(data):
        cols_keep = [i for i in list(data.columns) if not i.endswith("_FEIN")]
        data = data[cols_keep]
        return data, cols_keep


    @staticmethod
    def label_to_meaning(attribute, attribute_values):
        """
        Create a dictionary to map attribute labels to meanings
        """
        attribute_values_1 = attribute_values[attribute_values['Attribute'] == attribute]
        result = {}
        for i in range(len(attribute_values_1)):
            result[attribute_values_1.iloc[i]["Value"]] = attribute_values_1.iloc[i]["Meaning"]
        return result


    def engineer_cat_cols(self, data, attribute_values):
        """
        Perform ad-hoc transformation on selected attributes:
        """
        data['CAMEO_DEU_2015'] = data['CAMEO_DEU_2015']\
                                                 .replace("XX", float('nan'))\
                                                 .apply(lambda x: x if str(x) == "nan" else str(x)[0])
        data['CAMEO_DEUG_2015'] = data['CAMEO_DEUG_2015'].replace("X", np.float('nan')).astype("float")
        data['CAMEO_INTL_2015'] = data['CAMEO_INTL_2015'].replace("XX", np.float('nan')).astype("float")
        
        def custom_split1(x):
            x = str(x)
            try:
                splits1 = x.split(" - ")
                splits2 = splits1[1].split(" (")
                splits3 = splits2[1].split(", ")
                return [splits1[0], splits3[0]]
            except:
                return x
        
        PRAEGENDE_JUGENDJAHRE_dict = self.label_to_meaning("PRAEGENDE_JUGENDJAHRE", attribute_values)
        PRAEGENDE_JUGENDJAHRE_temp = data['PRAEGENDE_JUGENDJAHRE']\
                                     .apply(lambda x: PRAEGENDE_JUGENDJAHRE_dict[str(int(x))] if str(x) != "nan" else x)
        data['PRAEGENDE_JUGENDJAHRE_part1'] = PRAEGENDE_JUGENDJAHRE_temp\
                                              .apply(lambda x: custom_split1(x)[0] if str(x) != "nan" else x)
        data['PRAEGENDE_JUGENDJAHRE_part2'] = PRAEGENDE_JUGENDJAHRE_temp\
                                              .apply(lambda x: custom_split1(x)[1] if str(x) != "nan" else x)
        data = data\
               .join(pd.get_dummies(data[['PRAEGENDE_JUGENDJAHRE_part1','PRAEGENDE_JUGENDJAHRE_part2']]))\
               .drop(['PRAEGENDE_JUGENDJAHRE','PRAEGENDE_JUGENDJAHRE_part1','PRAEGENDE_JUGENDJAHRE_part2'], axis=1)
        
        one_hot_encode_cols = ['CJT_GESAMTTYP',
                               'D19_KONSUMTYP',
                               'GFK_URLAUBERTYP',
                               'HEALTH_TYP',
                               'LP_LEBENSPHASE_GROB',
                               'SHOPPER_TYP',
                               'ZABEOTYP']
        
        
        data = data\
               .join(pd.get_dummies(data[one_hot_encode_cols].astype("object"), dummy_na=False))\
               .drop(one_hot_encode_cols, axis=1)
        
        return data


    def pre_fit(self, data):

        data = self.code_unknown_to_nan(data, self.attribute_values)
        data, cols_keep1 = self.remove_cols_high_missing_rates(data)
        data, cols_keep2 = self.remove_fine_cols(data)
        data  = self.engineer_cat_cols(data, self.attribute_values)
        data = data[self.final_cols]
        
        self.index = data.index

        return data


    def fit(self, data):

        data = self.pre_fit(data)
        self.imputer = Imputer(missing_values=float('nan'), strategy="mean", axis=0).fit(data)
        data = self.imputer.transform(data)
        self.scaler = StandardScaler().fit(data)


    def transform(self, new_data):

        new_data = self.pre_fit(new_data)
        
        self.colnames = new_data.columns
        self.index = new_data.index
        
        new_data = self.imputer.transform(new_data)
        new_data = self.scaler.transform(new_data)

        return new_data