import numpy as np
import pandas as pd

np.set_printoptions(suppress=True) # prevent numpy exponential notation on print (remove eng notation in matrices)
#pd.set_option('display.max_rows', 40)


rawdata = pd.read_csv("dataset revisedpavetype additionalparameters + weather.csv")  # Import raw data

data = rawdata
data = data.groupby(["SECNUM", "YEAR"]).first()

activity_rows = data

sec_idx = activity_rows.index.get_level_values(0)
next_year_idx = activity_rows.index.get_level_values(1) + 1
prev_year_idx = activity_rows.index.get_level_values(1) - 1
exist_idx = activity_rows.index
after_idx = pd.MultiIndex.from_tuples(list(zip(sec_idx, next_year_idx)))
before_idx = pd.MultiIndex.from_tuples(list(zip(sec_idx, prev_year_idx)))
activity_rows = activity_rows.reindex(activity_rows.index.union(after_idx).union(before_idx))
activity_rows.loc[exist_idx, "NEXT_RQI"] = activity_rows.loc[after_idx, "RQI"].values
activity_rows.loc[exist_idx, "PREV_RQI"] = activity_rows.loc[before_idx, "RQI"].values

#---------------------------------------- Excluding the rows with missing data
activity_rows.dropna(subset=["RQI", "NEXT_RQI", "PREV_RQI"], inplace=True)
valid_rows = activity_rows

valid_rows.loc[(valid_rows["RQI"] <= 5) & (valid_rows["RQI"] >= 4.1), "STATE"] = 1
valid_rows.loc[(valid_rows["RQI"] < 4.1) & (valid_rows["RQI"] >= 3.1), "STATE"] = 2
valid_rows.loc[(valid_rows["RQI"] < 3.1) & (valid_rows["RQI"] >= 2.1), "STATE"] = 3
valid_rows.loc[(valid_rows["RQI"] < 2.1) & (valid_rows["RQI"] >= 1.1), "STATE"] = 4
valid_rows.loc[(valid_rows["RQI"] < 1.1) & (valid_rows["RQI"] >= 0), "STATE"] = 5
valid_rows.dropna(subset=["STATE"], inplace=True)
valid_rows["STATE"] = valid_rows["STATE"].astype(int)

valid_rows.loc[(valid_rows["NEXT_RQI"] <= 5) & (valid_rows["NEXT_RQI"] >= 4.1), "NEXT_STATE"] = 1
valid_rows.loc[(valid_rows["NEXT_RQI"] < 4.1) & (valid_rows["NEXT_RQI"] >= 3.1), "NEXT_STATE"] = 2
valid_rows.loc[(valid_rows["NEXT_RQI"] < 3.1) & (valid_rows["NEXT_RQI"] >= 2.1), "NEXT_STATE"] = 3
valid_rows.loc[(valid_rows["NEXT_RQI"] < 2.1) & (valid_rows["NEXT_RQI"] >= 1.1), "NEXT_STATE"] = 4
valid_rows.loc[(valid_rows["NEXT_RQI"] < 1.1) & (valid_rows["NEXT_RQI"] >= 0), "NEXT_STATE"] = 5
valid_rows.dropna(subset=["NEXT_STATE"], inplace=True)
valid_rows["NEXT_STATE"] = valid_rows["NEXT_STATE"].astype(int)

valid_rows.loc[(valid_rows["PREV_RQI"] <= 5) & (valid_rows["PREV_RQI"] >= 4.1), "PREV_STATE"] = 1
valid_rows.loc[(valid_rows["PREV_RQI"] < 4.1) & (valid_rows["PREV_RQI"] >= 3.1), "PREV_STATE"] = 2
valid_rows.loc[(valid_rows["PREV_RQI"] < 3.1) & (valid_rows["PREV_RQI"] >= 2.1), "PREV_STATE"] = 3
valid_rows.loc[(valid_rows["PREV_RQI"] < 2.1) & (valid_rows["PREV_RQI"] >= 1.1), "PREV_STATE"] = 4
valid_rows.loc[(valid_rows["PREV_RQI"] < 1.1) & (valid_rows["PREV_RQI"] >= 0), "PREV_STATE"] = 5
valid_rows.dropna(subset=["PREV_STATE"], inplace=True)
valid_rows["PREV_STATE"] = valid_rows["PREV_STATE"].astype(int)


valid_rows.loc[valid_rows["ACTIVTY"].notnull(), "END_STATE"] = valid_rows["NEXT_STATE"].astype(int)

valid_rows.loc[(valid_rows["ACTIVTY"].notnull()) & (valid_rows["PREV_STATE"] > valid_rows["STATE"]), "START_STATE"] = valid_rows["PREV_STATE"].astype(int)
valid_rows.loc[(valid_rows["ACTIVTY"].notnull()) & (valid_rows["PREV_STATE"] <= valid_rows["STATE"]), "START_STATE"] = valid_rows["STATE"].astype(int)

valid_rows.loc[valid_rows["ACTIVTY"].isnull(), "START_STATE"] = valid_rows["STATE"].astype(int)
valid_rows.loc[valid_rows["ACTIVTY"].isnull(), "END_STATE"] = valid_rows["NEXT_STATE"].astype(int)


valid_rows_clean = valid_rows
noise = valid_rows_clean.loc[(valid_rows_clean["ACTIVTY"].notnull())& ((valid_rows_clean["START_STATE"]) < (valid_rows_clean["END_STATE"]))] #Finding rows
valid_rows_clean = valid_rows_clean.drop(valid_rows_clean[(valid_rows_clean["ACTIVTY"].notnull())& ((valid_rows_clean["START_STATE"]) < (valid_rows_clean["END_STATE"]))].index) # excluding sections that got worse after maintenance
valid_rows_clean = valid_rows_clean.drop(valid_rows_clean[(valid_rows_clean["ACTIVTY"].isnull())& ((valid_rows_clean["START_STATE"]) > (valid_rows_clean["END_STATE"]))].index) #excluding sections that improved without maintenance

#------------------------------------- Deleting data from sections that jumped states
valid_rows_clean=valid_rows_clean.drop(valid_rows_clean[(valid_rows_clean["START_STATE"] == 1) & (valid_rows_clean["END_STATE"] == 3)].index)
valid_rows_clean=valid_rows_clean.drop(valid_rows_clean[(valid_rows_clean["START_STATE"] == 1) & (valid_rows_clean["END_STATE"] == 4)].index)
valid_rows_clean=valid_rows_clean.drop(valid_rows_clean[(valid_rows_clean["START_STATE"] == 1) & (valid_rows_clean["END_STATE"] == 5)].index)
valid_rows_clean=valid_rows_clean.drop(valid_rows_clean[(valid_rows_clean["START_STATE"] == 2) & (valid_rows_clean["END_STATE"] == 4)].index)
valid_rows_clean=valid_rows_clean.drop(valid_rows_clean[(valid_rows_clean["START_STATE"] == 2) & (valid_rows_clean["END_STATE"] == 5)].index)
valid_rows_clean=valid_rows_clean.drop(valid_rows_clean[(valid_rows_clean["START_STATE"] == 3) & (valid_rows_clean["END_STATE"] == 5)].index)


#--------------------------------------------Separating phases accordingto maintenance history
#pd.set_option('display.max_rows', None)

workdf = valid_rows_clean
change_idx = workdf["START_STATE"] > workdf["END_STATE"]
workdf.loc[~change_idx, "WORK"] = 0
workdf.loc[change_idx, "WORK"] = 1
workdf["ACTIVTY"] = workdf["ACTIVTY"].fillna("")


workdf = workdf.reset_index()
workdf = workdf.rename(columns={"level_0":"SECNUM","level_1":"YEAR"})


#Excluding activities that didn't cause improvement to increase precision.
idx_drop = workdf[(workdf["START_STATE"] <= workdf["END_STATE"]) & (workdf["ACTIVTY"].notnull())].index
workdf.loc[idx_drop, "ACTIVTY"] = ""

work_columns = ["SECNUM", "YEAR", "ACTIVTY", "WORK"]

work_col = workdf[work_columns]

work_mult = work_col.set_index(["SECNUM", "YEAR"])
work_cum = work_mult.groupby(level=0).cumsum()



events = workdf.groupby(['SECNUM'])['ACTIVTY'].apply(lambda items: [x for x in items if x])
work_cum["ACTIVTY"] = pd.Series(work_cum.index.get_level_values(0)).map(events).values

def map_idx(row):
    if row["WORK"] == 0:
        return np.nan
    return row["ACTIVTY"][row["WORK"] - 1]

work_cum["WORK"] = work_cum["WORK"].astype(int)
work_cum["Phase"] = work_cum.apply(map_idx, axis=1)
work_cum = work_cum.rename(columns={"WORK":"HISTORY", "ACTIVTY":"PAST"})


valid_rows_clean = valid_rows_clean.reset_index()
valid_rows_clean = valid_rows_clean.rename(columns={"level_0":"SECNUM","level_1":"YEAR"})
work_join = valid_rows_clean.set_index(["SECNUM", "YEAR"])
#jointdf = pd.concat(pd.DataFrame(i) for i in (work_join, work_cum))
jointdf = pd.concat([pd.DataFrame(i) for i in (work_join, work_cum)],sort=True)
jointdf = jointdf.groupby(["SECNUM", "YEAR"]).first()

#---------------------------------------------------Choosing sections with the same history 
data2001 = jointdf.reset_index()
data2001 = data2001.loc[data2001["YEAR"] >= 2001] #Use data from 2001
state2_BAB = data2001.loc[(data2001["REVISED PAVE"] == "BAB") & (data2001["STATE"] == 2)] #BOB and State 2
#homog_phase = state2_BOB.loc[state2_BOB["Phase"] == "Med M&OL"] #Sections that have received Medium mill & overlay
homog_phase = state2_BAB.groupby(["SECNUM", "YEAR"]).first()



#---------------------------------------------------Creating the trnasition column
transition_nat = homog_phase[homog_phase["ACTIVTY"]==""] #Rmove all possible effect from repait (Q)
transition_nat.loc[(transition_nat["START_STATE"] < transition_nat["END_STATE"]), "TRANSITION"] = 1
transition_nat.loc[(transition_nat["START_STATE"] == transition_nat["END_STATE"]), "TRANSITION"] = 0
transition_nat["TRANSITION"] = transition_nat["TRANSITION"].astype(int)


#---------------------------------------------------put it in the logistic regression format
data_reg = transition_nat
col2 = ["TRANSITION", "DISTRICT", "AADTA", "REVISED PAVE", "STATE", "PCT_TRUCKA", "ANNUAL ESAL", "FUNCTIONAL CLASS", "SPEED LIMIT", "COUNTY", "CONCRETE THK", "AC THK", "SURFACE THK", "Min Temperature", "Precipitation"]
data_reg = data_reg[col2] #use only relevant columns
data_reg_ind = data_reg.reset_index() #return SECNUM and YEAR back to columns (from indices)
col3 = ["TRANSITION", "SECNUM", "YEAR", "DISTRICT", "AADTA", "REVISED PAVE", "STATE", "PCT_TRUCKA", "ANNUAL ESAL", "FUNCTIONAL CLASS", "SPEED LIMIT", "COUNTY", "CONCRETE THK", "AC THK", "SURFACE THK", "Min Temperature", "Precipitation"]  #had to remove pavement type because it is a non integer variable
#col3 = ["TRANSITION", "DISTRICT", "AADTA", "STATE", "YEAR", "SECNUM"]#remove SECNUM and YEAR columns 
data_reg_ind = data_reg_ind[col3] 

reg_data = data_reg_ind

#creating the constant term (column with 1's)
reg_data.loc[reg_data["SECNUM"] >= 0, "Constant"] = 1
reg_data["Constant"] = reg_data["Constant"].astype(int)


#Only relevant columns
col4 = ["TRANSITION", "SECNUM", "AADTA", "YEAR", "DISTRICT", "PCT_TRUCKA", "ANNUAL ESAL", "FUNCTIONAL CLASS", "SPEED LIMIT", "COUNTY", "CONCRETE THK", "AC THK", "SURFACE THK", "Min Temperature", "Precipitation"]
reg_data_final = reg_data[col4]
reg_data_final = reg_data_final.rename(columns={"Min Temperature":"MIN TEMP", "Precipitation":"PRECIPITATION"})

#Drop missing AC
idx = reg_data_final[reg_data_final["AC THK"]==0].index
final_df = reg_data_final.drop(reg_data_final.index[idx])
final_df = final_df.reset_index()
final_df = final_df.drop("index", axis=1)

final_df.to_csv('transition data for regression - BAB - state 2 + secnum.csv')

#--------------------------------------------------Overdispersion format


overd = transition_nat
sum_tran = overd.groupby(["SECNUM","TRANSITION"])["TRANSITION"].count()
new_index = pd.MultiIndex.from_product(sum_tran.index.levels)
new_sum_tran=sum_tran.reindex(new_index)
new_sum_tran=new_sum_tran.fillna(0).astype(int)

no_tran = new_sum_tran.loc[:,0]
state_tran = new_sum_tran.loc[:,1]
total_tran = no_tran + state_tran

no_tran = no_tran.reset_index()
no_tran = no_tran.rename(columns={"index":"SECNUM","TRANSITION":"NO TRANSITION"})
no_tran = no_tran.groupby("SECNUM").first()
state_tran = state_tran.reset_index()
state_tran = state_tran.rename(columns={"index":"SECNUM","TRANSITION":"STATE TRANSITION"})
state_tran = state_tran.groupby("SECNUM").first()
total_tran = total_tran.reset_index()
total_tran = total_tran.rename(columns={"index":"SECNUM","TRANSITION":"TOTAL TRANSITION"})
total_tran = total_tran.groupby("SECNUM").first()
overd_data = pd.concat([pd.DataFrame(i) for i in (total_tran, no_tran, state_tran)], axis=1)

#overd_data.to_csv('overdispersion data - BAB - state 2.csv')

