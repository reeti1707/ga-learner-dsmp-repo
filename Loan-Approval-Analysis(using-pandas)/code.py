# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 
bank = pd.read_csv(path)

categorical_var = bank.select_dtypes(include = 'object')
# code starts here

print('categorical_var =',categorical_var)

numerical_var = bank.select_dtypes(include = 'number')
print(numerical_var)


# code ends here


# --------------
# code starts here

banks = bank.drop("Loan_ID",axis = 1)
#code ends 
null_1 = banks.isnull().sum()
print(null_1)
bank_mode =banks.mode().iloc[0]
print('bank_mode is ',bank)
banks.fillna(value = bank_mode,inplace = True)




# --------------
# Code starts here
#banks = bank.drop('Loan_ID',axis = 1)

avg_loan_amount = pd.pivot_table(banks,values ='LoanAmount', index = ['Gender', 'Married', 'Self_Employed'],aggfunc=np.mean)

# code ends here
avg_loan_amount


# --------------
# code starts here

loan_approved_se = len(banks[(banks['Self_Employed']== 'Yes') & (banks['Loan_Status']== 'Y')])
loan_approved_nse = len(banks[(banks['Self_Employed']== 'No') & (banks['Loan_Status']== 'Y')])

loan_count = len(banks['Loan_Status'])
print('loan_approved_se',loan_approved_se)
print('loan_approved_nse',loan_approved_nse)
print('loan_count',loan_count)
# code ends here
percentage_se = (loan_approved_se/loan_count) * 100
#print('percentage of loan approval',percentage_se)
print('percentage_se', percentage_se)
percentage_nse = (loan_approved_nse/loan_count) * 100
print('percentage_nse',percentage_nse)


# --------------
# code starts here


loan_term = banks['Loan_Amount_Term'].apply(lambda x : x/12)
print(len(loan_term))
big_loan_term = len(banks[loan_term >= 25])


print(big_loan_term)




# code ends here


# --------------
# code starts here
loan_groupby = banks.groupby(['Loan_Status'])

loan_groupby = loan_groupby[['ApplicantIncome', 'Credit_History']]

print(loan_groupby)
mean_values = loan_groupby.agg([np.mean])
print(mean_values)


# code ends here


