# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data= pd.read_csv(path)

loan_status = data['Loan_Status'].value_counts()
print(loan_status)
#Code starts heretype_2.plot(kind='bar')
loan_status.plot(kind ='bar')


# --------------
#Code starts here

property_and_loan = data.groupby(['Property_Area','Loan_Status'])
property_and_loan = property_and_loan.size().unstack()
property_and_loan.plot(kind='bar')
plt.xlabel('Property Area')
plt.ylabel('Loan Status')
plt.xticks(rotation='45')


# --------------
#Code starts here
education_and_loan = data.groupby(['Education','Loan_Status'])
education_and_loan = education_and_loan.size().unstack()
education_and_loan.plot(kind = 'bar')
plt.xlabel('Educattion Status')
plt.xticks(rotation='45')



# --------------
#Code starts here


graduate = data[data['Education']=="Graduate"]
not_graduate = data[data['Education']=="Not Graduate"]

graduate['LoanAmount'].plot(kind='density',label='Graduate')


#not_graduate.LoanAmount.

not_graduate['LoanAmount'].plot(kind='density',label='Not Graduate')




#Code ends here

#For automatic legend display
plt.legend()


# --------------
#Code starts here

fig,(ax_1,ax_2,ax_3) = plt.subplots(nrows = 3 , ncols = 1)
# electric.plot.scatter('HP','Attack')dat
# res.plot(kind='bar', stacked=True, ax=ax_1)

data.plot.scatter('ApplicantIncome','LoanAmount',ax= ax_1)
ax_1.set_title('Applicant Income')
data.plot.scatter('CoapplicantIncome','LoanAmount',ax= ax_2)
ax_2.set_title('Coapplicant Income')

data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
print(data.head(3))
data.plot.scatter('TotalIncome','LoanAmount',ax= ax_3)
ax_3.set_title('Total Income')
#df['C'] =  df[['A', 'B']].sum(axis=1)


