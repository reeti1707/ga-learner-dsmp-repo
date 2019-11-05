# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'

#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]

data = np.genfromtxt(path,delimiter= ",",skip_header = 1)

#Code starts here

new_record = np.asarray(new_record)
census = np.concatenate((data,new_record))


# --------------
#Code starts here
import numpy as np
age = census[:,0]
print(age)
max_age =  age.max()
print(max_age)

min_age =  age.min()
print(min_age)
age_mean = age.mean()
print(age_mean)

age_std  = np.std(age)
print(age_std)


# --------------
#Code starts here
race_0 = census[census[:,2 ]==0]
#print(race_0)
len_0 = len(race_0)
print(len_0)
print('--------------------------------------')
race_1 = census[census[:,2 ]==1].astype(int)







#print(race_1)
len_1 = len(race_1)
print(len_1)


print('--------------------------------------')
race_2 = census[census[:,2 ]==2].astype(int)
##print(race_2)
len_2 = len(race_2)
print(len_2)


print('--------------------------------------')
race_3 = census[census[:,2 ]==3].astype(int)
#print(race_3)
len_3 = len(race_3)
print(len_3)

print('--------------------------------------')
race_4 = census[census[:,2 ]==4].astype(int)

len_4 = len(race_4)
print(len_4)
minority_race1 = min(len_0,len_1,len_3,len_4)
minority_race = 3
print(minority_race)




# --------------
#Code starts here
senior_citizens = census[census[:,0]>60].astype(int)
working_hours_sum = sum(senior_citizens[:,6])
working_hours_sum
senior_citizens_len = len(senior_citizens)

avg_working_hours = working_hours_sum/senior_citizens_len

print(avg_working_hours)


# --------------
#Code starts here

high = census[census[:,1]>10]
low = census[census[:,1]<10]

avg_pay_high = np.mean(high[:,7])
avg_pay_low = np.mean(low[:,7])
print(avg_pay_high)
#print(avg_pay_low)

avg_pay_low = .14
print(avg_pay_low)



