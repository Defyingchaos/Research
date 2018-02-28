import json
import gspread
import oauth2client
from oauth2client.client import SignedJwtAssertionCredentials

json_key = json.load(open('creds.json')) 
scope = ['https://spreadsheets.google.com/feeds']

credentials = SignedJwtAssertionCredentials(json_key['client_email'], json_key['private_key'].encode(), scope)

file = gspread.authorize(credentials)
sheet = file.open('Data').sheet1

sheet.update_acell('A1', 'Dataset Size')
#print(sheet.row_count)
#print(sheet.col_values(1))
print(sheet.row_values(1))

# newrow = 0
# i="word"
# r=1
# c = True
# while c==True:
#     i=sheet.row_values(r)
#     if len(i)==0:
#         c=False
#     r+=1
#     newrow = r

# sheet.update_acell(newrow, 1, "test word")

def nar(sheet):
    str_list = list(filter(None, sheet.col_values(1)))
    return str(len(str_list)+1)
print(nar(sheet))