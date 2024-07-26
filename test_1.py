'''import the necessary library to read a dataset from web site in this case requests'''
import requests
'''read your data set froma the target website in this case,"https://api.github.com/users/naveenkrnl"'''
focus_url="https://api.github.com/users/naveenkrnl"
'''retrieve content from the url using (requests.get).'''
i_want =requests.get(focus_url)
naveen_contents= i_want.text
'''show the content of the file'''
print(naveen_contents)

